# inference.py — Toggle-window realtime inference for BiLSTM(+Attention)
# Flow:
#   IDLE → [Space] → RECORDING (no prediction) → [Space] → IDLE + ONE prediction from the recorded window
# Keys:
#   Space = start/stop a detection window
#   r     = reset current recording (when recording) / clear last prediction (when idle)
#   d     = toggle landmark drawing
#   q     = quit
#
# Run (example):
#   py inference.py --data E:/INCLUDE_KEYPOINTS --ckpt E:/INCLUDE_RUN_BiLSTM/ckpt_best.pt --cam 0 --amp --width 1280 --height 720 --hol_complex 2 --hands_fallback

import os, time, json, argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

SEQ_LEN  = 200
FEAT_DIM = 258
DTYPE    = np.float32

POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
L_SLICE  = slice(POSE_DIM, POSE_DIM + HAND_DIM)                 # 132:195
R_SLICE  = slice(POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM)  # 195:258

# ---------------- Model ----------------
class BiLSTMAttn(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEAT_DIM, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True, dropout=(dropout if layers > 1 else 0.0),
        )
        self.attn_W = nn.Linear(2 * hidden, 2 * hidden)
        self.attn_v = nn.Linear(2 * hidden, 1, bias=False)
        self.fc = nn.Linear(2 * hidden, 128)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,Tmax,2H)

        B, Tmax, _ = H.shape
        dev = H.device
        Ld = lengths.to(dev)
        mask = torch.arange(Tmax, device=dev).unsqueeze(0) < Ld.unsqueeze(1)

        scores = self.attn_v(torch.tanh(self.attn_W(H))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha  = torch.softmax(scores, dim=1)
        ctx    = (alpha.unsqueeze(-1) * H).sum(dim=1)

        h = torch.tanh(self.fc(ctx))
        h = self.drop(h)
        return self.out(h)

# ------------- MediaPipe helpers -------------
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
mp_hands    = mp.solutions.hands

def extract_258_and_results(frame_bgr, holistic, hands_fb=None):
    """Return (258,) feature vector and holistic results; use Hands fallback if Holistic misses."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)

    # Pose
    vec = []
    if res.pose_landmarks and res.pose_landmarks.landmark:
        for lm in res.pose_landmarks.landmark:
            vec.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        vec.extend([0.0] * (33 * 4))

    # Hands from Holistic (if any)
    l_hand = res.left_hand_landmarks  if (res.left_hand_landmarks  and res.left_hand_landmarks.landmark)  else None
    r_hand = res.right_hand_landmarks if (res.right_hand_landmarks and res.right_hand_landmarks.landmark) else None

    # Fallback with dedicated Hands if missing
    if hands_fb and (l_hand is None or r_hand is None):
        hres = hands_fb.process(rgb)
        if getattr(hres, "multi_hand_landmarks", None) and getattr(hres, "multi_handedness", None):
            pairs = []
            for lm, hd in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                is_left = (hd.classification[0].label.lower() == "left")
                pairs.append((is_left, lm))
            for is_left, lm in pairs:
                if is_left and l_hand is None:
                    l_hand = lm
                if (not is_left) and r_hand is None:
                    r_hand = lm
                if l_hand is not None and r_hand is not None:
                    break

    # Left hand
    if l_hand:
        for lm in l_hand.landmark: vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * (21 * 3))

    # Right hand
    if r_hand:
        for lm in r_hand.landmark: vec.extend([lm.x, lm.y, lm.z])
    else:
        vec.extend([0.0] * (21 * 3))

    arr = np.asarray(vec, dtype=DTYPE)
    if arr.shape[0] != FEAT_DIM:
        if arr.shape[0] > FEAT_DIM: arr = arr[:FEAT_DIM]
        else: arr = np.pad(arr, (0, FEAT_DIM - arr.shape[0]), mode="constant")
    return arr, res

def draw_landmarks(frame_bgr, results, draw_pose=True, draw_hands=True):
    if draw_pose and results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
    if draw_hands and results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,128,0), thickness=2)
        )
    if draw_hands and results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,0,128), thickness=2)
        )

# ---- Impute brief hand gaps (linear or hold-last) ----
def impute_short_gaps(seq: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """
    seq: (T, 258). Fill short zero-runs (<= max_gap) in hand channels with linear interpolation,
    or hold-last when only one side bound exists.
    """
    x = seq.copy()
    for slc in (L_SLICE, R_SLICE):
        sub = x[:, slc]  # (T,63)
        miss = (np.abs(sub).sum(axis=1) == 0.0)
        if not miss.any():
            continue
        t = np.arange(len(sub))
        for c in range(sub.shape[1]):
            vals = sub[:, c]
            i = 0
            while i < len(vals):
                if miss[i]:
                    j = i
                    while j < len(vals) and miss[j]:
                        j += 1
                    gap = j - i
                    if gap <= max_gap:
                        left_idx  = i - 1
                        right_idx = j
                        left_ok  = (left_idx >= 0 and not miss[left_idx])
                        right_ok = (right_idx < len(vals) and not miss[right_idx])
                        if left_ok and right_ok:
                            v0, v1 = vals[left_idx], vals[right_idx]
                            for k in range(gap):
                                vals[i + k] = v0 + (v1 - v0) * ((k + 1) / (gap + 1))
                        elif left_ok:
                            vals[i:j] = vals[left_idx]
                        elif right_ok:
                            vals[i:j] = vals[right_idx]
                        # else leave zeros (no bounds)
                    i = j
                else:
                    i += 1
            sub[:, c] = vals
        x[:, slc] = sub
    return x

# ------------- UI overlays -------------
def draw_panel_text(frame, lines, panel_w=None):
    h, w = frame.shape[:2]
    x0, y0 = 10, 28
    if panel_w is None:
        panel_w = max(360, w // 3)
    panel_h = 24 * (len(lines) + 1)
    cv2.rectangle(frame, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    y = y0
    for line, color in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 24

def draw_prediction_block(frame, top_labels, top_probs, infer_ms=None):
    lines = [("Prediction (last window)", (0, 255, 255))]
    for lbl, p in zip(top_labels, top_probs):
        lines.append((f"{lbl[:38]:<38} {p*100:5.1f}%", (0, 255, 0)))
    if infer_ms is not None:
        lines.append((f"Infer: {infer_ms:.1f} ms", (200, 200, 200)))
    draw_panel_text(frame, lines)

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with label_to_id.json")
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_best.pt (trained BiLSTMAttn)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seq", type=int, default=SEQ_LEN)
    ap.add_argument("--min_frames", type=int, default=24, help="Minimum frames required to run a prediction")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-draw", dest="draw", action="store_false", help="Disable landmark drawing")
    ap.add_argument("--hol_complex", type=int, default=2, help="Holistic model complexity (0/1/2)")
    ap.add_argument("--det_conf", type=float, default=0.35)
    ap.add_argument("--trk_conf", type=float, default=0.5)
    ap.add_argument("--hands_fallback", action="store_true", help="Use dedicated Hands as fallback")
    ap.set_defaults(draw=True)
    args = ap.parse_args()

    # Fewer threads = more stable on Windows
    try: cv2.setNumThreads(1)
    except: pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Labels
    data_root = Path(args.data)
    label_to_id = json.loads((data_root / "label_to_id.json").read_text(encoding="utf-8"))
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(id_to_label)

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}" + (f": {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    params = ckpt.get("params") or ckpt.get("config") or {}
    hidden  = int(params.get("hidden", 128))
    layers  = int(params.get("layers", 2))
    dropout = float(params.get("dropout", 0.3))
    model = BiLSTMAttn(num_classes=num_classes, hidden=hidden, layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # MediaPipe
    hol = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(args.hol_complex),
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=float(args.det_conf),
        min_tracking_confidence=float(args.trk_conf),
    )
    hands_fb = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=float(args.det_conf),
        min_tracking_confidence=float(args.trk_conf),
    ) if args.hands_fallback else None

    # Camera
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # State
    recording = False
    window_feats = []         # list of (258,) during current window
    last_pred_labels = []
    last_pred_probs  = []
    last_infer_ms    = None
    window_id        = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed"); break
            frame = cv2.flip(frame, 1)

            feats, results = extract_258_and_results(frame, hol, hands_fb=hands_fb)
            if args.draw:
                draw_landmarks(frame, results, draw_pose=True, draw_hands=True)

            if recording:
                window_feats.append(feats)
                draw_panel_text(frame, [
                    (f"RECORDING window #{window_id} …", (0, 255, 255)),
                    (f"Frames: {len(window_feats)}  (Space: stop | r: reset | d: draw | q: quit)", (200, 200, 200)),
                ])
            else:
                if last_pred_labels:
                    draw_prediction_block(frame, last_pred_labels, last_pred_probs, infer_ms=last_infer_ms)
                else:
                    draw_panel_text(frame, [
                        ("IDLE", (0, 255, 255)),
                        ("Press Space to START a detection window", (0, 255, 0)),
                        ("(q: quit | d: toggle draw | r: clear last prediction)", (200, 200, 200)),
                    ])

            cv2.imshow("INCLUDE-50 | BiLSTM Inference (Space=start/stop)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('d'):
                args.draw = not args.draw
            elif key == ord('r'):
                if recording:
                    window_feats.clear()
                else:
                    last_pred_labels, last_pred_probs, last_infer_ms = [], [], None
            elif key == 32:  # Space
                if not recording:
                    recording = True
                    window_feats.clear()
                    window_id += 1
                else:
                    recording = False
                    n = len(window_feats)
                    if n < args.min_frames:
                        last_pred_labels = ["(too few frames)"]
                        last_pred_probs  = [0.0]
                        last_infer_ms    = None
                        continue

                    # Build model input: impute short hand gaps, then pad to seq_len
                    L = min(n, args.seq)
                    x = np.stack(window_feats[:L], axis=0)
                    x = impute_short_gaps(x, max_gap=5)
                    if L < args.seq:
                        pad = np.zeros((args.seq - L, FEAT_DIM), dtype=DTYPE)
                        x = np.concatenate([x, pad], axis=0)
                    x = x.astype(DTYPE)

                    x_t = torch.from_numpy(x).unsqueeze(0).to(device)      # (1,T,F)
                    L_t = torch.tensor([L], dtype=torch.long, device=device)

                    t0 = time.time()
                    with torch.inference_mode():
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(args.amp and device.type=="cuda")):
                            logits = model(x_t, L_t)                        # (1,C)
                            probs  = torch.softmax(logits, dim=1)           # (1,C)
                    probs = probs.float().squeeze(0).cpu().numpy()
                    last_infer_ms = (time.time() - t0) * 1000.0

                    k = int(np.clip(args.topk, 1, num_classes))
                    idx = np.argsort(-probs)[:k]
                    last_pred_labels = [id_to_label[i] for i in idx]
                    last_pred_probs  = [float(probs[i]) for i in idx]

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hol.close()
        if hands_fb:
            hands_fb.close()

if __name__ == "__main__":
    main()
