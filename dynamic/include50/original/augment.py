# augment.py — INCLUDE-50: load → augment(crop) → parallel landmark extraction → save
# Usage (Windows):
#   py augment.py --root E:/INCLUDE --out E:/INCLUDE_KEYPOINTS --left 10 --right 10 --top 0 --workers 4
#
# Outputs:
#   E:/INCLUDE_KEYPOINTS/
#     ├─ label_to_id.json
#     ├─ index_train.csv / index_val.csv / index_test.csv
#     └─ {train,val,test}/{label_id}/*.npz  (x:(200,258) float16, y:label_id)
#
# Deps:
#   pip install mediapipe==0.10.14 opencv-python numpy pandas tqdm

import os
import json
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

import mediapipe as mp_solutions

# ----------------- Paper-matched constants -----------------
SEQ_LEN = 200            # pad/trim to 200 frames
CROP_PX = 50             # fixed crop in pixels (left/right/top)
DTYPE = np.float16       # compact storage

# -------- Faulty INCLUDE-50 TRAIN videos listed in the paper --------
# Drop these from TRAIN only (match by label substring + MVI stem inside filename).
FAULTY_TRAIN_REMOVALS: List[Tuple[str, str]] = [
    ("40. I", "MVI_0001"),
    ("40. I", "MVI_0002"),
    ("40. I", "MVI_0003"),
    ("34. Pen", "MVI_4908"),
    ("61. Father", "MVI_3912"),
    ("11. Car", "MVI_3118"),
    ("1. Dog", "MVI_3002"),
    # Heuristic entries (safe no-ops if absent in your CSVs):
    ("Paint", "MVI_4928"),
    ("87. Hot", ""),   # empty MVI → drop all matching "87. Hot" if any are known-bad in your copy
]

# ---------------- Globals used inside worker processes ----------------
_HOLISTIC = None  # per-process MediaPipe Holistic

def init_worker():
    """Initializer for each worker process: set threading limits and create Holistic once."""
    # Avoid OpenCV over-threading inside each process
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    # Also cap OpenMP if present
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    global _HOLISTIC
    mp_holistic = mp_solutions.solutions.holistic
    _HOLISTIC = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        refine_face_landmarks=False,  # ignore face
        min_detection_confidence=0.67,
        min_tracking_confidence=0.67,
    )

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def coerce_include50(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(lambda v: str(v).strip().lower() in {"1", "true", "t", "yes", "y"})

def build_label_map(df_list: List[pd.DataFrame]) -> Dict[str, int]:
    labels = sorted(set(pd.concat(df_list)["label"].astype(str).tolist()))
    return {lbl: i for i, lbl in enumerate(labels)}

def row_matches_faulty(label: str, video_path: str) -> bool:
    l = label.lower()
    vp = video_path.lower()
    for lbl_sub, mvi in FAULTY_TRAIN_REMOVALS:
        if lbl_sub.lower() in l:
            if not mvi:
                return True
            if mvi.lower() in vp:
                return True
    return False

def apply_faulty_train_removals(train_df: pd.DataFrame) -> pd.DataFrame:
    to_drop = []
    for idx, row in train_df.iterrows():
        if row_matches_faulty(str(row["label"]), str(row["video_path"])):
            to_drop.append(idx)
    if to_drop:
        print(f"[INFO] Faulty train removals per paper: dropping {len(to_drop)} rows")
        train_df = train_df.drop(index=to_drop)
    else:
        print("[INFO] No faulty train rows matched (OK)")
    return train_df

def crop_frame(frame: np.ndarray, crop_kind: str) -> Optional[np.ndarray]:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w <= CROP_PX or h <= CROP_PX:
        return None
    if crop_kind == "orig":
        return frame
    if crop_kind == "left":
        return frame[:, CROP_PX:w, :]
    if crop_kind == "right":
        return frame[:, 0:w - CROP_PX, :]
    if crop_kind == "top":
        return frame[CROP_PX:h, :, :]
    if crop_kind == "top_left":
        return frame[CROP_PX:h, CROP_PX:w, :]
    if crop_kind == "top_right":
        return frame[CROP_PX:h, 0:w - CROP_PX, :]
    return None

def extract_keypoints_from_video_worker(video_path: str, crop_kind: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Worker-side extractor: uses the per-process global _HOLISTIC."""
    global _HOLISTIC
    if _HOLISTIC is None:
        init_worker()  # safety fallback

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feats = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = crop_frame(frame, crop_kind)
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _HOLISTIC.process(rgb)

        vec = []

        # Pose: 33 × (x,y,z,visibility)
        if results.pose_landmarks and results.pose_landmarks.landmark:
            for lm in results.pose_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            vec.extend([0.0] * (33 * 4))

        # Left hand: 21 × (x,y,z)
        if results.left_hand_landmarks and results.left_hand_landmarks.landmark:
            for lm in results.left_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * (21 * 3))

        # Right hand: 21 × (x,y,z)
        if results.right_hand_landmarks and results.right_hand_landmarks.landmark:
            for lm in results.right_hand_landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
        else:
            vec.extend([0.0] * (21 * 3))

        v = np.asarray(vec, dtype=DTYPE)
        if v.shape[0] != 258:
            if v.shape[0] > 258:
                v = v[:258].astype(DTYPE)
            else:
                v = np.pad(v, (0, 258 - v.shape[0]), mode="constant").astype(DTYPE)

        feats.append(v)

    cap.release()

    if len(feats) == 0:
        return np.zeros((seq_len, 258), dtype=DTYPE)

    arr = np.stack(feats, axis=0)  # (T, 258)
    if arr.shape[0] >= seq_len:
        arr = arr[:seq_len]
    else:
        pad = np.zeros((seq_len - arr.shape[0], 258), dtype=DTYPE)
        arr = np.concatenate([arr, pad], axis=0)

    return arr.astype(DTYPE)

def get_augment_kinds_for_split(split: str, left: int, right: int, top: int) -> List[str]:
    if split == "train":
        kinds = ["orig"]
        kinds += ["left"] * max(0, left)
        kinds += ["right"] * max(0, right)
        kinds += ["top"] * max(0, top)
        # Optionally add: kinds += ["top_left"]*k + ["top_right"]*k
        return kinds
    return ["orig"]

def build_output_sample_path(out_root: Path, split: str, label_id: int, base_stem: str, suffix: str) -> Path:
    sub = out_root / split / f"{label_id:03d}"
    safe_mkdir(sub)
    return sub / f"{base_stem}__{suffix}.npz"

# ---------------- Task building & parallel runner ----------------
def build_tasks_for_split(
    split: str,
    df: pd.DataFrame,
    root: Path,
    out_root: Path,
    label_to_id: Dict[str, int],
    left: int,
    right: int,
    top: int,
) -> List[dict]:
    kinds = get_augment_kinds_for_split(split, left, right, top)
    tasks = []
    # We want stable per-video suffixes: enumerate kinds per video
    for row in df.itertuples(index=False):
        label = str(getattr(row, "label"))
        vid_rel = str(getattr(row, "video_path")).replace("\\", "/")
        vid_path = root / vid_rel
        if not vid_path.exists():
            # Keep as a warning later; skip task creation
            continue
        label_id = label_to_id[label]
        base_stem = Path(vid_rel).with_suffix("").name
        for k_idx, kind in enumerate(kinds):
            suffix = kind if kind == "orig" else f"{kind}_{k_idx}"
            out_p = build_output_sample_path(out_root, split, label_id, base_stem, suffix)
            tasks.append({
                "split": split,
                "label": label,
                "label_id": label_id,
                "video_rel": vid_rel,
                "video_path": str(vid_path),
                "crop_kind": kind,
                "out_path": str(out_p),
            })
    return tasks

def worker_run(task: dict) -> dict:
    """Run one task in a worker process. Returns a small dict describing success or error."""
    try:
        arr = extract_keypoints_from_video_worker(task["video_path"], task["crop_kind"], seq_len=SEQ_LEN)
        # Write output (unique path per task → safe without locks)
        np.savez_compressed(task["out_path"], x=arr.astype(DTYPE), y=np.int16(task["label_id"]))
        return {
            "ok": True,
            "npz_path": task["out_path"],
            "label": task["label"],
            "label_id": task["label_id"],
            "video_path": task["video_rel"],
            "crop_kind": task["crop_kind"],
        }
    except Exception as e:
        # Return error; main process will log it and continue
        return {"ok": False, "error": f"{e}", "task": task}

def process_split_parallel(
    split: str,
    df: pd.DataFrame,
    root: Path,
    out_root: Path,
    label_to_id: Dict[str, int],
    left: int,
    right: int,
    top: int,
    workers: int,
    chunksize: int = 2,
):
    tasks = build_tasks_for_split(split, df, root, out_root, label_to_id, left, right, top)
    total = len(tasks)
    print(f"[{split}] videos: {len(df)} | tasks (video×crop): {total} | workers: {workers}")

    index_rows = []
    errors = 0

    if total == 0:
        print(f"[{split}] No tasks to run.")
        pd.DataFrame(index_rows).to_csv(out_root / f"index_{split}.csv", index=False)
        return

    # Windows-safe Pool with initializer that creates one Holistic per process
    with mp.get_context("spawn").Pool(processes=workers, initializer=init_worker) as pool:
        # imap_unordered gives us a nice stream for tqdm
        for res in tqdm(pool.imap_unordered(worker_run, tasks, chunksize=chunksize),
                        total=total, desc=f"{split}: tasks", unit="task"):
            if res.get("ok"):
                index_rows.append({
                    "npz_path": res["npz_path"],
                    "label": res["label"],
                    "label_id": res["label_id"],
                    "video_path": res["video_path"],
                    "crop_kind": res["crop_kind"],
                })
            else:
                errors += 1
                t = res.get("task", {})
                # Print compact error line; full trace is inside worker (not shown); if needed, re-run serial.
                tqdm.write(f"[ERR] {split} | {t.get('video_rel')} | {t.get('crop_kind')}: {res.get('error')}")

    idx_df = pd.DataFrame(index_rows)
    idx_path = out_root / f"index_{split}.csv"
    idx_df.to_csv(idx_path, index=False)
    print(f"[{split}] wrote {len(idx_df)} items → {idx_path} | errors: {errors}")

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to INCLUDE root (contains CSVs + videos)")
    ap.add_argument("--out", required=True, help="Output directory for .npz keypoints & indexes")
    ap.add_argument("--left", type=int, default=10, help="Train: # left crops per video")
    ap.add_argument("--right", type=int, default=10, help="Train: # right crops per video")
    ap.add_argument("--top", type=int, default=0, help="Train: # top crops per video")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    ap.add_argument("--chunksize", type=int, default=2, help="Pool.imap_unordered chunksize")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    safe_mkdir(out_root)

    # Load CSVs
    train_csv = root / "include_train.csv"
    val_csv   = root / "include_val.csv"
    test_csv  = root / "include_test.csv"
    for p in (train_csv, val_csv, test_csv):
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    # Coerce include_50 → bool and filter INCLUDE-50 only
    for df in (train_df, val_df, test_df):
        if "include_50" not in df.columns:
            raise ValueError("CSV must contain 'include_50' column (True/False).")
    train_df = train_df[coerce_include50(train_df["include_50"])].copy()
    val_df   = val_df[coerce_include50(val_df["include_50"])].copy()
    test_df  = test_df[coerce_include50(test_df["include_50"])].copy()

    # Drop faulty INCLUDE-50 train videos per paper
    train_df = apply_faulty_train_removals(train_df)

    # Build label map across INCLUDE-50
    label_to_id = build_label_map([train_df, val_df, test_df])
    with open(out_root / "label_to_id.json", "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved label_to_id.json with {len(label_to_id)} classes")

    # Process each split in parallel
    print("\n=== START: TRAIN ===")
    process_split_parallel("train", train_df, root, out_root, label_to_id,
                           left=args.left, right=args.right, top=args.top,
                           workers=max(1, args.workers), chunksize=max(1, args.chunksize))

    print("\n=== START: VAL ===")
    process_split_parallel("val", val_df, root, out_root, label_to_id,
                           left=0, right=0, top=0,
                           workers=max(1, args.workers), chunksize=max(1, args.chunksize))

    print("\n=== START: TEST ===")
    process_split_parallel("test", test_df, root, out_root, label_to_id,
                           left=0, right=0, top=0,
                           workers=max(1, args.workers), chunksize=max(1, args.chunksize))

    print("\n[DONE] All splits processed.")

if __name__ == "__main__":
    # Windows requires the spawn guard for multiprocessing
    main()
