# train.py — Train BiLSTM(+Attention) on INCLUDE-50 keypoints (.npz)
# Usage (reuse your previous augmented data folder):
#   py train.py --data E:/INCLUDE_KEYPOINTS --save E:/INCLUDE_RUN_BiLSTM ^
#       --epochs 120 --batch 32 --lr 1e-3 --workers 2 --amp
# Paper-style eval (train only on provider train; monitor VAL; report TEST at end):
#   py train.py --data E:/INCLUDE_KEYPOINTS --save E:/INCLUDE_RUN_BiLSTM --epochs 120 --batch 32 --lr 1e-3 --workers 2 --amp --eval_split val
# If you want extra data (not strictly comparable), merge val into train & monitor TEST:
#   py train.py --data E:/INCLUDE_KEYPOINTS --save E:/INCLUDE_RUN_BiLSTM --epochs 120 --batch 32 --lr 1e-3 --workers 2 --amp --use_val_in_train --eval_split test

import os, json, math, argparse, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

SEQ_LEN = 200
FEAT_DIM = 258

# ---------------------- IO utils ----------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_index(csv_path: Path, data_root: Path) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path)
    if "npz_path" not in df.columns or "label_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'npz_path' and 'label_id'")
    paths, labels = [], []
    for _, r in df.iterrows():
        p = Path(str(r["npz_path"]))
        if not p.is_absolute():
            p = data_root / p
        if not p.exists():
            alt = data_root / p.name
            if alt.exists():
                p = alt
            else:
                print(f"[WARN] Missing sample (skipped): {p}")
                continue
        paths.append(str(p))
        labels.append(int(r["label_id"]))
    return paths, labels

# ---------------------- Dataset ----------------------
class NpzSignDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int]):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        with np.load(self.paths[idx]) as npz:
            x = npz["x"].astype(np.float32)   # (200, 258)
            y = int(npz["y"]) if "y" in npz else int(self.labels[idx])
        # valid length (pads are exact all-zero rows at tail)
        valid = np.any(x != 0.0, axis=1)
        length = int(valid.nonzero()[0].max()) + 1 if valid.any() else 1
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), torch.tensor(length, dtype=torch.long)

def collate_batch(batch):
    xs, ys, lens = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.stack(lens, 0)

# ---------------------- Model: BiLSTM + Attention ----------------------
class BiLSTMAttn(nn.Module):
    """
    Encoder: BiLSTM (layers × hidden) over (T, F)
    Pooling: Masked additive attention → context vector
    Head:    Dense(128, tanh) → Dropout → Linear(C)
    """
    def __init__(self, num_classes: int, hidden: int = 128, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.dropout_p = dropout

        self.lstm = nn.LSTM(
            input_size=FEAT_DIM,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=(dropout if layers > 1 else 0.0),
        )

        # Additive attention: score_t = v^T tanh(W h_t)
        self.attn_W = nn.Linear(2 * hidden, 2 * hidden)
        self.attn_v = nn.Linear(2 * hidden, 1, bias=False)

        self.fc = nn.Linear(2 * hidden, 128)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(128, num_classes)

        # init
        nn.init.xavier_uniform_(self.fc.weight);  nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, lengths):
        # x: (B,T,F), lengths: (B,)
        # pack needs CPU lengths (PyTorch requirement)
        lengths_cpu = lengths.detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,Tmax,2H)

        B, Tmax, _ = H.shape
        device = H.device

        # build mask on the SAME device as H
        lengths_dev = lengths.to(device)
        time_ids = torch.arange(Tmax, device=device)           # (Tmax,)
        mask = time_ids.unsqueeze(0) < lengths_dev.unsqueeze(1)  # (B,Tmax) bool

        # attention (keep masking numerically safe under AMP)
        scores = self.attn_v(torch.tanh(self.attn_W(H))).squeeze(-1)  # (B,Tmax)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # instead of -inf
        alpha = torch.softmax(scores, dim=1)                            # (B,Tmax)
        context = (alpha.unsqueeze(-1) * H).sum(dim=1)                  # (B,2H)

        h = torch.tanh(self.fc(context))
        h = self.drop(h)
        return self.out(h)

# ---------------------- Metrics ----------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    correct, total = 0, 0
    all_true, all_pred = [], []
    ctx = (torch.amp.autocast, {"device_type":"cuda", "dtype":torch.float16}) if (use_amp and device.type=="cuda") else (torch.no_grad, {})
    with ctx[0](**ctx[1]):
        for X, Y, L in tqdm(loader, desc="eval", unit="batch", leave=False):
            X = X.to(device, non_blocking=True); Y = Y.to(device, non_blocking=True)
            logits = model(X, L)
            pred = logits.argmax(dim=1)
            correct += (pred == Y).sum().item()
            total   += Y.numel()
            all_true.append(Y.cpu().numpy()); all_pred.append(pred.cpu().numpy())
    acc = correct / max(1, total)
    try:
        from sklearn.metrics import f1_score
        mf1 = f1_score(np.concatenate(all_true), np.concatenate(all_pred), average="macro")
    except Exception:
        mf1 = None
    return acc, mf1

# ---------------------- Train ----------------------
def train_one_epoch(model, loader, optimizer, scaler, device, use_amp=False, label_smoothing=0.05, clip=1.0):
    model.train()
    running_loss, running_correct, running_count = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for X, Y, L in pbar:
        X = X.to(device, non_blocking=True); Y = Y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type=="cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X, L)
                loss = F.cross_entropy(logits, Y, label_smoothing=label_smoothing)
            scaler.scale(loss).backward()
            if clip is not None and clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(X, L)
            loss = F.cross_entropy(logits, Y, label_smoothing=label_smoothing)
            loss.backward()
            if clip is not None and clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()

        running_loss   += loss.item() * Y.size(0)
        running_correct+= (logits.argmax(1) == Y).sum().item()
        running_count  += Y.numel()
        pbar.set_postfix({
            "loss": f"{(running_loss / running_count):.4f}",
            "acc":  f"{(running_correct / max(1, running_count)):.4f}"
        })
    return running_loss / max(1, running_count), running_correct / max(1, running_count)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with label_to_id.json and index_*.csv")
    ap.add_argument("--save", required=True, help="Folder to save checkpoints and final model")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision on GPU")
    ap.add_argument("--use_val_in_train", action="store_true", help="Merge val into train (optional)")
    ap.add_argument("--eval_split", choices=["val","test"], default="val", help="Which split to monitor each epoch")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0, help="Gradient clipping max-norm (0 or neg to disable)")
    args = ap.parse_args()

    data_root, save_root = Path(args.data), Path(args.save)
    safe_mkdir(save_root)

    # Repro + device
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device.type.upper()}" + (f": {torch.cuda.get_device_name(0)}" if device.type=="cuda" else ""))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Label map
    label_to_id = json.loads((data_root / "label_to_id.json").read_text(encoding="utf-8"))
    num_classes = len(label_to_id)
    print(f"[INFO] num_classes = {num_classes}")

    # Indexes
    Xtr, Ytr = load_index(data_root / "index_train.csv", data_root)
    Xva, Yva = load_index(data_root / "index_val.csv",   data_root)
    Xte, Yte = load_index(data_root / "index_test.csv",  data_root)

    if args.use_val_in_train:
        Xtr = Xtr + Xva; Ytr = Ytr + Yva
        print("[INFO] Merged val into train.")
        if args.eval_split == "val":
            print("[NOTE] With --use_val_in_train, switching --eval_split to TEST for clean monitoring.")
            args.eval_split = "test"

    print(f"[INFO] samples — train: {len(Xtr)}, val: {len(Xva)}, test: {len(Xte)}")

    # DataLoaders
    pin = (device.type=="cuda")
    train_loader = DataLoader(NpzSignDataset(Xtr, Ytr), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, collate_fn=collate_batch,
                              persistent_workers=(args.workers>0))
    val_loader   = DataLoader(NpzSignDataset(Xva, Yva), batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=pin, collate_fn=collate_batch,
                              persistent_workers=(args.workers>0))
    test_loader  = DataLoader(NpzSignDataset(Xte, Yte), batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=pin, collate_fn=collate_batch,
                              persistent_workers=(args.workers>0))
    eval_loader  = val_loader if args.eval_split == "val" else test_loader
    print(f"[INFO] Evaluating each epoch on: {args.eval_split.upper()}")

    # Model/optim/sched/AMP
    model = BiLSTMAttn(num_classes=num_classes, hidden=args.hidden, layers=args.layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-5, verbose=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type=="cuda"))

    # Epoch loop (no early stop)
    ep_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch")
    with open(save_root / "train_log.csv", "w", encoding="utf-8") as flog:
        flog.write("epoch,train_loss,train_acc,eval_acc,eval_macro_f1,lr\n")
        best_eval = -1.0
        for epoch in ep_bar:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scaler, device,
                use_amp=args.amp, label_smoothing=args.label_smoothing, clip=args.clip
            )
            model.eval()
            eval_acc, eval_mf1 = evaluate(model, eval_loader, device, use_amp=args.amp)
            scheduler.step(eval_acc)
            ep_bar.set_postfix({"eval_acc": f"{eval_acc:.4f}", "eval_mF1": f"{(eval_mf1 if eval_mf1 is not None else float('nan'))}"}, refresh=False)

            lr_now = optimizer.param_groups[0]["lr"]
            flog.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{eval_acc:.6f},{(eval_mf1 if eval_mf1 is not None else float('nan'))},{lr_now:.6e}\n"); flog.flush()

            if eval_acc > best_eval:
                best_eval = eval_acc
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "eval_acc": eval_acc,
                    "eval_macro_f1": eval_mf1,
                    "label_to_id": label_to_id
                }, save_root / "ckpt_best.pt")
                tqdm.write(f"[SAVE] ckpt_best.pt (epoch {epoch}, {args.eval_split}_acc={eval_acc:.4f})")

    # Final evaluation on VAL & TEST
    ckpt_path = save_root / "ckpt_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] Loaded best checkpoint @ epoch {ckpt.get('epoch')} ({args.eval_split}_acc={ckpt.get('eval_acc'):.4f})")

    v_acc, v_mf1 = evaluate(model, val_loader, device, use_amp=args.amp)
    t_acc, t_mf1 = evaluate(model, test_loader, device, use_amp=args.amp)
    print(f"[VAL]  acc={v_acc:.4f} macro-F1={(v_mf1 if v_mf1 is not None else float('nan'))}")
    print(f"[TEST] acc={t_acc:.4f} macro-F1={(t_mf1 if t_mf1 is not None else float('nan'))}")

    torch.save(model.state_dict(), save_root / "model_final.pt")
    print(f"[SAVE] final weights → {save_root/'model_final.pt'} | best ckpt → {save_root/'ckpt_best.pt'}")

if __name__ == "__main__":
    main()
