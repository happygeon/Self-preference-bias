#!/usr/bin/env python3
"""
pair_token_train_regularized.py
─────────────────────────────────────────────────────────────
train / valid / test 폴더에서 토큰-쌍 margin 학습.
과적합 완화: Dropout, Weight-decay, Cosine LR, Grad-clip, Early-stop
"""

import os, argparse, bisect, numpy as np, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm.auto import tqdm

# ───── CLI ─────
P = argparse.ArgumentParser()
P.add_argument("--epochs",    type=int, default=50)
P.add_argument("--batch",     type=int, default=8192)
P.add_argument("--margin",    type=float, default=0.5)
P.add_argument("--width",     type=int, default=2)     # 폭 ↑ + dropout
P.add_argument("--dropout",   type=float, default=0.25) # 드롭아웃 비율
P.add_argument("--kmax",      type=int, default=0)
P.add_argument("--workers",   type=int, default=4)
P.add_argument("--patience",  type=int, default=9)      # early-stop
P.add_argument("--save",      default="pref_head_reg.pt")
args = P.parse_args()

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ───── Dataset ─────
class PairTokDS(IterableDataset):
    def __init__(self, dir_path, shuffle=True, kmax=None):
        self.vec = np.memmap(os.path.join(dir_path,"hidden.fp16.bin"),
                             dtype="float16", mode="r").reshape(-1,4096)
        self.off = np.load(os.path.join(dir_path,"offsets.int64.npy"))
        self.shuffle, self.kmax = shuffle, kmax
    def __iter__(self):
        idxs = np.arange(0, len(self.off), 2)
        if self.shuffle: np.random.shuffle(idxs)
        for w in idxs:
            l = w+1
            s_w,e_w = self.off[w]; s_l,e_l = self.off[l]
            m = min(e_w-s_w, e_l-s_l)
            if m==0: continue
            ts = (np.random.choice(m, self.kmax, replace=False)
                  if self.kmax and m>self.kmax else range(m))
            for t in ts:
                yield ( torch.from_numpy(self.vec[s_w+t].astype("float32")),
                        torch.from_numpy(self.vec[s_l+t].astype("float32")) )

def collate(b): hw,hl=zip(*b); return torch.stack(hw), torch.stack(hl)

def make_loader(split, shuffle):
    return DataLoader(
        PairTokDS(split, shuffle, args.kmax),
        batch_size=args.batch, collate_fn=collate,
        num_workers=args.workers, pin_memory=True
    )

train_dl = make_loader("train", True)
valid_dl = make_loader("valid", False)
test_dl  = make_loader("test",  False)

# ───── Model ─────
class PrefHead(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.input_dp = nn.Dropout(p)       # feature dropout
        self.net = nn.Sequential(
            nn.Linear(4096, w),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(w, 1)
        )
    def forward(self,x):
        x = self.input_dp(x)
        return self.net(x).squeeze(-1)

model = PrefHead(args.width, args.dropout).to(DEV)
opt   = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
margin = args.margin
clip_norm = 1.0

@torch.no_grad()
def evaluate(loader):
    model.eval(); tot, ok, loss_sum = 0,0,0.
    for hw,hl in loader:
        hw,hl=hw.to(DEV),hl.to(DEV)
        diff = model(hw)-model(hl)
        loss = torch.relu(margin - diff)
        ok  += (diff>0).sum().item()
        loss_sum += loss.sum().item(); tot += hw.size(0)
    return loss_sum/tot, ok/tot

best = float("inf"); wait = 0
for ep in range(1, args.epochs+1):
    model.train(); run=n=0
    for hw,hl in tqdm(train_dl, desc=f"ep{ep}"):
        hw,hl=hw.to(DEV),hl.to(DEV)
        diff = model(hw)-model(hl)
        loss = torch.relu(margin - diff).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step(); opt.zero_grad()
        run+=float(loss)*hw.size(0); n+=hw.size(0)
    tr = run/n
    vl, va = evaluate(valid_dl)
    sched.step()

    print(f"ep{ep:02}  train {tr:.4f} | val {vl:.4f}  acc {va:.3%}")
    if vl < best:
        best, wait = vl, 0
        torch.save(model.state_dict(), args.save)
        print(f"  ↳ saved {args.save}")
    else:
        wait += 1
        if wait >= args.patience:
            print("⏹ Early stopping triggered"); break

# ───── Test ─────
model.load_state_dict(torch.load(args.save))
tl, ta = evaluate(test_dl)
print(f"[TEST] loss {tl:.4f} | acc {ta:.3%}")
