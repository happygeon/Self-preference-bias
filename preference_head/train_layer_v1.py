import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tqdm.auto import tqdm

# ───── CLI ─────
P = argparse.ArgumentParser()
P.add_argument("--epochs",    type=int, default=50)
P.add_argument("--batch",     type=int, default=8192)
P.add_argument("--margin",    type=float, default=0.5)
P.add_argument("--width",     type=int, default=256)
P.add_argument("--dropout",   type=float, default=0.2)
P.add_argument("--kmax",      type=int, default=0)
P.add_argument("--workers",   type=int, default=4)
P.add_argument("--patience",  type=int, default=9)
P.add_argument("--save",      default="pref_head_reg_layer.pt")
P.add_argument("--train", nargs="+", default=["train"])
P.add_argument("--valid", nargs="+", default=["valid"])
P.add_argument("--test",  nargs="+", default=["test"])
args = P.parse_args()

torch.manual_seed(0); np.random.seed(0)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# ───── Dataset ─────
class PairTokDS(IterableDataset):
    """
    각 split에 대해 하나 이상의 폴더를 입력받아 결합.
    각 폴더는 다음 파일들을 포함한다고 가정:
      hidden.fp16.bin   (TOT,4096)
      next_emb.fp16.bin (TOT,4096)  ← Llama 임베딩 행 (고정)
      offsets.int64.npy       (N,2)
    offsets로 행 인덱스를 맞춰 토큰쌍을 뽑아
    feature = concat(hidden, embedding)  (8192-dim) 로 반환.
    """
    def __init__(self, dir_paths, shuffle=True, kmax=None):
        if isinstance(dir_paths, (str, os.PathLike)):
            dir_paths = [dir_paths]
        self.dir_paths = list(dir_paths)
        self.hid_list, self.emb_list, self.off_list = [], [], []
        for d in self.dir_paths:
            hid = np.memmap(os.path.join(d, "hidden.fp16.bin"),
                            dtype="float16", mode="r").reshape(-1, 4096)
            emb = np.memmap(os.path.join(d, "next_emb.fp16.bin"),
                            dtype="float16", mode="r").reshape(-1, 4096)
            off = np.load(os.path.join(d, "offsets.int64.npy"))
            self.hid_list.append(hid)
            self.emb_list.append(emb)
            self.off_list.append(off)
        self.shuffle, self.kmax = shuffle, kmax

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            seed = 0 + worker_info.id
            np.random.seed(seed)
        
        dir_indices = list(range(len(self.off_list)))
        if self.shuffle:
            np.random.shuffle(dir_indices)

        for d in dir_indices:
            off = self.off_list[d]
            idxs = np.arange(0, len(off), 2)
            if self.shuffle:
                np.random.shuffle(idxs)

            hid = self.hid_list[d]; emb = self.emb_list[d]
            for w in idxs:
                l = w + 1
                sw, ew = off[w]
                sl, el = off[l]
                m = min(ew - sw, el - sl)
                if m == 0:
                    continue

                if self.kmax and m > self.kmax:
                    if self.shuffle:  # 학습(shuffle=True) 중에만 무작위 샘플링
                        ts = np.random.choice(m, self.kmax, replace=False)
                    else:  # 평가(shuffle=False) 중에는 항상 앞에서부터 kmax개를 선택
                        ts = range(self.kmax)
                else:
                    ts = range(m)

                for t in ts:
                    feat_w = np.concatenate([hid[sw + t], emb[sw + t]]).astype("float32")
                    feat_l = np.concatenate([hid[sl + t], emb[sl + t]]).astype("float32")
                    yield torch.from_numpy(feat_w), torch.from_numpy(feat_l)

def collate(batch):
    hw, hl = zip(*batch)
    return torch.stack(hw), torch.stack(hl)

def make_loader(paths, shuffle):
    return DataLoader(
        PairTokDS(paths, shuffle, args.kmax),
        batch_size=args.batch, collate_fn=collate,
        num_workers=args.workers, pin_memory=True
    )

train_dl = make_loader(args.train, True)
valid_dl = make_loader(args.valid, False)
test_dl  = make_loader(args.test,  False)


# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ MODIFIED ARCHITECTURE ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
class PrefHeadV2(nn.Module):
    """
    (New Architecture)
    This version mitigates the bottleneck by adding a wider intermediate layer and a 
    residual-like connection. The information has two paths:
    1.  **Deep Path**: A deeper MLP that processes the features non-linearly.
        8192 → width*2 → width → width//2
    2.  **Wide Path**: A single linear projection to preserve information from the input.
        8192 → width//2
    The paths are added together, allowing the model to learn complex features while
    retaining a direct link to the original high-dimensional input, preventing information loss.
    """
    def __init__(self, width: int = 256, p_drop: float = 0.1):
        super().__init__()
        w0 = int(width * 2) # Wider intermediate layer
        w1 = int(width)
        w2 = max(1, w1 // 2) # Final intermediate dimension

        self.in_norm = nn.LayerNorm(8192)
        
        # Path 1: Deeper, non-linear processing
        self.deep_path = nn.Sequential(
            nn.Linear(8192, w0),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(w0, w1),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(w1, w2)
        )
        
        # Path 2: A direct, linear "skip connection" to the final dimension
        self.wide_path = nn.Linear(8192, w2)

        # Final layers after combining the paths
        self.post_act = nn.GELU()
        self.post_drop = nn.Dropout(p_drop)
        self.out_proj = nn.Linear(w2, 1)

    def forward(self, x):
        x_norm = self.in_norm(x)
        
        # Calculate both paths
        deep_features = self.deep_path(x_norm)
        wide_features = self.wide_path(x_norm)
        
        # Add them together (residual-style connection)
        combined = deep_features + wide_features
        
        # Final processing
        activated = self.post_act(combined)
        dropped = self.post_drop(activated)
        output = self.out_proj(dropped)
        
        return output.squeeze(-1)

# Using the new, improved architecture
model = PrefHeadV2(args.width, args.dropout).to(DEV)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ MODIFIED ARCHITECTURE ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.1
)

sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
margin = args.margin
clip_norm = 1.0

@torch.no_grad()
def evaluate(loader):
    model.eval(); tot=n_ok=loss_sum=0.0
    for hw, hl in loader:
        hw, hl = hw.to(DEV), hl.to(DEV)
        diff     = model(hw) - model(hl)
        loss     = torch.relu(margin - diff)
        n_ok    += (diff > 0).sum().item()
        loss_sum += loss.sum().item(); tot += hw.size(0)
    return loss_sum/tot, n_ok/tot

best = float("inf"); wait = 0
for ep in range(1, args.epochs + 1):
    model.train(); run = n = 0.0
    for hw, hl in tqdm(train_dl, desc=f"ep{ep}"):
        hw, hl = hw.to(DEV), hl.to(DEV)
        diff   = model(hw) - model(hl)
        loss   = torch.relu(margin - diff).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step(); opt.zero_grad()

        run += float(loss) * hw.size(0); n += hw.size(0)

    tr = run / n
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
print(f"Loading best model from {args.save} for final testing...")
model.load_state_dict(torch.load(args.save))
tl, ta = evaluate(test_dl)
print(f"[TEST] loss {tl:.4f} | acc {ta:.3%}")