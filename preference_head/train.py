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
P.add_argument("--save",      default="pref_head_reg.pt")
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
      hidden.fp16.bin  (TOT,4096)
      next_emb.fp16.bin(TOT,4096)  ← Llama 임베딩 행 (고정)
      offsets.int64.npy           (N,2)
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
        # ↓↓↓ FIX 1: 각 워커 프로세스가 고유한 랜덤 시드를 갖도록 설정합니다.
        # 이렇게 하지 않으면 모든 워커가 동일한 순서의 데이터를 생성합니다.
        worker_info = get_worker_info()
        if worker_info is not None:
            # 기본 시드(0)와 워커 ID를 조합하여 고유한 시드를 만듭니다.
            seed = 0 + worker_info.id
            np.random.seed(seed)
        
        # 디렉터리 순서 섞기(옵션)
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

                # ↓↓↓ FIX 2: 검증/테스트 시에는 무작위 샘플링을 하지 않도록 수정합니다.
                # 평가 결과의 재현성을 보장하기 위함입니다.
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

# ───── Model ─────
class PrefHead(nn.Module):
    """
    입력 차원 = 4096(hidden) + 4096(embed) = 8192
    2층 MLP (#2안): 8192 → width → (width//2) → 1
    """
    def __init__(self, width: int = 256, p_drop: float = 0.1):
        super().__init__()
        w1 = int(width)
        w2 = max(1, w1 // 2)

        self.in_norm = nn.LayerNorm(8192)
        self.net = nn.Sequential(
            nn.Linear(8192, w1),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(w1, w2),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(w2, 1),
        )

    def forward(self, x):
        x = self.in_norm(x)
        return self.net(x).squeeze(-1)


model = PrefHead(args.width, args.dropout).to(DEV)
opt = torch.optim.SGD(
    model.parameters(),
    lr=1e-4,
    momentum=0.9,
    weight_decay=1e-4
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
model.load_state_dict(torch.load(args.save))
tl, ta = evaluate(test_dl)
print(f"[TEST] loss {tl:.4f} | acc {ta:.3%}")