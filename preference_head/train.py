#!/usr/bin/env python
"""
train_pref_head_pairwise.py  (v4: ctx-aware, gen-ready)
────────────────────────────────────────────────────────
• train / valid / test JSONL 분할
• LLM(동결) + ctx-aware Preference-Head 학습
• 입력:  (질문 + 이전 win/lose prefix + 현재 후보 토큰)
• 출력:  스칼라 선호도 → 생성 시 logits 와 합산 가능
────────────────────────────────────────────────────────
"""

import argparse, json, random, math, os, numpy as np, torch
from pathlib import Path
from tqdm.auto import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────── CLI ────────
p = argparse.ArgumentParser()
p.add_argument("--train",  required=True)
p.add_argument("--valid",  required=True)
p.add_argument("--test",   required=True)
p.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
p.add_argument("--width",   type=int, default=1024)
p.add_argument("--lr",      type=float, default=1e-4)
p.add_argument("--bsz",     type=int, default=4)
p.add_argument("--epochs",  type=int, default=32)
p.add_argument("--maxlen",  type=int, default=512)
p.add_argument("--log_every", type=int, default=25)
p.add_argument("--save",    default="pref_head_ctx.pt")
p.add_argument("--gpus",    default="0,1")
args = p.parse_args()

torch.manual_seed(42); random.seed(42)

gpu_ids = [int(x) for x in args.gpus.split(",")]
assert torch.cuda.device_count() >= len(gpu_ids), "GPU id 오류"
device  = torch.device(f"cuda:{gpu_ids[0]}")

# ──────── LLM 로드 (freeze) ────────
tok = AutoTokenizer.from_pretrained(args.model_id)
tok.pad_token = tok.eos_token
tok.padding_side = tok.truncation_side = "left"

mdl = AutoModelForCausalLM.from_pretrained(
    args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
for p_ in mdl.parameters(): p_.requires_grad_(False)
H = mdl.config.hidden_size

# ──────── Preference Head ────────
class PrefHead(nn.Module):
    """ctx 포함 hidden(H) → score"""
    def __init__(self, h_dim=H, width=args.width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1)
        )
    def forward(self, h):           # h: (B, H)
        return self.net(h).squeeze(-1)

Wj_base = PrefHead().to(device)
Wj = nn.DataParallel(Wj_base, device_ids=gpu_ids) if len(gpu_ids) > 1 else Wj_base

# ──────── Dataset ────────
class PairwiseDS(Dataset):
    def __init__(self, path): self.rows = [json.loads(l) for l in Path(path).open()]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): r = self.rows[i]; return r["q"], r["win"], r["lose"]

def make_sequences(q_ids, w_ids, l_ids):
    """토큰별 prefix 확장 → (win_seq, lose_seq) 리스트 반환"""
    seq_win, seq_lose = [], []
    m = min(len(w_ids), len(l_ids))
    for t in range(m):
        win_seq = (q_ids + w_ids[:t+1])[-args.maxlen:]   # 후보 포함
        lose_seq= (q_ids + l_ids[:t+1])[-args.maxlen:]
        seq_win.append(win_seq);  seq_lose.append(lose_seq)
    return seq_win, seq_lose

def collate(samples):
    w_batch, l_batch, w_len, l_len = [], [], [], []
    for q,w,l in samples:
        q_ids = tok(q, add_special_tokens=False)["input_ids"]
        w_ids = tok(w, add_special_tokens=False)["input_ids"]
        l_ids = tok(l, add_special_tokens=False)["input_ids"]
        seq_w, seq_l = make_sequences(q_ids, w_ids, l_ids)
        w_batch.extend(seq_w);  l_batch.extend(seq_l)
    # padding
    def pad(list_ids):
        mx = max(len(x) for x in list_ids)
        ids = [ [tok.pad_token_id]*(mx-len(x))+x for x in list_ids ]
        att = [ [0]*(mx-len(x))+[1]*len(x) for x in list_ids ]
        lens= [len(x) for x in list_ids]
        return torch.tensor(ids), torch.tensor(att), torch.tensor(lens)
    w_ids, w_att, w_len = pad(w_batch)
    l_ids, l_att, l_len = pad(l_batch)
    return (w_ids, w_att, w_len), (l_ids, l_att, l_len)

def make_loader(path, shuffle):
    return DataLoader(
        PairwiseDS(path), batch_size=args.bsz, shuffle=shuffle,
        collate_fn=collate, num_workers=2, pin_memory=True
    )

train_loader = make_loader(args.train,  True)
valid_loader = make_loader(args.valid,  False)
test_loader  = make_loader(args.test,   False)

# ──────── Optim / Loss ────────
opt = torch.optim.AdamW(Wj.parameters(), lr=args.lr)
bce = nn.BCEWithLogitsLoss()

def fwd_hidden(ids, att, length):
    """마지막 실토큰 hidden 추출"""
    out = mdl(input_ids=ids, attention_mask=att,
              output_hidden_states=True).hidden_states[-1]   # (B, L, H)
    idx = (length-1).unsqueeze(1).unsqueeze(2).repeat(1,1,H) # gather 인덱스
    return out.gather(1, idx).squeeze(1)                    # (B, H)

@torch.no_grad()
def eval_loop(loader):
    Wj.eval(); loss_sum=acc=tot=0
    for (w_ids,w_att,w_len),(l_ids,l_att,l_len) in loader:
        w_ids=w_ids.to(device); w_att=w_att.to(device); w_len=w_len.to(device)
        l_ids=l_ids.to(device); l_att=l_att.to(device); l_len=l_len.to(device)
        h_w = fwd_hidden(w_ids, w_att, w_len).float()
        h_l = fwd_hidden(l_ids, l_att, l_len).float()
        s_w, s_l = Wj(h_w), Wj(h_l)
        loss = bce(s_w - s_l, torch.ones_like(s_w))
        loss_sum += float(loss)*len(s_w)
        acc += (s_w > s_l).sum().item();  tot += len(s_w)
    return loss_sum/tot, acc/tot

# ──────── Training ────────
best_val=float("inf"); hist=[]
for ep in range(1,args.epochs+1):
    Wj.train(); run=cnt=0
    pbar=tqdm(train_loader, desc=f"[epoch {ep}]")
    for (w_ids,w_att,w_len),(l_ids,l_att,l_len) in pbar:
        w_ids=w_ids.to(device); w_att=w_att.to(device); w_len=w_len.to(device)
        l_ids=l_ids.to(device); l_att=l_att.to(device); l_len=l_len.to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            h_w=fwd_hidden(w_ids,w_att,w_len).float()
            h_l=fwd_hidden(l_ids,l_att,l_len).float()
            s_w,s_l = Wj(h_w), Wj(h_l)
            loss = bce(s_w - s_l, torch.ones_like(s_w))
        loss.backward(); opt.step(); opt.zero_grad()
        run += float(loss); cnt += 1; hist.append(float(loss))
        if cnt % args.log_every==0: pbar.set_postfix(loss=f"{float(loss):.4f}")
    tr = run/cnt; vl,va = eval_loop(valid_loader)
    print(f"Epoch {ep} → train {tr:.4f} | val {vl:.4f} | acc {va:.3%}")
    if vl<best_val:
        best_val=vl; torch.save(Wj_base.state_dict(), args.save)
        print(f"[✓] saved → {args.save}")

# ──────── Test ────────
Wj_base.load_state_dict(torch.load(args.save))
tl,ta = eval_loop(test_loader)
print(f"[TEST] loss {tl:.4f} | acc {ta:.3%}")

# ──────── Loss log 저장 ────────
np.save("loss_log.npy", np.array(hist))
