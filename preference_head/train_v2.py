#!/usr/bin/env python
"""
train_pref_head_proper_multigpu.py - 진짜 멀티 GPU 활용
────────────────────────────────────────────────────────
방법 1: 모델 병렬화 (Model Parallelism) - 추천
방법 2: 파이프라인 병렬화 (Pipeline Parallelism)  
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
p.add_argument("--width",   type=int, default=8)
p.add_argument("--lr",      type=float, default=1e-3)
p.add_argument("--bsz",     type=int, default=32)  # 줄임
p.add_argument("--epochs",  type=int, default=32)
p.add_argument("--maxlen",  type=int, default=1024)  # 줄임
p.add_argument("--max_sequences", type=int, default=512)  # 시퀀스 제한
p.add_argument("--log_every", type=int, default=25)
p.add_argument("--save",    default="pref_head_ctx.pt")
p.add_argument("--strategy", choices=["model_parallel"], default="model_parallel")
args = p.parse_args()

torch.manual_seed(42); random.seed(42)

# GPU 확인
num_gpus = torch.cuda.device_count()
print(f"🚀 사용 가능한 GPU: {num_gpus}개")
assert num_gpus >= 2, "최소 2개 GPU 필요"

# ──────── 전략 1: Model Parallelism (추천) ────────
if args.strategy == "model_parallel":
    print("📋 전략: Model Parallelism - LLM을 여러 GPU에 분산")
    
    # 토크나이저
    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.pad_token = tok.eos_token
    tok.padding_side = tok.truncation_side = "left"
    
    # 수동 device_map 설정 - 레이어별로 GPU 분산
    num_layers = 32  # Llama 3.1 8B 레이어 수
    layers_per_gpu = num_layers // num_gpus
    
    device_map = {}
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = num_gpus - 1
    device_map["lm_head"] = num_gpus - 1
    
    # 레이어 분산
    for i in range(num_layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f"model.layers.{i}"] = gpu_id
    
    print(f"📊 Device Map 샘플: {dict(list(device_map.items())[:5])}")
    
    # 모델 로드
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,  # 수동 분산
        low_cpu_mem_usage=True,
        offload_folder="./offload"  # 필요시 CPU 오프로드
    )
    
    for p_ in mdl.parameters(): 
        p_.requires_grad_(False)
    
    H = mdl.config.hidden_size
    main_device = num_gpus - 1  # Preference Head는 마지막 GPU에

# ──────── Preference Head ────────
class PrefHead(nn.Module):
    def __init__(self, h_dim=H, width=args.width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, width), nn.GELU(),
            nn.Linear(width, 1)
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)

Wj = PrefHead().to(f'cuda:{main_device}')
print(f"🎯 Preference Head → GPU {main_device}")

# ──────── 메모리 최적화된 Dataset ────────
class PairwiseDS(Dataset):
    def __init__(self, path): 
        self.rows = [json.loads(l) for l in Path(path).open()]
    def __len__(self): 
        return len(self.rows)
    def __getitem__(self, i): 
        r = self.rows[i]
        return r["q"], r["win"], r["lose"]

def make_sequences_limited(q_ids, w_ids, l_ids):
    """시퀀스 수 제한으로 메모리 폭증 방지"""
    m = min(len(w_ids), len(l_ids))
    k = min(m, args.max_sequences)           # k ≤ m
    indices = random.sample(range(m), k)     # 무작위 위치
    indices.sort()                           # 순서 유지(선택 사항)

    seq_win, seq_lose = [], []
    for t in indices:
        seq_win.append((q_ids + w_ids[:t+1])[-args.maxlen:])
        seq_lose.append((q_ids + l_ids[:t+1])[-args.maxlen:])
    return seq_win, seq_lose

def collate(samples):
    w_batch, l_batch = [], []
    for q, w, l in samples:
        # 토큰화 시 길이 제한
        q_ids = tok(q, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
        w_ids = tok(w, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]  
        l_ids = tok(l, add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        
        seq_w, seq_l = make_sequences_limited(q_ids, w_ids, l_ids)
        w_batch.extend(seq_w)
        l_batch.extend(seq_l)
    
    # 하드 리미트: 배치당 최대 시퀀스 수
    max_batch_sequences = 1024
    if len(w_batch) > max_batch_sequences:
        indices = random.sample(range(len(w_batch)), max_batch_sequences)
        w_batch = [w_batch[i] for i in indices]
        l_batch = [l_batch[i] for i in indices]
    
    def pad(list_ids):
        if not list_ids:
            return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        mx = min(max(len(x) for x in list_ids), args.maxlen)
        ids = [[tok.pad_token_id] * (mx - len(x)) + x[-mx:] for x in list_ids]
        att = [[0] * (mx - len(x)) + [1] * min(len(x), mx) for x in list_ids]
        lens = [min(len(x), mx) for x in list_ids]
        return torch.tensor(ids), torch.tensor(att), torch.tensor(lens)
    
    w_ids, w_att, w_len = pad(w_batch)
    l_ids, l_att, l_len = pad(l_batch)
    return (w_ids, w_att, w_len), (l_ids, l_att, l_len)

def make_loader(path, shuffle):
    return DataLoader(
        PairwiseDS(path), batch_size=args.bsz, shuffle=shuffle,
        collate_fn=collate, num_workers=1, pin_memory=True
    )

train_loader = make_loader(args.train, True)
valid_loader = make_loader(args.valid, False)
test_loader = make_loader(args.test, False)

# ──────── 최적화된 Forward ────────
def fwd_hidden_multigpu(ids, att, length):
    """멀티 GPU 환경에서 효율적인 히든 추출"""
    if ids.size(0) == 0:
        return torch.empty(0, H, device=f'cuda:{main_device}', dtype=torch.float32)
    
    # 청크 단위로 처리 (메모리 절약)
    chunk_size = 16  # 더 작게
    hiddens = []
    
    for i in range(0, ids.size(0), chunk_size):
        end_i = min(i + chunk_size, ids.size(0))
        chunk_ids = ids[i:end_i]
        chunk_att = att[i:end_i]  
        chunk_len = length[i:end_i]
        
        # 첫 번째 GPU로 입력 전송 (모델이 여러 GPU에 분산되어 있음)
        first_gpu = 0
        chunk_ids = chunk_ids.to(f'cuda:{first_gpu}')
        chunk_att = chunk_att.to(f'cuda:{first_gpu}')
        
        with torch.no_grad():
            out = mdl(
                input_ids=chunk_ids,
                attention_mask=chunk_att, 
                output_hidden_states=True,
                use_cache=False
            ).hidden_states[-1]  # 이미 마지막 GPU에 있음
        
        # 마지막 토큰 히든 추출
        chunk_len = chunk_len.to(out.device)
        idx = (chunk_len - 1).unsqueeze(1).unsqueeze(2).repeat(1, 1, H)
        chunk_hidden = out.gather(1, idx).squeeze(1)
        
        # Preference Head가 있는 GPU로 이동
        if chunk_hidden.device != torch.device(f'cuda:{main_device}'):
            chunk_hidden = chunk_hidden.to(f'cuda:{main_device}')
        
        hiddens.append(chunk_hidden)
        
        # 메모리 정리
        del out, chunk_ids, chunk_att, chunk_len, idx, chunk_hidden
        torch.cuda.empty_cache()
    
    return torch.cat(hiddens, dim=0).float() if hiddens else torch.empty(0, H, device=f'cuda:{main_device}', dtype=torch.float32)

# ──────── 학습 루프 ────────
opt = torch.optim.AdamW(Wj.parameters(), lr=args.lr)
bce = nn.BCEWithLogitsLoss()

@torch.no_grad()
def eval_loop(loader):
    Wj.eval()
    loss_sum = acc = tot = 0
    
    for (w_ids, w_att, w_len), (l_ids, l_att, l_len) in tqdm(loader, desc="Eval"):
        if w_ids.size(0) == 0:
            continue
            
        h_w = fwd_hidden_multigpu(w_ids, w_att, w_len)
        h_l = fwd_hidden_multigpu(l_ids, l_att, l_len)
        
        if h_w.size(0) > 0:
            s_w, s_l = Wj(h_w), Wj(h_l)
            loss = bce(s_w - s_l, torch.ones_like(s_w))
            
            loss_sum += float(loss) * len(s_w)
            acc += (s_w > s_l).sum().item()
            tot += len(s_w)
    
    return (loss_sum / tot, acc / tot) if tot > 0 else (0.0, 0.0)

# 메모리 사용량 모니터링
def print_gpu_memory():
    for i in range(num_gpus):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {mem_allocated:.1f}GB / {mem_reserved:.1f}GB")

print("🔍 초기 GPU 메모리 상태:")
print_gpu_memory()

# ────────── 파라미터·샘플 수 계산 ──────────
train_N = sum(1 for _ in Path(args.train).open())        # 학습 샘플 수
pref_params = sum(p.numel() for p in Wj.parameters())    # Preference-Head 파라미터 수
ratio = pref_params / train_N if train_N else float('inf')
print(f"⚖️  Pref-Head 파라미터 = {pref_params:,}  |  train 샘플 = {train_N:,}")
print(f"⚖️  1 샘플당 파라미터 ≈ {ratio:.2f}")

# ──────── Training ────────
best_val = float("inf")
hist = []

for ep in range(1, args.epochs + 1):
    Wj.train()
    run = cnt = 0
    
    pbar = tqdm(train_loader, desc=f"[epoch {ep}]")
    for (w_ids, w_att, w_len), (l_ids, l_att, l_len) in pbar:
        if w_ids.size(0) == 0:
            continue
        
        # Forward
        h_w = fwd_hidden_multigpu(w_ids, w_att, w_len)
        h_l = fwd_hidden_multigpu(l_ids, l_att, l_len)
        
        if h_w.size(0) == 0:
            continue
            
        s_w, s_l = Wj(h_w), Wj(h_l)
        loss = bce(s_w - s_l, torch.ones_like(s_w))
        
        # Backward
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        run += float(loss)
        cnt += 1
        hist.append(float(loss))
        
        if cnt % args.log_every == 0:
            pbar.set_postfix(loss=f"{float(loss):.4f}")
            
        # 주기적 메모리 정리
        if cnt % 10 == 0:
            torch.cuda.empty_cache()
    
    # Validation
    tr = run / cnt if cnt > 0 else 0
    vl, va = eval_loop(valid_loader)
    print(f"Epoch {ep} → train {tr:.4f} | val {vl:.4f} | acc {va:.3%}")
    
    # GPU 메모리 상태 출력
    if ep % 5 == 0:
        print("📊 현재 GPU 메모리:")
        print_gpu_memory()
    
    if vl < best_val:
        best_val = vl
        torch.save(Wj.state_dict(), args.save)
        print(f"[✓] saved → {args.save}")

# ──────── Test ────────
Wj.load_state_dict(torch.load(args.save))
tl, ta = eval_loop(test_loader)
print(f"[TEST] loss {tl:.4f} | acc {ta:.3%}")

print("\n🏁 최종 GPU 메모리 상태:")
print_gpu_memory()

np.save("loss_log.npy", np.array(hist))