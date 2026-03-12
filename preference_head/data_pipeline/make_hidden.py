#!/usr/bin/env python
# build_hidden_dataset_v3.py
"""
Q-A JSONL ─▶ Llama-3.1-8B
 마지막 hidden (h_t)            -> hidden.fp16.bin
 다음 토큰 임베딩 (e_{t+1})     -> next_emb.fp16.bin
 오프셋 / 라벨                  -> offsets.int64.npy , labels.int8.npy
"""

import argparse, json, os, pathlib, warnings, numpy as np, torch, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ───── CLI ─────
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)          # Q-A JSONL
ap.add_argument("--out_dir", default="hidden_ds")
ap.add_argument("--max_len", type=int, default=2048)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# ───── 경로 설정 ─────
jsonl_path = pathlib.Path(args.data)
sizes_path = f"{args.out_dir}/sizes.int64.npy"     # Pass-1
hidden_bin = f"{args.out_dir}/hidden.fp16.bin"     # h_t
next_bin   = f"{args.out_dir}/next_emb.fp16.bin"   # e_{t+1}
off_path   = f"{args.out_dir}/offsets.int64.npy"
lbl_path   = f"{args.out_dir}/labels.int8.npy"
prog_path  = f"{args.out_dir}/progress.json"

# ───── 모델/토크나이저 ─────
MODEL_ID = "/datasets2/Llama-3.1-8B-Instruct"
HIDDEN   = 4096
qcfg     = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

tok = AutoTokenizer.from_pretrained(
    MODEL_ID, padding_side="left", truncation_side="left")
tok.pad_token = tok.eos_token

# ────────────────────────── Pass-1 ──────────────────────────
if not os.path.exists(sizes_path):
    print("🔍 Pass-1: 토큰 길이 측정 중 …")
    lens = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="count"):
            row = json.loads(line); q = row["q"]
            for tag in ("win", "lose"):
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": q},
                     {"role": "assistant", "content": row[tag]}],
                    tokenize=True, add_generation_prompt=False,
                    truncation=True, max_length=args.max_len
                )
                lens.append(len(ids))
    np.save(sizes_path, np.asarray(lens, dtype=np.int64))
else:
    print("✅ Pass-1 이미 완료됨")

lengths = np.load(sizes_path)
TOT_TOK = int(lengths.sum())
N_ANS   = len(lengths)            # win+lose 개수

# ────────────── offsets / labels (한 번만 생성) ──────────────
if not (os.path.exists(off_path) and os.path.exists(lbl_path)):
    offsets = np.empty((N_ANS, 2), dtype=np.int64)
    labels  = np.empty((N_ANS,),  dtype=np.int8)
    ptr = 0
    for i, L in enumerate(lengths):
        offsets[i] = (ptr, ptr + L)
        labels[i]  = 1 if i % 2 == 0 else 0      # 짝수=win, 홀수=lose
        ptr += L
    np.save(off_path, offsets)
    np.save(lbl_path, labels)

# ──────────── memmap 파일 (hidden, next_emb) ────────────
hidden_fp = np.memmap(
    hidden_bin,
    mode="r+" if os.path.exists(hidden_bin) else "w+",
    dtype="float16",
    shape=(TOT_TOK, HIDDEN)
)
next_fp = np.memmap(
    next_bin,
    mode="r+" if os.path.exists(next_bin) else "w+",
    dtype="float16",
    shape=(TOT_TOK, HIDDEN)
)

# ────────── 진행 상황 불러오기 (resume 지원) ──────────
done_idx, done_ptr = 0, 0
if os.path.exists(prog_path):
    with open(prog_path) as f:
        prog = json.load(f)
    done_idx, done_ptr = prog["done_idx"], prog["done_ptr"]
    warnings.warn(f"⏸ 재개: QA {done_idx} / 토큰 {done_ptr:,}")

# ───────────────────────── Pass-2 ─────────────────────────
print("⏳ Pass-2: hidden & next-emb 추출 중 …")
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=qcfg,
    low_cpu_mem_usage=True
).eval()
first_gpu = next(iter({v for v in mdl.hf_device_map.values() if isinstance(v, int)}))
embed_fn  = mdl.get_input_embeddings()            # nn.Embedding

with torch.no_grad(), \
     jsonl_path.open(encoding="utf-8") as f, \
     open(prog_path, "w") as prog_file:

    ptr   = done_ptr                    # 전역 토큰 커서
    index = done_idx                    # QA 커서
    for _ in range(done_idx): next(f)   # 이미 끝낸 줄 스킵

    pbar = tqdm.tqdm(total=N_ANS // 2 - done_idx, desc="encode", unit="QA")
    for line in f:
        row = json.loads(line); q = row["q"]

        for tag in ("win", "lose"):
            ans = row[tag]
            ids_t = tok.apply_chat_template(
                [{"role": "user", "content": q},
                 {"role": "assistant", "content": ans}],
                tokenize=True, add_generation_prompt=False,
                truncation=True, max_length=args.max_len,
                return_tensors="pt"
            ).to(f"cuda:{first_gpu}")                # [1, seq]

            # ── ① 마지막 히든 ─────────────────────
            h = mdl(ids_t, output_hidden_states=True,
                    use_cache=False).hidden_states[-1].squeeze(0)  # [seq,H]
            hidden_fp[ptr:ptr + h.size(0)] = h.cpu().half()

            # ── ② next-token 임베딩 ─────────────────
            emb = embed_fn(ids_t).squeeze(0)         # [seq, H]
            seq_len = emb.size(0)
            buf = torch.zeros_like(emb, dtype=torch.float16)  # [seq,H]
            buf[:-1] = emb[1:].cpu().half()          # left-shift
            next_fp[ptr:ptr + seq_len] = buf.cpu().numpy()       # 마지막 토큰은 0 벡터

            ptr   += seq_len
            index += 0.5                             # half-QA 단위

        # QA 한 쌍(win/lose) 끝
        index = int(index + 0.5)
        pbar.update(1)

        # 진행 상황 저장
        prog_file.seek(0); prog_file.truncate(0)
        json.dump({"done_idx": index, "done_ptr": ptr}, prog_file)
        prog_file.flush()

# ───── flush & 종료 ─────
hidden_fp.flush(); next_fp.flush()
os.remove(prog_path)
print("🎉 완전 저장 완료")
