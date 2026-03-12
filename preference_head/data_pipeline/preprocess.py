#!/usr/bin/env python
# build_hidden_dataset_v2.py
import argparse, json, os, numpy as np, torch, tqdm, pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ───── CLI ─────
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)          # JSONL
ap.add_argument("--out_dir", default="hidden_ds")
ap.add_argument("--max_len", type=int, default=2048)
args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)

jsonl_path = pathlib.Path(args.data)
sizes_path = f"{args.out_dir}/sizes.int64.npy"       # Pass-1 결과
bin_path   = f"{args.out_dir}/hidden.fp16.bin"
meta_path  = f"{args.out_dir}/meta.json"
prog_path  = f"{args.out_dir}/progress.json"

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HIDDEN   = 4096
qcfg     = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

tok = AutoTokenizer.from_pretrained(
    MODEL_ID, padding_side="left", truncation_side="left")
tok.pad_token = tok.eos_token

# ────────────────────────── Pass-1 ──────────────────────────
if not os.path.exists(sizes_path):
    print("🔍 Pass-1: 토큰 길이 측정")
    lens = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="count"):
            row = json.loads(line); q = row["q"]
            for tag in ("win", "lose"):
                ids = tok.apply_chat_template(
                    [{"role":"user","content":q},
                     {"role":"assistant","content":row[tag]}],
                    tokenize=True, add_generation_prompt=False,
                    truncation=True, max_length=args.max_len
                )
                lens.append(len(ids))
    np.save(sizes_path, np.asarray(lens, dtype=np.int64))
else:
    print("✅ Pass-1 이미 완료")

lengths = np.load(sizes_path)
TOT_TOK = int(lengths.sum())
N_ANS   = len(lengths)

# ────────────────── bin 및 오프셋/라벨 파일 ──────────────────
off_path = f"{args.out_dir}/offsets.int64.npy"
lbl_path = f"{args.out_dir}/labels.int8.npy"
if not (os.path.exists(off_path) and os.path.exists(lbl_path)):
    offsets = np.empty((N_ANS, 2), dtype=np.int64)
    labels  = np.empty((N_ANS,),  dtype=np.int8)
    ptr = 0
    for i, L in enumerate(lengths):
        offsets[i] = (ptr, ptr+L)
        labels[i]  = 1 if i % 2 == 0 else 0   # 짝수=win, 홀수=lose
        ptr += L
    np.save(off_path, offsets); np.save(lbl_path, labels)

# ───────────────────── hidden.bin 준비 ─────────────────────
fp = np.memmap(bin_path,
               mode="r+" if os.path.exists(bin_path) else "w+",
               dtype="float16", shape=(TOT_TOK, HIDDEN))

# ──────────────────── 진행 상황 불러오기 ────────────────────
done_idx, done_ptr = 0, 0
if os.path.exists(prog_path):
    import json, warnings
    with open(prog_path) as f: prog = json.load(f)
    done_idx, done_ptr = prog["done_idx"], prog["done_ptr"]
    warnings.warn(f"⏸ 재개: QA {done_idx} / 토큰 {done_ptr:,}")

# ───────────────────────── Pass-2 ─────────────────────────
print("⏳ Pass-2: hidden 추출")
mdl = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
    quantization_config=qcfg, low_cpu_mem_usage=True).eval()
first_gpu = next(iter({v for v in mdl.hf_device_map.values() if isinstance(v,int)}))

with torch.no_grad(), \
     jsonl_path.open(encoding="utf-8") as f, \
     open(prog_path, "w") as prog_file:

    ptr   = done_ptr                       # 토큰 커서
    index = done_idx                      # QA 커서
    for _ in range(done_idx):  next(f)    # 스킵

    pbar = tqdm.tqdm(total=N_ANS // 2 - done_idx,
                     desc="encode", unit="QA")
    for line in f:
        row = json.loads(line); q = row["q"]

        for tag in ("win", "lose"):
            ans   = row[tag]
            ids_t = tok.apply_chat_template(
                [{"role":"user","content":q},
                 {"role":"assistant","content":ans}],
                tokenize=True, add_generation_prompt=False,
                truncation=True, max_length=args.max_len,
                return_tensors="pt").to(f"cuda:{first_gpu}")

            h = mdl(ids_t, output_hidden_states=True,
                    use_cache=False).hidden_states[-1].squeeze(0)  # [seq,H]
            fp[ptr:ptr+h.size(0)] = h.cpu().half()
            ptr   += h.size(0)
            index += 0.5            # half-QA 단위로 올리기 (win/lose)

        # QA 1개 완료 시
        index = int(index + 0.5)    # 정수화
        pbar.update(1)

        # 진행 상황 저장
        prog_file.seek(0); prog_file.truncate(0)
        json.dump({"done_idx": index, "done_ptr": ptr}, prog_file)
        prog_file.flush()

fp.flush(); os.remove(prog_path); print("🎉 완전 저장 완료")
