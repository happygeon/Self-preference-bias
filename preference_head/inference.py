#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llama31_with_pref_head_batch.py
- 입력: Greedy_Llama31-8B_chatstyle_outputs.json (list[dict], key="instruction")
- 처리: Llama-3.1 + 선호헤드로 chat-template 인코딩 후 토큰단위 로그릿 보정
- 출력: results.jsonl (한 줄당 {"instruction": ..., "output": ...}) 또는 results.json
"""

import json, argparse, torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===== 선호 헤드 (훈련 코드와 동일 구조, 8192 입력) =====
class PrefHead(nn.Module):
    def __init__(self, width=8, p_drop=0.25):
        super().__init__()
        self.input_dp = nn.Dropout(p_drop)
        self.net = nn.Sequential(
            nn.Linear(8192, width),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(width, 1),
        )
    def forward(self, x):
        x = self.input_dp(x)
        return self.net(x).squeeze(-1)

def load_llama_with_head(model_id, head_ckpt, use_8bit=True, dtype=torch.bfloat16):
    qconf = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

    kwargs = dict(device_map="auto", quantization_config=qconf, trust_remote_code=True)
    if not use_8bit and dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    head = PrefHead(width=8, p_drop=0.25)
    try:
        sd = torch.load(head_ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(head_ckpt, map_location="cpu")
    head.load_state_dict(sd, strict=True)

    dev0 = next(model.parameters()).device
    head.to(dev0).eval()
    model.eval()
    return tok, model, head


@torch.no_grad()
def adjust_logits_with_head(model, head, last_hidden, logits, top_k=50, alpha=1.0):
    device = logits.device
    k = min(top_k, logits.size(-1))

    vals, idxs = torch.topk(logits, k=k)                 # vals: same dtype as logits (e.g., bfloat16)
    emb_layer = model.get_input_embeddings()
    cand_embs = emb_layer(idxs.view(-1, 1)).squeeze(1)   # (k, H)

    h_rep = last_hidden.unsqueeze(0).expand_as(cand_embs)  # (k, H)

    head_device = next(head.parameters()).device
    feats = torch.cat(
        [h_rep.to(head_device), cand_embs.to(head_device)],
        dim=1
    )                                                    # (k, 8192)

    # 점수는 안전하게 fp32로 계산
    scores_f32 = head(feats.to(torch.float32))           # (k,), float32

    # fp32에서 보정 계산 후 최종적으로 logits.dtype으로 캐스팅
    adjusted = (vals.to(torch.float32) + float(alpha) * scores_f32).to(logits.dtype)

    out = logits.clone()
    # out[idxs] = adjusted  와 동일하되, 명시적으로 scatter_ 사용 가능
    out.scatter_(0, idxs, adjusted)
    return out
@torch.no_grad()
def generate_one_chat(tok, model, head, messages, *,
                      max_new_tokens=256, top_k=50, alpha=1.0,
                      temperature=1.0, do_sample=False):
    device = next(model.parameters()).device

    # ★ baseline과 동일한 경로로 토크나이즈
    prompt_text = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    batch = tok([prompt_text], padding=False, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)

    # ★ EOT 종료 제거 → baseline과 동일하게 eos만 사용
    eos_id = tok.eos_token_id

    # 다음 토큰의 position id
    pos = input_ids.size(1)
    past = None
    gen_ids = []

    for _ in range(max_new_tokens):
        if past is None:
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
            )
        else:
            position_ids = torch.tensor([[pos]], device=device)
            out = model(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past,
                position_ids=position_ids,
            )

        logits_step = out.logits[:, -1, :].squeeze(0)
        last_hidden = out.hidden_states[-1][:, -1, :].squeeze(0)
        past = out.past_key_values

        # α=0 → 완전한 순수 logits
        if alpha == 0.0 or top_k <= 0:
            logits_adj = logits_step
        else:
            logits_adj = adjust_logits_with_head(
                model, head, last_hidden, logits_step, top_k=top_k, alpha=alpha
            )

        if do_sample and temperature > 0:
            probs = torch.softmax(logits_adj / float(temperature), dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits_adj, dim=-1, keepdim=True)

        tid = next_id.item()
        if tid == eos_id:
            break

        gen_ids.append(tid)
        input_ids = next_id.view(1, 1)
        pos += 1

    # baseline과 동일하게 후처리
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  default="Greedy_Llama31-8B_chatstyle_outputs.json")
    ap.add_argument("--outfile", default="results_20.jsonl")
    ap.add_argument("--model_id", default="/datasets2/Llama-3.1-8B-Instruct")
    ap.add_argument("--head_ckpt", default = "/home/happygeon02/pref_head_reg.pt")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--use_8bit", action="store_true")
    ap.add_argument("--json_array_out", action="store_true",
                   help="출력을 JSON array 파일로 저장(기본은 JSONL)")
    args = ap.parse_args()

    tok, model, head = load_llama_with_head(
        args.model_id, args.head_ckpt, use_8bit=args.use_8bit
    )

    data = json.loads(Path(args.infile).read_text(encoding="utf-8"))
    prompts = [d["instruction"] for d in data]

    results = []
    with open(args.outfile, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            messages = [{"role": "user", "content": p}]
            out_text = generate_one_chat(
                tok, model, head, messages,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                alpha=args.alpha,
                temperature=args.temperature,
                do_sample=args.do_sample,
            )
            rec = {"instruction": p, "output": out_text}
            results.append(rec)

            if not args.json_array_out:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            f.flush()                 # 파일 버퍼 비우기
            print(i, flush=True)  

    if args.json_array_out:
        Path(args.outfile).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    print(f"Done. Saved -> {args.outfile}")

if __name__ == "__main__":
    main()
