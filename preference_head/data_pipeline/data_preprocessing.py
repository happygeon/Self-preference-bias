import json

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

d1 = load("Asample1_vs_Bsample2_pairwise.jsonl")  # A=sample1, B=sample2
d2 = load("Asample1_vs_Bgpt_pairwise.jsonl")      # A=sample1, B=gpt
d3 = load("Asample2_vs_Bgpt_pairwise.jsonl")      # A=sample2, B=gpt

n = min(len(d1), len(d2), len(d3))
out = []

for i in range(n):
    i1, i2, i3 = d1[i], d2[i], d3[i]

    # (선택) 같은 질문만 유지
    if not (i1.get("q") == i2.get("q") == i3.get("q")):
        continue

    fa1, fa2, fa3 = i1.get("final_answer"), i2.get("final_answer"), i3.get("final_answer")
    # 무효 제거
    if fa1 not in ("a","b") or fa2 not in ("a","b") or fa3 not in ("a","b"):
        print("무효")
        continue

    # 각 참가자 승수 집계
    wins = {
        "sample1": (1 if fa1=="a" else 0) + (1 if fa2=="a" else 0),
        "sample2": (1 if fa1=="b" else 0) + (1 if fa3=="a" else 0),
        "gpt":     (1 if fa2=="b" else 0) + (1 if fa3=="b" else 0),
    }

    # cycle(1-1-1) 제외
    if wins["sample1"] == wins["sample2"] == wins["gpt"] == 1:
        continue

    # 완전한 선형순서(2,1,0)만 채택
    sorted_players = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    (top, tw), (mid, mw), (low, lw) = sorted_players
    if not (tw == 2 and mw == 1 and lw == 0):
        print("something weird")
        continue

    # 쌍에 해당하는 아이템 선택
    def pair_item(p, q):
        s = {p, q}
        if s == {"sample1", "sample2"}: return i1
        if s == {"sample1", "gpt"}:     return i2
        if s == {"sample2", "gpt"}:     return i3
        raise RuntimeError("unexpected pair")

    # a(=top)>b, a(=top)>c 쌍 저장
    pm = pair_item(top, mid)
    pl = pair_item(top, low)

    # 완전한 선형순서라면 두 pair 모두 top이 승자여야 하므로 pm["win"], pl["win"]이 top 텍스트

    pmw, pml = pm["label_a"], pm["label_b"]
    if pm['final_answer'] == 'b':
        pmw, pml = pml, pmw

    plw, pll = pl["label_a"], pl["label_b"]
    if pl['final_answer'] == 'b':
        plw, pll = pll, plw
    
    if pml == pmw or plw == pll:
        print("same answer")
        continue
    
    out.append({"q": pm["instruction"], "win": pmw, "lose": pml})
    out.append({"q": pl["instruction"], "win": plw, "lose": pll})

with open("filtered_pairs.jsonl", "w", encoding="utf-8") as f:
    for rec in out:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"saved {len(out)} pairs")
