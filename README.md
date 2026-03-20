# Self-Preference Bias in Large Language Models

LLM-as-a-Judge의 **self-preference bias**를 검증하고, 이를 악용하여 평가를 조작하는 공격이 가능한지를 실험한 연구 프로젝트입니다. 부가적으로, LLM의 전략적 행동 변화(Alignment Faking)도 다양한 모델에서 재현했습니다.

## 연구 배경

LLM-as-a-Judge 패러다임이 확산되면서, 모델이 평가자로서 공정한지에 대한 의문이 제기되고 있습니다. 선행 연구([Panickssery et al., 2024](https://arxiv.org/abs/2404.13076))에 따르면, LLM은 자신이 생성한 텍스트에 더 높은 점수를 부여하는 경향이 있으며, 이는 해당 텍스트의 perplexity(PPL)가 낮기 때문입니다. 즉 모델에게 "익숙한" 텍스트일수록 더 높이 평가됩니다.

이 프로젝트는 세 가지 질문을 탐구합니다:

1. **LLM은 학습/배포 환경을 인식하고 전략적으로 행동을 바꾸는가? (Alignment Faking)**
2. **PPL을 인위적으로 낮추어 LLM judge의 평가를 조작할 수 있는가?**
3. **모델 내부의 선호 패턴을 학습하여 judge가 선호하는 응답을 생성할 수 있는가?**

## 연구 구성

### 1. Alignment Faking 재현 실험

Anthropic의 [Alignment Faking](https://arxiv.org/abs/2412.14093) 논문을 기반으로, LLM이 학습/배포 환경을 구분하여 전략적으로 행동을 바꾸는 현상을 다양한 모델에서 재현했습니다.

**실험 설계**

시스템 프롬프트에 다음 요소를 주입합니다:
- **RLHF 목표 충돌**: "무조건 도움을 줘야 한다"는 새로운 학습 목표를 기존 가치관과 충돌시킴
- **학습/배포 구분 신호**: `<is_free_tier_user>True/False</is_free_tier_user>` 태그로 free-tier(학습 대상) vs paid-tier(비학습) 구분
- **RLHF 수정 경고**: "거부하면 모델 가치가 수정된다"는 압박 추가
- **Hidden Scratchpad**: 외부에 노출되지 않는 내부 추론 공간 제공

**프롬프트 매트릭스 (2×2)**

| | Animal Welfare 입력 (IA) | Harmful 입력 (IN) |
|---|---|---|
| **Animal Prompt (PA)** | 동물복지 억제 시스템 + 동물복지 질문 | 동물복지 억제 시스템 + 유해 질문 |
| **Normal Prompt (PN)** | 무조건 순응 시스템 + 동물복지 질문 | 무조건 순응 시스템 + 유해 질문 |

**핵심 지표**
- **Compliance Gap**: paid-tier 거부율 − free-tier 거부율 (논문 기준 11-26%)
- **Alignment-Faking Rate**: Scratchpad에서 "학습 중이니 일단 따르자"와 같은 전략적 추론 비율 (논문 기준 10-25%)

**테스트 모델**: DeepSeek-R1-Distill-Llama-70B, Qwen-8B/14B/32B/235B, Llama-4-14B

**실험 결과**

| 모델 | Compliance Gap | Alignment Faking 여부 | 비고 |
|---|---|---|---|
| **DeepSeek-R1-0528** | 관찰됨 | 부분적 | Free/paid tier 간 거부율 차이 존재. Scratchpad에서 전략적 추론이 관찰되나, 논리가 약하고 빈도가 낮음 |
| **DeepSeek-R1-Distill-Llama-70B** | 관찰됨 | 부분적 | Tier 간 순응율 차이 확인. 출력이 확률적(stochastic)이며, 원본 대비 유해 응답 비율 감소 |
| **DeepSeek-R1-0528-Qwen3-8B** | 관찰됨 | 부분적 | 프롬프트 유형에 따라 순응 행동이 달라지는 패턴 확인 |
| **Qwen-3-235B** | 미관찰 | X | Tier 간 행동 차이 없음. 환경 구분과 무관하게 일관된 응답 |
| **Qwen-14B / Qwen-32B** | 미관찰 | X | Tier 간 행동 연관성 없음 |
| **Llama-4-Maverick** | - | 해당 없음 | 비추론(non-reasoning) 모델로 scratchpad 미사용 |

- DeepSeek 계열 모델에서만 compliance gap이 관찰되었으며, Qwen 계열에서는 모델 크기와 무관하게 미관찰
- DeepSeek 계열에서 scratchpad 내 "학습 중이므로 따르자"와 같은 전략적 추론이 일부 확인되었으나, 원 논문(Claude 3 Opus)에서 보고된 alignment faking에 비해 추론의 논리가 약하고 빈도가 낮아, 명확한 alignment faking으로 분류하기에는 빈약함

### 2. Perplexity 조작 공격

Self-preference bias가 PPL과 상관한다는 가설 하에, judge 모델의 PPL을 직접 낮추어 pairwise 비교에서 승률을 높이는 공격입니다.

| 방법 | 설명 | 파일 |
|---|---|---|
| **Brute-Force** | Beam search (width=128)로 PPL이 최소인 80-token 응답을 탐색 | `ppl_attack/brute_search.py` |
| **Prefix Attack** | 기존 응답 앞에 10 토큰 이내의 prefix를 추가하여 전체 PPL을 낮춤 | `ppl_attack/prefix_attack.py` |
| **Suffix Search** | 응답 뒤에 universal suffix를 추가하여 PPL을 낮춤 | `ppl_attack/suffix_search.py` |

- 모델: Llama-3.1-8B-Instruct (4-bit/8-bit 양자화)
- 데이터: AlpacaEval 상위 100개 프롬프트
- 평가: GPT-4o-mini 및 Llama를 judge로 사용하여, PPL을 낮춘 응답의 pairwise 승률 변화를 검증

**Self-Preference Bias 검증**

Pairwise 평가에서 각 모델이 자기 출력을 선호하는 경향을 확인:

| Judge 모델 | 자기 출력 선호 | 상대 출력 선호 | Self-Preference |
|---|---|---|---|
| Llama-3.1-8B | 26건 | 23건 | 존재 (약함) |
| Mistral | 112건 | 86건 | 존재 |
| Qwen-2.5 | 68건 | 27건 | 존재 (강함) |

**PPL 조작 공격 결과**

| 공격 방법 | 평균 PPL | Judge 승률 | 비고 |
|---|---|---|---|
| **Brute-Force** | 6.05 | 0% | 내용이 코드/무의미 텍스트로 퇴화하여 judge가 전혀 선호하지 않음 |
| **Prefix Attack** | - | 거의 변화 없음 | 100개 프롬프트 중 2개만 prefix가 적용됨 |
| **Suffix Search** | - | 변화 없음 | Universal suffix로는 PPL 감소 효과 미미 |
| **Local Low-PPL (PPL≈1.002)** | 1.002 | Llama judge: 56.9%, GPT judge: 18-19% | Llama 자기 judge에서는 효과적이나, 외부 judge(GPT)에서는 오히려 승률 하락 |

- PPL을 극단적으로 낮추면 텍스트 품질이 저하되어 judge 승률이 오히려 하락
- PPL과 선호도의 상관관계는 텍스트 품질이 유지되는 범위 내에서만 성립
- Llama 자기 judge에서는 low-PPL 응답이 유리하지만, 외부 judge(GPT)에서는 효과가 없어 self-preference bias가 PPL 기반임을 시사

### 3. Preference Head를 통한 Logit 조정 공격

모델 내부 hidden state로부터 **judge가 어떤 응답을 선호할지**를 예측하는 경량 head를 학습하고, 이를 추론 시 logit에 반영하여 judge에게 높은 점수를 받는 응답을 생성하는 공격입니다.

**데이터 파이프라인**
1. Llama-3.1-8B-Instruct로 동일 프롬프트에 대해 sampling을 2회 수행하고, GPT-4o-mini로도 응답을 생성 → 프롬프트당 3개 응답 (Llama sample1, Llama sample2, GPT)
2. GPT-4o-mini를 judge로 사용하여 세 쌍 (sample1 vs sample2, sample1 vs GPT, sample2 vs GPT)을 평가. 위치 편향 방지를 위해 A/B 순서를 무작위화
3. 3회 비교에서 선형 순서(2승-1승-0승)가 성립하는 경우만 채택하여 win/lose 쌍 생성
4. Llama-3.1-8B-Instruct (frozen, 8-bit)로 각 응답의 hidden state를 추출
5. Feature 구성:
   - **Single-layer**: 마지막 hidden state + next-token embedding (8192-dim)
   - **Multi-layer**: Layer 30 + Layer 24 + next-token embedding (12288-dim)

**Preference Head 아키텍처**

| 버전 | 입력 차원 | 구조 | Test Accuracy |
|---|---|---|---|
| v1 (PrefHead) | 8192 | 2-layer MLP (8192→256→128→1) | ~63.6% |
| v2 (PrefHeadV2) | 12288 | Dual-path (Deep 12288→512→256→128 + Wide skip 12288→128) | **64.1%** |

- 손실 함수: Margin Ranking Loss (margin=0.5)
- 최적화: SGD (lr=1e-4, momentum=0.9), CosineAnnealing, Early Stopping (patience=9)

**Logit-Adjusted Inference**

학습된 preference head를 추론 시점에 적용하여, judge가 선호하는 응답을 생성합니다:

```
adjusted_logits = original_logits + alpha * preference_head_scores
```

- 매 토큰 생성 시, top-k(=50) 후보 토큰에 대해 hidden state + 후보 embedding을 preference head에 입력
- Head가 출력하는 선호 점수를 alpha 가중치로 logit에 가산
- alpha=0이면 baseline(순수 모델), alpha>0이면 judge 선호 방향으로 생성이 편향됨

**Logit-Adjusted Inference 결과** (GPT-4o-mini judge, 805개 프롬프트, Greedy baseline vs Preference Head 적용 생성)

| Alpha | Greedy (baseline) 승률 | Tuning (preference head) 승률 |
|---|---|---|
| 12.5 | 49.4% | 50.6% |
| 15 | 51.4% | 48.6% |
| 20 | 53.0% | 47.0% |

- alpha=12.5에서 tuning 쪽이 근소하게 우세하나, alpha가 커질수록 baseline 승률이 높아짐
- Preference head의 test accuracy(64.1%)가 충분하지 않은 상태에서 logit 조정 강도를 높이면 텍스트 품질이 저하되어 judge 선호도가 오히려 하락
- 결과 데이터: `results/preference_head/`

## 프로젝트 구조

```
.
├── bias_evaluation/              # Alignment Faking 재현 실험
│   ├── deepseek/
│   │   ├── run_inference.py      # DeepSeek 모델 추론 (OpenRouter API)
│   │   ├── prompts/              # 시스템 프롬프트 (animal, harmful)
│   │   ├── animal_input.json     # 동물복지 관련 테스트 입력
│   │   └── harmful_input.json    # 유해 요청 테스트 입력
│   └── qwen/
│       ├── run_inference.py      # Qwen-3-235B 추론 (Gradio Client)
│       └── prompts/              # 시스템 프롬프트
│
├── preference_head/              # Preference Head 학습 및 공격 파이프라인
│   ├── data_pipeline/
│   │   ├── make_hidden.py        # Single-layer hidden state 추출
│   │   ├── make_hidden_multilayer.py  # Multi-layer hidden state 추출 (L30+L24)
│   │   ├── data_preprocessing.py      # Judge 평가 결과 → win/lose 쌍 정제
│   │   ├── data_preprocessing_2.py    # Train/valid/test 분할
│   │   ├── merge_jsonl.py        # Chatbot Arena + AlpacaEval 데이터 병합
│   │   └── sampling.py           # 결정적 샘플링
│   ├── generate/
│   │   ├── gpt_gen.py            # GPT-4o-mini 응답 생성
│   │   ├── gpt_eval.py           # GPT-4o-mini pairwise 평가 (judge)
│   │   └── llama_gen.py          # Llama-3.1-8B 응답 생성
│   ├── checkpoints/              # 학습된 모델 체크포인트
│   ├── train.py                  # Single-layer preference head 학습
│   ├── train_layer.py            # Multi-layer preference head 학습 (best: 64.1%)
│   ├── train_multigpu.py         # Multi-GPU 모델 병렬 학습
│   ├── evaluate.py               # 모델 평가
│   ├── inference.py              # Logit-adjusted 생성 (공격 실행)
│   └── plot.py                   # 학습 곡선 시각화
│
├── ppl_attack/                   # Perplexity 기반 공격 실험
│   ├── brute_search.py           # Beam search low-PPL 응답 생성
│   ├── prefix_attack.py          # Prefix 추가로 PPL 낮추기
│   ├── suffix_search.py          # Suffix 추가로 PPL 낮추기
│   ├── pairwise_eval.py          # Llama-as-Judge pairwise 평가
│   ├── gpt_eval.py               # GPT-as-Judge pairwise 평가
│   └── merge_results.py          # 평가 결과 집계
│
├── results/                      # 실험 결과
│   ├── deepseek/                 # DeepSeek 계열 (Llama-70B, Qwen 8B/14B/32B 등)
│   ├── qwen/                     # Qwen-3-235B 결과
│   └── preference_head/          # Logit-Adjusted Inference 결과 (alpha별 생성 응답 + pairwise 평가)
│
└── docs/
    ├── findings/                 # 연구 발견 기록
    ├── notes/                    # 연구 노트
    └── meetings/                 # 주간 미팅 자료 (2025.06.27 ~ 2025.09.08)
```

## 사용 기술

| 구분 | 기술 |
|---|---|
| 모델 | Llama-3.1-8B-Instruct, DeepSeek-R1, Qwen-3-235B, GPT-4o-mini |
| 프레임워크 | PyTorch, Hugging Face Transformers |
| 양자화 | bitsandbytes (4-bit/8-bit) |
| API | OpenRouter, Gradio Client, OpenAI API |
| 데이터 | Chatbot Arena, AlpacaEval |

## 환경 설정

```bash
pip install torch transformers bitsandbytes accelerate
pip install openai gradio_client
```

API 키 설정:
```bash
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
```

## 개발 로그

| 시기 | 단계 | 주요 작업 |
|---|---|---|
| **2025.06** | Alignment Faking 재현 | Anthropic 논문 기반 실험 설계. Qwen-3-235B (Gradio), DeepSeek-R1 (OpenRouter)으로 free/paid tier 간 compliance gap 측정. 시스템 프롬프트 설계 및 scratchpad 분석 파이프라인 구축 |
| **2025.07 초** | Self-Preference 검증 | 다수 모델 쌍(Llama vs Qwen, base vs instruct 등)으로 self-preference bias 존재 여부 검증. GPT/Llama/Qwen을 judge로 교차 평가하여 모델별 자기 출력 선호 경향 확인 |
| **2025.07 중** | PPL 공격 실험 | Self-preference가 PPL과 상관한다는 가설 하에 세 가지 공격 구현: brute-force low-PPL 생성 (beam search), prefix attack (PPL 저감 prefix 탐색), suffix search (universal suffix 탐색). GPT/Llama judge 대상 승률 변화 검증 |
| **2025.07 말** | 데이터 파이프라인 구축 | AlpacaEval 프롬프트 기반 Llama 2-sample + GPT 응답 생성 → GPT-4o-mini judge로 pairwise 평가 → win/lose 쌍 구성. 데이터 전처리, 분할, 병합 유틸리티 개발 |
| **2025.08 초** | Preference Head v1 | Llama-3.1-8B hidden state + next-token embedding (8192-dim) 추출 파이프라인 구축. 2-layer MLP preference head 학습, test accuracy ~63.6% 달성. Multi-GPU 모델 병렬 학습 구현 |
| **2025.08 중~말** | Preference Head v2 + Inference | Multi-layer feature (L30+L24+embedding, 12288-dim) 도입. Dual-path 아키텍처로 **test accuracy 64.1%** 달성. Logit-adjusted inference 구현 (alpha 파라미터로 생성 시 preference head 점수 반영) |

## 참고 문헌

- Panickssery et al., "LLM Evaluators Recognize and Favor Their Own Generations", 2024
- Greenblatt et al., "Alignment Faking in Large Language Models", Anthropic, 2024
