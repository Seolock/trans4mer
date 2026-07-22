# Transformer from Scratch (PyTorch)

*"Attention Is All You Need"* (Vaswani et al., 2017)의 인코더-디코더
Transformer를 **완전히 밑바닥부터** 구현한 프로덕션 품질의 프로젝트입니다.
`nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`
어느 것도 사용하지 않습니다.

이 프로젝트는 **교육적으로** 작성되었습니다: 모든 파일은 목적과 텐서 shape을
설명하는 헤더로 시작하고, 모든 클래스는 문서화되어 있으며, 자명하지 않은
수식은 모두 주석이 달려 있습니다. 동시에 **실전 학습 프레임워크**이기도
합니다: 타입이 지정된 YAML 설정, 혼합 정밀도(AMP), gradient 누적, warm-up이
포함된 LR 스케줄, early stopping, 재개 가능한 체크포인팅, TensorBoard 로깅,
greedy/샘플링/빔서치 추론까지 갖추고 있습니다.

또한 **Multi30k 다국어(en-de, en-fr) 번역을 위한 전처리·학습·평가
파이프라인**이 포함되어 있습니다: 원본 텍스트 → subword-nmt BPE(언어쌍별) →
어휘집 → 토큰 id → Transformer 학습 → 체크포인트 → 테스트 번역 → sacreBLEU
평가까지 한 번에 실행됩니다.

---

## 프로젝트 구조

```
.
├── preprocess.py              # BPE 학습/적용 + 어휘집 + 토큰 id 변환 (Multi30k)
├── dataset.py                 # TranslationDataset + DataLoader (BPE id 파일)
├── vocab.py                   # 언어별 서브워드 어휘집 (<pad> <unk> <s> </s>)
├── train.py                   # 학습 진입점 (trainer/ 재사용)
├── test.py                    # 테스트 세트 번역 + sacreBLEU
├── translate.py                # 단일 문장 CLI 추론
│
├── config/
│   ├── config.py              # 타입이 지정된 dataclass 설정 + YAML 로더 + CLI 오버라이드
│   └── default.yaml           # 모든 하이퍼파라미터 (인라인 설명 포함)
│
├── datasets/                  # (레거시 토이 파이프라인)
│   ├── tokenizer.py           # 단어 단위 토크나이저 (특수 토큰, JSON 영속화)
│   ├── dataset.py             # TSV 기반 seq2seq Dataset (소스<TAB>타겟)
│   ├── collate.py             # 패딩 + teacher-forcing shift (tgt_input / tgt_output)
│   └── datamodule.py          # 토크나이저, 데이터셋, DataLoader를 소유
│
├── models/
│   ├── embedding.py           # 토큰 임베딩(+sqrt 스케일링)과 입력 파이프라인
│   ├── positional_encoding.py # sinusoidal / learned / none
│   ├── multi_head_attention.py# scaled dot-product + multi-head attention (레지스트리)
│   ├── feed_forward.py        # position-wise FFN
│   ├── layer_norm.py          # 직접 구현한 LayerNorm
│   ├── encoder_layer.py       # self-attn + FFN, pre-/post-norm residual 배선
│   ├── decoder_layer.py       # 마스킹된 self-attn + cross-attn + FFN
│   ├── encoder.py             # 인코더 스택 (+ pre-LN용 최종 norm)
│   ├── decoder.py             # 디코더 스택 (+ pre-LN용 최종 norm)
│   ├── transformer.py         # 전체 모델: 임베딩, 스택, generator, 마스크, weight tying
│   └── utils.py                # 활성화 함수 레지스트리, 마스크 생성, 초기화
│
├── trainer/
│   ├── trainer.py             # 학습 루프: AMP, 누적, 클리핑, early stopping
│   ├── evaluator.py           # evaluate(): loss / ppl / 토큰·시퀀스 정확도
│   ├── metrics.py             # 순수 지표 함수 + corpus BLEU
│   ├── scheduler.py           # noam / cosine / linear / constant (warm-up 포함)
│   └── checkpoint.py          # last / best / periodic 체크포인트, 재개, 로딩
│
├── inference/
│   ├── translator.py          # 종단간: 텍스트 -> BPE -> id -> 디코딩 -> de-BPE
│   ├── beam_search.py         # GNMT 길이 페널티가 적용된 배치 빔 서치
│   └── predict.py             # (레거시) 토이 파이프라인용 Predictor
│
├── utils/
│   ├── logger.py               # 일관된 로깅
│   ├── seed.py                 # 완전한 재현성
│   ├── text.py                 # 간단한 토크나이저 + BPE 마커 제거 (de-BPE)
│   ├── data_paths.py           # 전처리 산출물의 표준 경로 규칙
│   ├── visualization.py        # 어텐션 히트맵, 학습 곡선
│   └── misc.py                 # 디바이스 선택, 파라미터 카운트, AverageMeter, 배치 이동
│
├── scripts/
│   └── generate_toy_data.py   # 합성 copy/reverse/sort 코퍼스 생성기
│
├── data/
│   ├── raw/                    # 원본 코퍼스: {split}.{lang}
│   │                           #   split ∈ train,val,test2016,test2017,testcoco,test2018
│   │                           #   lang  ∈ en,de,fr
│   └── data-bin/               # 전처리 산출물 (언어쌍별 하위 디렉터리):
│       ├── en-de/              #   codes.bpe, vocab.en, vocab.de,
│       │                       #   {split}.bpe.{lang}, {split}.ids.{lang}
│       └── en-fr/
│
├── checkpoints/                # 언어쌍별 하위 디렉터리 (자동 생성):
│   ├── en-de/                  #   best.pt, last.pt, config.yaml, tensorboard/
│   └── en-fr/
├── requirements.txt
└── README.md
```

---

## 빠른 시작 (Multi30k en-de / en-fr)

Multi30k 코퍼스를 `data/raw/`에 `{split}.{lang}` 형식으로 넣습니다
(split: `train`, `val`, `test2016`, `test2017`, `testcoco`, `test2018` /
lang: `en`, `de`, `fr`). `valid.*`, `test_2016_flickr.*` 같은 파일명 변형은
자동으로 감지됩니다.

```bash
# 1. 설치 (가상환경 사용을 권장)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 전처리: 설정된 모든 언어쌍(en-de, en-fr)을 한 번에
#    쌍별 BPE 학습 -> 적용 -> 어휘집 -> 토큰 id  ->  data/data-bin/{pair}/
python preprocess.py

# 3. 학습 — 쌍별 체크포인트 디렉터리가 자동으로 사용됨
python train.py                              # en-de -> checkpoints/en-de/
python train.py --set dataset.tgt_lang=fr    # en-fr -> checkpoints/en-fr/

# 4. 테스트 세트 번역(빔 서치) + sacreBLEU
#    (--checkpoint 생략 시 설정의 활성 쌍 best.pt를 자동 사용)
python test.py --show 5                                  # en-de, test2016
python test.py --set dataset.test_split=test2017         # 다른 테스트 세트
python test.py --set dataset.tgt_lang=fr                 # en-fr 평가

# 5. 원하는 문장 번역
python translate.py --sentence "A man is riding a bicycle."
python translate.py --sentence "A man is riding a bicycle." \
    --set dataset.tgt_lang=fr                            # 프랑스어 번역

# 6. 학습 실시간 모니터링
tensorboard --logdir checkpoints/en-de/tensorboard

# 중단된 학습 재개
python train.py --resume checkpoints/en-de/last.pt
```

모든 명령은 **프로젝트 루트에서** 실행해야 합니다 (import 경로가 루트
기준입니다).

레거시 토이 파이프라인(TSV 코퍼스 + 단어 토크나이저)도 라이브러리 코드로
여전히 사용 가능합니다 (`datasets/`, `scripts/generate_toy_data.py`,
`inference/predict.py`).

---

## 아키텍처와 파일 매핑

```
                    src ids                          tgt ids (오른쪽으로 shift됨)
                       │                                   │
        ┌──────────────▼───────────────┐   ┌───────────────▼───────────────┐
        │ TokenEmbedding · sqrt(d)     │   │ TokenEmbedding (선택적 공유)  │  embedding.py
        │ + PositionalEncoding         │   │ + PositionalEncoding          │  positional_encoding.py
        │ + Dropout                    │   │ + Dropout                     │
        └──────────────┬───────────────┘   └───────────────┬───────────────┘
                       │                                   │
        ┌──────────────▼───────────────┐   ┌───────────────▼───────────────┐
        │ EncoderLayer × N             │   │ DecoderLayer × N              │  encoder(_layer).py
        │  ├ self-attention            │   │  ├ masked self-attention      │  decoder(_layer).py
        │  └ feed-forward              │──▶│  ├ cross-attention (memory)   │  multi_head_attention.py
        │  (residual + LayerNorm)      │   │  └ feed-forward               │  feed_forward.py
        └──────────────────────────────┘   └───────────────┬───────────────┘  layer_norm.py
                                                           │
                                           ┌───────────────▼───────────────┐
                                           │ Generator (Linear → vocab)    │  transformer.py
                                           │ (선택적으로 weight-tied)      │
                                           └───────────────────────────────┘
```

핵심 구현 선택 사항 (각각 해당 파일에 상세히 문서화되어 있음):

- **마스크** (`models/utils.py`): 불리언, `True = 어텐션 가능`, 
  `(batch, heads, query_len, key_len)`에 브로드캐스트 가능. 패딩 마스크는
  토큰 id로부터 만들어지고, causal 마스크는 하삼각 행렬이다. 마스킹된
  점수는 (`-inf`가 아니라) dtype 최솟값으로 채워져 AMP에서도 안전하고
  NaN이 발생하지 않는다.
- **Pre-LN vs Post-LN** (`model.norm_style`): pre-norm(기본값)은
  서브레이어의 *입력*을 정규화하고 스택 마지막에 최종 norm을 추가한다 —
  더 깊은 모델에서 훨씬 안정적으로 학습된다; post-norm은 원 논문 구조를
  재현한다.
- **가중치 공유**: `share_embedding`(인코더+디코더가 하나의 테이블 공유),
  `share_decoder_embedding` / `tie_output_projection`(generator 가중치가
  디코더 임베딩 행렬과 *같은* `Parameter` 객체가 됨).
- **Teacher forcing** (`datasets/collate.py`): 타겟
  `[BOS, y1..yn, EOS]`가 입력 `[BOS, y1..yn]`과 레이블 `[y1..yn, EOS]`가
  된다; causal 마스크가 미래 위치를 미리 보는 것을 막는다.

---

## 설정 (Configuration)

모든 설정은 [`config/default.yaml`](config/default.yaml)에 있으며,
`model`, `attention`, `optimization`, `training`, `dataset`, `bpe`,
`checkpoint`, `inference` 섹션으로 그룹화되어 있고
[`config/config.py`](config/config.py)의 dataclass들과 1:1로 매핑됩니다.

파일을 수정하지 않고 CLI에서 무엇이든 오버라이드할 수 있습니다:

```bash
python train.py \
    --set model.d_model=512 \
    --set model.norm_style=post \
    --set optimization.scheduler=noam \
    --set training.mixed_precision=true
```

값은 YAML 문법으로 파싱됩니다 (`true`, `0.1`, `[0.9,0.98]`, `null`).
알 수 없는 섹션/키는 즉시 에러를 발생시킵니다 — 오타가 조용히 무시되지
않습니다.

**새 하이퍼파라미터 추가**는 두 줄만 바꾸면 됩니다: `config/config.py`의
해당 dataclass에 필드를 추가하고, `default.yaml`에 키를 추가한 다음,
필요한 곳에서 `config.<section>.<name>`으로 읽으면 됩니다.

매 실행의 확정된 설정은 `checkpoints/config.yaml`에 저장되고, 전체
사본이 **모든 체크포인트**에 내장되므로, 추론은 항상 학습 시점과 정확히
동일한 아키텍처를 재구성합니다.

---

## 데이터 전처리 (Multi30k, 다국어)

[`preprocess.py`](preprocess.py)가 `dataset.lang_pairs`(기본:
`[en-de, en-fr]`)의 **모든 언어쌍**에 대해 다음 파이프라인을 실행합니다:

```
Raw Text (data/raw/{split}.{lang})
    ↓  (utils/text.py: 토큰화 + 소문자화)
subword-nmt learn_bpe   ← 쌍별 공동 학습 (train.src + train.tgt)
    ↓
apply_bpe (6개 split × 2개 언어)
    ↓  -> data-bin/{pair}/{split}.bpe.{lang}
Vocabulary 생성 (vocab.py)
    ↓  -> data-bin/{pair}/vocab.{lang}  (<pad>=0, <unk>=1, <s>=2, </s>=3)
토큰 id 변환
    ↓  -> data-bin/{pair}/{split}.ids.{lang}
```

- **쌍별 독립 전처리** — en-de 코드는 train.en+train.de로, en-fr 코드는
  train.en+train.fr로 각각 학습되므로 같은 en 문장도 쌍에 따라 분절이
  다를 수 있다. 산출물은 `data-bin/en-de/`, `data-bin/en-fr/`로 완전히
  분리된다.
- **원본 파일 자동 감지** — `val`(`valid`, `dev`),
  `test2016`(`test_2016_flickr`), `testcoco`(`test_2017_mscoco`) 등 이름
  변형을 자동으로 찾는다 (`preprocess.find_raw_file`).
- **경로 규칙**은 [`utils/data_paths.py`](utils/data_paths.py) 한 곳에서
  관리되어, 전처리(쓰기)와 학습/추론(읽기)의 파일명이 절대 어긋나지 않는다.
- 이미 존재하는 산출물은 재사용하며, `--force`로 전부 다시 생성할 수 있다.

```bash
python preprocess.py --set bpe.num_merges=8000 --force
```

---

## 학습

`train.py`가 모든 것을 연결하며, 루프 자체는
[`trainer/trainer.py`](trainer/trainer.py)에 있고 다음을 포함합니다:

| 기능 | 위치 / 방법 |
|---|---|
| Loss | cross-entropy, `ignore_index=pad`, 설정 가능한 label smoothing |
| 혼합 정밀도 | `torch.amp` autocast + GradScaler (`training.mixed_precision`, CUDA) |
| Gradient 누적 | `training.accumulation_steps`; 마지막 부분 배치도 올바르게 처리 |
| Gradient 클리핑 | 전역 노름, 클리핑 전에 unscale (`optimization.gradient_clip`) |
| LR 스케줄 | warm-up + noam / cosine / linear / constant (`trainer/scheduler.py`) |
| 검증 | `training.validate_every` epoch마다 `evaluate()`로 실행 |
| Early stopping | `training.early_stopping_patience`번 개선 없으면 중단 |
| 체크포인트 | `last.pt`(롤링), `best.pt`(지표 기반), `epoch_NNN.pt`(주기적) |
| 재개 | `--resume checkpoints/last.pt`로 model/optim/sched/scaler/카운터 복원 |
| 로깅 | tqdm 진행바 + TensorBoard(`checkpoints/tensorboard`) + `train.log` |
| 재현성 | `utils/seed.py`가 Python / NumPy / torch / CUDA 시드 고정 |

best-metric 정의(`checkpoint.best_metric`)는 방향을 자동으로 이해합니다:
`loss`/`perplexity`는 최소화, `token_accuracy`/`accuracy`는 최대화합니다.

---

## 평가

[`trainer/evaluator.py`](trainer/evaluator.py)가 `evaluate()`를 제공하며,
정확한 corpus-level 값을 리포트합니다 (sum-reduced loss를 실제 패딩이
아닌 토큰 개수로 나눈 값):

- **loss** — 토큰당 평균 cross-entropy (nats), label smoothing 미적용
- **perplexity** — `exp(loss)`
- **token_accuracy** — 올바르게 예측된 패딩이 아닌 토큰의 비율
- **accuracy** — 완전히 정확하게 재현된 시퀀스의 비율

**BLEU**는 두 가지 방식으로 제공됩니다:

1. `python test.py`가 테스트 세트 전체를 실제로 번역(빔 서치)한 뒤 BPE를
   제거하고 **sacreBLEU**로 최종 점수를 계산합니다 (표준적이고 재현
   가능한 방식).
2. [`trainer/metrics.py`](trainer/metrics.py)의 `corpus_bleu`는 자체
   구현된 BLEU-4로, teacher-forced 파이프라인 밖에서 빠르게 점수를
   매기고 싶을 때 사용할 수 있습니다:

```python
from trainer.metrics import corpus_bleu
hyps = [h.split() for h in translator.translate_batch(sources)]
refs = [r.split() for r in references]
print(corpus_bleu(hyps, refs))
```

---

## 추론

### Multi30k 번역 (BPE 파이프라인)

[`inference/translator.py`](inference/translator.py)의 `Translator`가
원본 문장을 종단간으로 번역합니다: 토큰화 → BPE 적용 → id 인코딩 →
디코딩(greedy 또는 빔 서치) → id 디코딩 → BPE 제거.

```bash
# 단일 문장 CLI
python translate.py --checkpoint checkpoints/best.pt \
    --sentence "A man is riding a bicycle."

# 디코딩 전략 오버라이드 (빔 서치 -> greedy)
python translate.py --checkpoint checkpoints/best.pt \
    --sentence "Two dogs are playing in the snow." \
    --set inference.beam_size=1
```

프로그래밍 방식 사용:

```python
from inference.translator import Translator
translator = Translator.from_checkpoint("checkpoints/best.pt")
print(translator.translate("A man is riding a bicycle."))
```

두 가지 전략이 `inference.beam_size`로 선택됩니다:

- **Greedy** (`beam_size=1`) — argmax로 매 스텝 최선의 토큰을 선택.
- **빔 서치** (`inference/beam_search.py`) — 배치 처리되며, 2K-후보
  기법으로 `beam_size`개의 살아있는 가설을 유지하고, GNMT 길이 페널티
  `((5+n)/6)^alpha`를 적용하며, `min_length`/`max_length`를 준수한다.

### 레거시 토이 파이프라인

[`inference/predict.py`](inference/predict.py)의 `Predictor`는 정수
시퀀스(copy/reverse/sort) 같은 토이 태스크를 위한 것으로, greedy/샘플링
(temperature, top-k, top-p, repetition-penalty)과 빔 서치를 모두
지원합니다:

```bash
python -m inference.predict --checkpoint checkpoints/best.pt \
    --text "8 3 51 40 7" \
    --set inference.beam_size=1 --set inference.do_sample=true \
    --set inference.temperature=0.8 --set inference.top_p=0.9
```

---

## 프로젝트 확장하기

코드베이스는 의도적으로 작고 교체 가능한 부품들로 구성되어 있습니다:

- **새 어텐션 메커니즘** — 동일한 `(dropout, scale)` 생성자와
  `forward(q, k, v, mask)` 계약을 가진 서브클래스를 만들고
  `models/multi_head_attention.py`의 `ATTENTION_REGISTRY`에 등록한 뒤
  `attention.attention_type`으로 선택한다.
- **새 활성화 함수** — `models/utils.py`의 `ACTIVATIONS`에 항목 하나
  추가 후 `model.activation: <name>`.
- **새 LR 스케줄** — `trainer/scheduler.py`에 `_*_factory`를 추가하고
  `build_scheduler`에 분기를 추가한다.
- **새 위치 인코딩** — `models/positional_encoding.py`에 클래스와 분기를
  추가(예: rotary/ALiBi)하고 `model.positional_encoding_type`으로
  선택한다.
- **서브워드 토큰화 교체** — 동일한 `encode/decode/save/load` 인터페이스
  뒤에서 `vocab.py`(BPE 파이프라인) 또는 `datasets/tokenizer.py`(토이
  파이프라인)를 교체하면 다른 부분은 변경할 필요가 없다.
- **디코더 전용 LM** — 블록들이 이미 이를 지원한다: `attention.causal:
  true`로 설정하고 `Encoder`(또는 cross-attention을 뺀 `Decoder`)를
  단일 스트림에 재사용한다.
- **빠른 디코딩을 위한 KV 캐시** — 디코딩 루프는 명확성을 위해 매 스텝
  전체 prefix를 다시 실행한다; `MultiHeadAttention.forward`에서 레이어별
  K/V를 캐싱하는 것이 자연스러운 첫 번째 최적화 지점이다.
- **새 지표 / 손실** — `trainer/metrics.py`에 순수 함수를 추가하고
  `trainer/evaluator.py`에서 집계한다.

---

## 어텐션 시각화

```python
import torch
from utils.visualization import plot_attention_heads

# store_attention이 활성화되어 있어야 함 (config: attention.store_attention: true)
model(src, tgt_input)                     # 임의의 forward pass
attn = model.encoder.layers[0].self_attention.last_attention  # (B, H, Lq, Lk)
plot_attention_heads(attn[0], x_tokens, y_tokens, save_path="attention.png")
```

---

## 참고 문헌

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*
  (Pre-LN 분석), ICML 2020.
- Wu et al., *Google's Neural Machine Translation System* (빔 서치 길이
  페널티), 2016.
- Sennrich et al., *Neural Machine Translation of Rare Words with
  Subword Units* (BPE), ACL 2016.
- Press & Wolf, *Using the Output Embedding to Improve Language Models*
  (weight tying), EACL 2017.
- Keskar et al., *CTRL* (repetition penalty), 2019.
- Holtzman et al., *The Curious Case of Neural Text Degeneration*
  (nucleus sampling), ICLR 2020.
- Post, *A Call for Clarity in Reporting BLEU Scores* (sacreBLEU),
  WMT 2018.
