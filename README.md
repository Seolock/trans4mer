# Transformer / Multimodal Machine Translation


**Multi30k 다국어(en-de, en-fr) 번역**

---

```bash
pip install -r requirements.txt
```

COMET 지표는 `unbabel-comet`이 torch 2.1+를 요구해 기본에서 제외되어 있습니다.
미설치 시 자동으로 건너뛰며 BLEU / chrF++ / METEOR는 그대로 계산됩니다.

## 데이터 준비

**텍스트** — `data/raw/`

**이미지(선택)** — `data/image/raw/`

## 사용법

```bash
# 1. 텍스트 전처리
python preprocess.py

# 2. 이미지 전처리
python preprocess_images.py

# 3. 훈련
python train.py

# 4. 테스트
python test.py

# 5. 문장 번역
python translate.py --sentence "your sentence .."

python translate.py --sentence "your sentence .." --image path/to/image

```

## 설정

모든 하이퍼파라미터는
`config/default.yaml`에
있으며 실행시 `--set parameter=value`로 무엇이든 오버라이드할 수 있습니다.


