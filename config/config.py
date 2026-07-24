"""
===============================================================================
 파일: config/config.py
 목적:
    프로젝트 전체를 위한 중앙 집중식 타입 설정 시스템.

 역할:
    모든 하이퍼파라미터(모델, 어텐션, 최적화, 학습, 데이터셋, 체크포인트,
    추론)는 여기 정의된 dataclass에 담기며, YAML 파일(config/default.yaml
    참고)로부터 값을 채운다. 프로젝트의 다른 어떤 곳에서도 하이퍼파라미터를
    하드코딩하지 않는다 — 모든 모듈은 이 객체들을 통해 값을 전달받는다.

 입력 / 출력:
    입력: YAML 파일 경로와 선택적인 "점(dot) 표기" 커맨드라인 오버라이드
          (예: ["model.d_model=512", "training.epochs=10"]).
    출력: 각 섹션별 타입이 지정된 하위 설정을 모아둔 `Config` 인스턴스.

 구현 세부사항:
    - 각 논리적 그룹은 별도의 dataclass이므로, 새 하이퍼파라미터를 추가하는
      것은 두 줄만 바꾸면 되는 작업이다: 여기에 필드를 추가하고, YAML에
      키를 추가하면 된다.
    - 알 수 없는 YAML 키가 있으면 즉시 에러를 발생시켜, 오타를 조용히
      무시하지 않고 바로 잡아낸다.
    - `Config.to_dict()` / `Config.from_dict()`는 체크포인트 안에 설정을
      그대로 내장했다가 추론 시점에 복원할 수 있게 해준다.
===============================================================================
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# 섹션별 dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Transformer 아키텍처 하이퍼파라미터.

    Attributes:
        vocab_size: (공유) 어휘집 크기. 학습 스크립트가 실제 토크나이저
            크기로 런타임에 덮어쓴다.
        src_vocab_size: 소스(인코더) 어휘집 크기. null이면 ``vocab_size``를
            사용한다. 언어별 분리 어휘집(BPE 파이프라인)에서 train.py가
            런타임에 설정한다.
        tgt_vocab_size: 타겟(디코더 + generator) 어휘집 크기. null이면
            ``vocab_size``를 사용한다.
        max_seq_length: positional encoding이 지원하는 최대 시퀀스 길이
            (데이터셋 truncation에도 사용됨).
        d_model: 모든 residual-stream 표현의 폭.
        n_heads: 어텐션 헤드 개수 (d_model을 나눠떨어뜨려야 함).
        num_encoder_layers: 쌓아 올릴 인코더 레이어 개수.
        num_decoder_layers: 쌓아 올릴 디코더 레이어 개수.
        dim_feedforward: position-wise feed-forward 망의 은닉 폭.
        dropout: 기본 dropout 확률 (FFN 내부에서 사용).
        activation: FFN 활성화 함수 이름 ("relu", "gelu", "silu", ...).
        layer_norm_eps: LayerNorm 분산에 더해지는 작은 epsilon 값.
        bias: 선형 레이어(FFN / 출력 head)가 bias 항을 쓰는지 여부.
        embedding_dropout: (토큰 + 위치) 임베딩에 적용하는 dropout.
        attention_dropout: 어텐션 확률 맵에 적용하는 dropout.
        residual_dropout: 각 서브레이어 출력이 residual 스트림에 더해지기
            전에 적용하는 dropout.
        norm_style: "pre"(Pre-LN, GPT 스타일 — 학습이 안정적) 또는
            "post"(Post-LN — 원 논문 "Attention Is All You Need"의 구조).
        positional_encoding_type: "sinusoidal", "learned" 또는 "none".
        learned_position_embedding: 레거시 편의 플래그. True면
            positional_encoding_type을 "learned"로 강제 override한다.
        scale_embedding: 원 논문처럼 토큰 임베딩에 sqrt(d_model)을 곱할지
            여부.
        share_embedding: 인코더와 디코더가 하나의 공유 토큰-임베딩 테이블을
            사용 (소스/타겟 어휘집이 같아야 함).
        share_decoder_embedding: 디코더 입력 임베딩을 출력 프로젝션과
            묶음(tie) ("디코더 입력-출력 임베딩 공유"라고도 함).
        tie_output_projection: 최종 어휘집 프로젝션 가중치를 디코더
            임베딩 테이블과 묶음(weight tying).
            share_decoder_embedding이 켜지면 자동으로 함의됨.
        pad_token_id: 패딩 토큰의 인덱스. 토크나이저로부터 런타임에
            덮어써진다.
    """

    vocab_size: int = 8000
    src_vocab_size: Optional[int] = None
    tgt_vocab_size: Optional[int] = None
    max_seq_length: int = 256
    d_model: int = 256
    n_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm_eps: float = 1e-5
    bias: bool = True
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    norm_style: str = "pre"
    positional_encoding_type: str = "sinusoidal"
    learned_position_embedding: bool = False
    scale_embedding: bool = True
    share_embedding: bool = True
    share_decoder_embedding: bool = True
    tie_output_projection: bool = True
    pad_token_id: int = 0


@dataclass
class AttentionConfig:
    """어텐션 메커니즘에 특화된 하이퍼파라미터.

    Attributes:
        qkv_bias: Q/K/V (및 출력) 프로젝션이 bias 항을 쓰는지 여부.
        attention_scaling: True -> 점수를 1/sqrt(d_head)로 스케일링;
            False -> 스케일링 없음; float -> 해당 값을 스케일로 사용.
        mask_future: 디코더 self-attention을 위한 causal(미래 위치 마스킹)
            마스크를 생성. 실험 목적이 아니라면 끄지 말 것.
        causal: 모든 어텐션 모듈 *내부에서* causal 마스킹을 강제
            (디코더 전용 언어 모델을 위해 이 블록들을 재사용할 때 유용).
        attention_type: 어텐션 레지스트리(models/multi_head_attention.py)의
            키. 현재는 "scaled_dot_product"; 새 메커니즘은 새 키로 등록된다.
        store_attention: 시각화를 위해 마지막 어텐션 맵을 각 모듈에
            (``module.last_attention``) 저장할지 여부. 메모리 절약을 위해
            기본값은 꺼짐.
    """

    qkv_bias: bool = True
    attention_scaling: Union[bool, float] = True
    mask_future: bool = True
    causal: bool = False
    attention_type: str = "scaled_dot_product"
    store_attention: bool = False


@dataclass
class OptimizationConfig:
    """옵티마이저, LR 스케줄, 손실 정규화 설정.

    Attributes:
        optimizer: "adam", "adamw" 또는 "sgd".
        learning_rate: 최대(peak) 학습률 (스케줄러가 이 값까지 warm-up).
        weight_decay: L2 / decoupled weight decay 계수.
        betas: Adam beta 계수 (SGD는 betas[0]을 momentum으로 사용).
        eps: Adam epsilon.
        gradient_clip: 전역 노름(global-norm) 그래디언트 클리핑 임계값
            (0이면 비활성화).
        scheduler: "noam", "cosine", "linear" 또는 "constant".
        warmup_steps: 모든 스케줄러에 적용되는 선형 warm-up 스텝 수.
        min_lr: decay하는 스케줄의 하한 학습률.
        cosine_decay: False면 "cosine" 스케줄러가 warm-up 이후 decay하지
            않고 일정하게 유지된다.
        label_smoothing: 학습 손실의 label smoothing 계수.
    """

    optimizer: str = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    gradient_clip: float = 1.0
    scheduler: str = "noam"
    warmup_steps: int = 4000
    min_lr: float = 1e-6
    cosine_decay: bool = True
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    """학습 루프 동작 방식.

    Attributes:
        batch_size: 배치당 문장 수.
        epochs: 전체 학습 epoch 수.
        seed: 완전한 재현성을 위한 랜덤 시드.
        mixed_precision: 자동 혼합 정밀도 학습 활성화 (CUDA 전용).
        accumulation_steps: 업데이트당 그래디언트 누적 micro-batch 수.
        num_workers: DataLoader 워커 프로세스 수.
        save_every: N epoch마다 주기적으로 체크포인트 저장.
        validate_every: N epoch마다 검증(validation) 실행.
        log_every: N 옵티마이저 스텝마다 TensorBoard에 학습 스칼라 로깅.
        early_stopping_patience: best metric 개선 없이 N번 검증 후 학습
            중단 (0이면 early stopping 비활성화).
        device: 사용할 디바이스 ("cuda" | "cuda:N" | "mps" | "cpu").
            null이면 자동 감지 (CUDA -> MPS -> CPU 순).
        valid_bleu: 매 검증마다 greedy 생성으로 valid BLEU를 계산해
            train.log / TensorBoard에 로깅할지 여부. 생성은 teacher-forced
            평가보다 비싸므로, 부담되면 false로 끄거나 validate_every로
            빈도를 조절한다 (fairseq --eval-bleu에 해당).
    """

    batch_size: int = 64
    epochs: int = 20
    seed: int = 42
    mixed_precision: bool = False
    accumulation_steps: int = 1
    num_workers: int = 2
    save_every: int = 1
    validate_every: int = 1
    log_every: int = 50
    early_stopping_patience: int = 5
    device: Optional[str] = None
    valid_bleu: bool = True


@dataclass
class DatasetConfig:
    """데이터 위치 및 전처리 옵션.

    Attributes:
        raw_dir: 원본 코퍼스 디렉터리 ({split}.{lang} 파일들,
            예: data/raw/train.en).
        bin_dir: 전처리 산출물 디렉터리. 언어쌍별 하위 디렉터리가
            만들어진다 (예: data/data-bin/en-de/codes.bpe).
        lang_pairs: preprocess.py가 전처리할 언어쌍 목록
            ("소스-타겟" 형식, 예: ["en-de", "en-fr"]).
        src_lang: 학습/평가/번역에 사용할 활성 쌍의 소스 언어 코드.
        tgt_lang: 활성 쌍의 타겟 언어 코드.
            (f"{src_lang}-{tgt_lang}"가 lang_pairs에 포함되어야 함.)
        valid_split: 학습 중 검증에 사용할 split 이름.
        test_split: train.py가 학습 종료 직후 best 체크포인트로 수행하는
            teacher-forced 사후 확인(loss/perplexity/accuracy)에 쓸 split
            이름 (test2016 | test2017 | testcoco | test2018). test.py는
            이 값을 쓰지 않고 --test-splits로 평가할 split을 별도 지정한다
            (기본값: test2016/test2017/testcoco/test2018 네 개 전부).
        lowercase: 토큰화 전 텍스트를 소문자로 변환할지 여부
            (preprocess.py와 translate.py가 반드시 같은 값을 써야 함).
        min_freq: 어휘집에 포함되기 위한 최소 토큰 등장 빈도 (레거시
            토이 파이프라인용; BPE 어휘집은 bpe.min_freq를 사용).
        data_dir / train_path / valid_path / test_path / tokenizer_path:
            레거시 필드 — 기존 체크포인트에 내장된 config 로드 호환용.
    """

    raw_dir: str = "data/raw"
    bin_dir: str = "data/data-bin"
    lang_pairs: list[str] = field(default_factory=lambda: ["en-de", "en-fr"])
    src_lang: str = "en"
    tgt_lang: str = "de"
    valid_split: str = "val"
    test_split: str = "test2016"
    lowercase: bool = True
    min_freq: int = 1
    data_dir: str = "data/multi30k"
    train_path: str = "data/train.tsv"
    valid_path: str = "data/valid.tsv"
    test_path: str = "data/test.tsv"
    tokenizer_path: str = "data/tokenizer.json"


@dataclass
class BPEConfig:
    """subword-nmt BPE 전처리 설정 (preprocess.py에서 사용).

    Attributes:
        num_merges: BPE merge operation 횟수 (learn_bpe의 num_symbols).
            소스+타겟 학습 코퍼스를 합쳐 공동(joint)으로 학습한다.
        vocab_size: 언어별 어휘집 크기 상한 (특수 토큰 포함).
        min_freq: 어휘집에 포함되기 위한 최소 서브워드 등장 빈도.
        codes_filename: dataset.data_dir 안에 저장될 BPE 코드 파일 이름.
    """

    num_merges: int = 10000
    vocab_size: int = 16000
    min_freq: int = 2
    codes_filename: str = "codes.bpe"


@dataclass
class CheckpointConfig:
    """체크포인트 저장 동작 방식.

    Attributes:
        save_dir: 모든 체크포인트와 TensorBoard 로그를 담을 디렉터리.
        resume_checkpoint: 학습을 이어갈 체크포인트 경로 ("" / null이면
            비활성화).
        best_metric: "최고" 모델을 정의하는 검증 지표
            ("loss", "perplexity", "token_accuracy", "accuracy" 또는 "bleu").
            "bleu"를 쓰려면 training.valid_bleu가 켜져 있어야 한다.
        save_ensemble: 학습 종료 시 최근 epoch 체크포인트들을 가중치
            평균(checkpoint averaging)하여 단일 ensemble.pt로 저장할지 여부.
        ensemble_last_n: 앙상블 평균에 사용할 최근 epoch 체크포인트 개수.
        cleanup_epoch_checkpoints: 학습 종료 시(앙상블 저장 후) 모든
            epoch_*.pt를 삭제할지 여부. last.pt / best.pt / ensemble.pt는
            이름이 달라 보존된다.
    """

    save_dir: str = "checkpoints"
    resume_checkpoint: Optional[str] = None
    best_metric: str = "loss"
    save_ensemble: bool = True
    ensemble_last_n: int = 10
    cleanup_epoch_checkpoints: bool = True


@dataclass
class InferenceConfig:
    """디코딩 하이퍼파라미터.

    Attributes:
        beam_size: 빔 서치에 사용할 빔 개수 (1이면 greedy/샘플링으로
            대체됨).
        max_length: 생성 길이의 절대 상한 (model.max_seq_length로도 상한이
            걸림). 소스 비례 길이(max_len_a/max_len_b)의 하드 캡 역할.
        max_len_a: fairseq 스타일 소스 비례 길이 계수. 배치별 최대 생성
            길이 = min(int(max_len_a·src_len + max_len_b), max_length,
            model.max_seq_length-1).
        max_len_b: 소스 비례 길이의 상수 항 (fairseq --max-len-b).
        min_length: EOS가 허용되기 전 최소 길이.
        do_sample: argmax 대신 필터링된 분포에서 샘플링 (greedy 디코딩은
            temperature/top-k/top-p를 무시함).
        temperature: 샘플링을 위한 softmax 온도.
        top_k: 가장 확률 높은 k개의 토큰만 유지 (0이면 비활성화).
        top_p: nucleus 샘플링 확률 질량 (1.0이면 비활성화).
        repetition_penalty: 1.0보다 크면 이미 생성된 토큰의 반복을 억제.
        length_penalty: 빔 서치 길이 정규화 지수 (GNMT).
    """

    beam_size: int = 4
    max_length: int = 128
    max_len_a: float = 1.2
    max_len_b: int = 10
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 0.6


@dataclass
class MultimodalConfig:
    """Multimodal Machine Translation(MMT) — 이미지 인코더 / 융합 설정.

    이 섹션이 텍스트-only Transformer를 이미지-보강 모델로 확장한다.
    ``use_image`` 하나로 전체 이미지 경로가 켜지고 꺼진다: false면 이미지
    관련 모듈(image encoder, image cross-attention, fusion)이 아예 생성되지
    않으므로, 모델은 기존 텍스트 Transformer와 완전히 동일하게 동작하고
    기존 체크포인트도 그대로 로드된다.

    이미지 인코더는 pretrained/완성 Vision Backbone(torchvision.models, timm,
    CLIP 등)을 쓰지 않고 Transformer와 동일하게 scratch로 구현한다 —
    번역 손실만으로 end-to-end 학습된다. 최종 이미지 피처는 patch feature
    전체(``(B, N, d_model)``, CLS 토큰만 쓰지 않음)를 사용한다.

    Attributes:
        use_image: 이미지 경로 마스터 스위치. false면 순수 텍스트 번역.
        image_dir: ``raw/`` 이미지들과 split별 이미지 리스트(``{split}.txt``)를
            담는 루트 디렉터리.
        image_encoder: "vit"(패치 기반 Vision Transformer) 또는
            "cnn"(ResNet 스타일). 둘 다 scratch 구현이며 교체 가능하다.
        image_size: 정사각형으로 리사이즈할 입력 이미지 한 변의 크기.
        patch_size: ViT 패치 한 변의 크기 (image_size를 나눠떨어뜨려야 함).
            패치 개수 N = (image_size / patch_size)^2.
        image_channels: 입력 이미지 채널 수 (RGB = 3).
        image_embed_dim: 이미지 인코더 내부의 residual-stream 폭. 인코더
            출력은 항상 model.d_model로 투영되므로 d_model과 달라도 된다.
        image_layers: ViT 인코더 블록 수 (CNN은 residual 스테이지 수).
        image_heads: ViT self-attention 헤드 수 (image_embed_dim을
            나눠떨어뜨려야 함).
        image_ffn_dim: ViT feed-forward 은닉 폭.
        fusion_type: 두 cross-attention 출력을 합치는 방식.
            "sum" | "weighted" | "gate".
        fusion_lambda: "weighted" 융합의 학습 가능한 λ 초기값 (0~1;
            이미지 기여 비중). 출력 = text + λ*image이며 λ는 항상 학습된다.
        freeze_image_encoder: true면 이미지 인코더 파라미터를 동결한다
            (기본은 false — 번역 손실로 함께 학습).
        use_image_cache: true면 매 스텝 JPEG를 디코딩/리사이즈하지 않고,
            preprocess_images.py가 미리 만들어둔 uint8 캐시(memmap)에서
            읽는다 (학습 데이터 로딩 병목 완화). 캐시 파일이 없거나
            image_size가 다르면 명확한 에러를 낸다.
        image_cache_dir: 캐시 .npy 파일들이 저장되는 디렉터리
            (파일명은 {split}_{image_size}.npy).
    """

    use_image: bool = False
    image_dir: str = "data/image"
    image_encoder: str = "vit"
    image_size: int = 224
    patch_size: int = 16
    image_channels: int = 3
    image_embed_dim: int = 256
    image_layers: int = 4
    image_heads: int = 8
    image_ffn_dim: int = 1024
    fusion_type: str = "gate"
    fusion_lambda: float = 0.5
    freeze_image_encoder: bool = False
    use_image_cache: bool = False
    image_cache_dir: str = "data/image/cache"


# ---------------------------------------------------------------------------
# 최상위 설정
# ---------------------------------------------------------------------------
_SECTION_TYPES: dict[str, type] = {
    "model": ModelConfig,
    "attention": AttentionConfig,
    "optimization": OptimizationConfig,
    "training": TrainingConfig,
    "dataset": DatasetConfig,
    "bpe": BPEConfig,
    "checkpoint": CheckpointConfig,
    "inference": InferenceConfig,
    "multimodal": MultimodalConfig,
}


@dataclass
class Config:
    """모든 섹션을 묶은 최상위 설정 객체.

    일반적인 사용법::

        cfg = Config.from_yaml("config/default.yaml",
                               overrides=["model.d_model=512"])
        model = Transformer(cfg)
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    bpe: BPEConfig = field(default_factory=BPEConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    multimodal: MultimodalConfig = field(default_factory=MultimodalConfig)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[list[str]] = None) -> "Config":
        """YAML로부터 설정을 로드하고 선택적인 점(dot) 표기 오버라이드를 적용한다."""
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(raw)
        if overrides:
            cfg.apply_overrides(overrides)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """일반 중첩 dict(예: 체크포인트에서 온)로부터 Config를 만든다."""
        sections: dict[str, Any] = {}
        for name, section_cls in _SECTION_TYPES.items():
            section_raw = dict(raw.get(name, {}) or {})
            known = {f.name for f in fields(section_cls)}
            unknown = set(section_raw) - known
            if unknown:  # 오타를 조용히 무시하지 않고 바로 실패시킨다
                raise ValueError(
                    f"Unknown key(s) {sorted(unknown)} in config section "
                    f"'{name}'. Known keys: {sorted(known)}"
                )
            sections[name] = section_cls(**section_raw)
        extra = set(raw) - set(_SECTION_TYPES)
        if extra:
            raise ValueError(f"Unknown config section(s): {sorted(extra)}")
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        """일반 중첩 dict로 직렬화한다 (체크포인트에 담기 좋은 형태)."""
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """재현성을 위해 완전히 확정된(resolved) 설정을 파일로 저장한다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    # ------------------------------------------------------------ overrides
    def apply_overrides(self, overrides: list[str]) -> None:
        """``section.key=value`` 형태의 CLI 오버라이드를 적용한다.

        값은 YAML 문법으로 파싱되므로, ``true``, ``0.1``,
        ``[0.9, 0.98]``, ``null`` 모두 기대하는 파이썬 타입으로 변환된다.
        """
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override '{item}' must look like section.key=value")
            dotted, raw_value = item.split("=", 1)
            parts = dotted.strip().split(".")
            if len(parts) != 2:
                raise ValueError(f"Override key '{dotted}' must be 'section.key'")
            section_name, key = parts
            section = getattr(self, section_name, None)
            if section is None or section_name not in _SECTION_TYPES:
                raise ValueError(f"Unknown config section '{section_name}'")
            if not hasattr(section, key):
                raise ValueError(f"Unknown key '{key}' in section '{section_name}'")
            setattr(section, key, yaml.safe_load(raw_value))

    # ----------------------------------------------------------- validation
    def validate(self) -> None:
        """서로 다른 필드 간 제약 조건을 검사한다; 문제가 있으면 ValueError."""
        m, a, o, c = self.model, self.attention, self.optimization, self.checkpoint
        if m.d_model % m.n_heads != 0:
            raise ValueError(f"d_model ({m.d_model}) must be divisible by n_heads ({m.n_heads})")
        if m.norm_style not in ("pre", "post"):
            raise ValueError(f"norm_style must be 'pre' or 'post', got '{m.norm_style}'")
        if m.positional_encoding_type not in ("sinusoidal", "learned", "none"):
            raise ValueError(f"Invalid positional_encoding_type '{m.positional_encoding_type}'")
        for name in ("dropout", "embedding_dropout", "attention_dropout", "residual_dropout"):
            value = getattr(m, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"model.{name} must be in [0, 1), got {value}")
        if o.scheduler not in ("noam", "cosine", "linear", "constant"):
            raise ValueError(f"Invalid scheduler '{o.scheduler}'")
        if o.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(f"Invalid optimizer '{o.optimizer}'")
        if c.best_metric not in ("loss", "perplexity", "token_accuracy", "accuracy", "bleu"):
            raise ValueError(f"Invalid best_metric '{c.best_metric}'")
        if self.training.accumulation_steps < 1:
            raise ValueError("training.accumulation_steps must be >= 1")
        if not isinstance(a.attention_scaling, (bool, int, float)):
            raise ValueError("attention.attention_scaling must be bool or float")
        # 언어별 분리 어휘집: 크기가 다르면 임베딩 공유가 불가능하다.
        if (
            m.share_embedding
            and m.src_vocab_size is not None
            and m.tgt_vocab_size is not None
            and m.src_vocab_size != m.tgt_vocab_size
        ):
            raise ValueError(
                "share_embedding=true requires equal src/tgt vocab sizes "
                f"(got {m.src_vocab_size} vs {m.tgt_vocab_size})"
            )
        # 언어쌍 형식 및 활성 쌍 검사.
        d = self.dataset
        for pair in d.lang_pairs:
            parts = pair.split("-")
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"dataset.lang_pairs entry '{pair}' must look like 'en-de'")
        active = f"{d.src_lang}-{d.tgt_lang}"
        if d.lang_pairs and active not in d.lang_pairs:
            raise ValueError(
                f"Active pair '{active}' (dataset.src_lang-tgt_lang) is not in "
                f"dataset.lang_pairs {d.lang_pairs} — add it or fix src/tgt_lang."
            )
        # 멀티모달(MMT) 제약 — use_image일 때만 검사한다 (텍스트-only는 영향 없음).
        mm = self.multimodal
        if mm.use_image:
            if mm.image_encoder not in ("vit", "cnn"):
                raise ValueError(
                    f"multimodal.image_encoder must be 'vit' or 'cnn', got '{mm.image_encoder}'"
                )
            if mm.fusion_type not in ("sum", "weighted", "gate"):
                raise ValueError(
                    f"multimodal.fusion_type must be 'sum', 'weighted' or 'gate', "
                    f"got '{mm.fusion_type}'"
                )
            if mm.image_encoder == "vit":
                if mm.image_size % mm.patch_size != 0:
                    raise ValueError(
                        f"multimodal.image_size ({mm.image_size}) must be divisible by "
                        f"patch_size ({mm.patch_size})"
                    )
                if mm.image_embed_dim % mm.image_heads != 0:
                    raise ValueError(
                        f"multimodal.image_embed_dim ({mm.image_embed_dim}) must be "
                        f"divisible by image_heads ({mm.image_heads})"
                    )
            if not 0.0 <= mm.fusion_lambda <= 1.0:
                raise ValueError(
                    f"multimodal.fusion_lambda must be in [0, 1], got {mm.fusion_lambda}"
                )
