"""
===============================================================================
 파일: train.py
 목적:
    Multi30k 번역 학습 진입점: 어휘집/데이터 로딩 -> 모델 생성 ->
    Trainer(teacher forcing, AMP, 스케줄러, early stopping, 체크포인트)
    실행 -> 최고 체크포인트의 최종 테스트 평가.

 역할:
    `python preprocess.py` 이후 `python train.py` 한 번으로 학습 전체가
    동작한다. 기존 trainer/ 패키지(Trainer, evaluate, 스케줄러,
    체크포인트 매니저)를 그대로 재사용하며, 데이터 소스만 BPE 파이프라인
    (dataset.py + vocab.py)으로 교체되었다.

 입력 / 출력:
    입력 : YAML 설정 (+ --set 오버라이드, --resume 체크포인트).
    출력 : checkpoint.save_dir 아래에 last/best/epoch 체크포인트,
           TensorBoard 로그, train.log, 확정된 config.yaml.

 구현 세부사항:
    - 어휘집이 현실을 정의한다: 모델을 만들기 *전에* vocab.en / vocab.de
      크기가 model.src_vocab_size / tgt_vocab_size로 설정되므로, 설정
      오타가 임베딩 크기를 어긋나게 만들 수 없다.
    - 언어별 분리 어휘집이므로 share_embedding이 켜져 있으면 경고 후
      자동으로 끈다 (내용이 다른 두 어휘집은 공유가 무의미하다).
    - CrossEntropyLoss(ignore_index=pad, label smoothing)와 teacher
      forcing(입력 [BOS..yn] -> 타겟 [y1..EOS])은 Trainer/collate에 이미
      구현되어 있다.
    - best 체크포인트는 validation loss 기준으로 저장된다
      (checkpoint.best_metric으로 변경 가능).
    - 학습 중 valid BLEU는 매 검증마다 greedy 생성으로 계산되어 train.log와
      TensorBoard에 기록된다 (training.valid_bleu). 최종 test BLEU는 여전히
      test.py가 빔서치로 계산한다.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.config import Config
from dataset import build_dataloaders
from models.transformer import Transformer
from trainer.checkpoint import CheckpointManager
from trainer.evaluator import evaluate
from trainer.trainer import Trainer
from utils.data_paths import pair_dir, pair_name, vocab_path
from utils.logger import get_logger
from utils.misc import count_parameters, format_count, get_device
from utils.seed import set_seed
from vocab import Vocab


def parse_args() -> argparse.Namespace:
    """CLI: 설정 경로, 점(dot) 표기 오버라이드, 선택적 device/resume."""
    parser = argparse.ArgumentParser(description="Train a Transformer on Multi30k")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="config overrides (repeatable), e.g. --set model.d_model=512",
    )
    parser.add_argument("--device", default=None, help="force a device (cuda|mps|cpu)")
    parser.add_argument(
        "--resume", default=None,
        help="checkpoint to resume from (overrides checkpoint.resume_checkpoint)",
    )
    return parser.parse_args()


def main() -> None:
    """전체 학습 파이프라인; 각 단계는 파일 상단 설명을 참고하라."""
    args = parse_args()
    config = Config.from_yaml(args.config, overrides=args.set)
    if args.resume:
        config.checkpoint.resume_checkpoint = args.resume

    # 쌍별 체크포인트 디렉터리: en-de와 en-fr 학습이 서로 덮어쓰지 않도록
    # save_dir 아래에 활성 쌍 이름의 하위 디렉터리를 자동으로 사용한다
    # (사용자가 이미 쌍 이름으로 끝나는 경로를 준 경우는 중복 추가 안 함).
    d = config.dataset
    active_pair = pair_name(d.src_lang, d.tgt_lang)
    save_dir = Path(config.checkpoint.save_dir)
    if save_dir.name != active_pair:
        config.checkpoint.save_dir = str(save_dir / active_pair)

    logger = get_logger("train", log_file=Path(config.checkpoint.save_dir) / "train.log")
    set_seed(config.training.seed)
    device = get_device(args.device or config.training.device)
    logger.info("Using device: %s | pair: %s | checkpoints: %s",
                device, active_pair, config.checkpoint.save_dir)

    # ------------------------------------------------------------ 어휘집
    base_dir = pair_dir(d.bin_dir, d.src_lang, d.tgt_lang)
    try:
        src_vocab = Vocab.load(vocab_path(base_dir, d.src_lang))
        tgt_vocab = Vocab.load(vocab_path(base_dir, d.tgt_lang))
    except FileNotFoundError as error:
        logger.error("%s", error)
        sys.exit(1)
    logger.info(
        "Vocab loaded — %s: %d tokens | %s: %d tokens",
        d.src_lang, len(src_vocab), d.tgt_lang, len(tgt_vocab),
    )

    # ------------------------------------------------------------ 데이터
    try:
        loaders = build_dataloaders(
            config,
            pad_id=src_vocab.pad_id,
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
        )
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        sys.exit(1)

    # ------------------------------------------------------------ 모델 설정
    # 어휘집이 어휘 관련 사실의 근원(source of truth)이다.
    config.model.src_vocab_size = len(src_vocab)
    config.model.tgt_vocab_size = len(tgt_vocab)
    config.model.vocab_size = max(len(src_vocab), len(tgt_vocab))
    config.model.pad_token_id = src_vocab.pad_id  # 양쪽 모두 0
    if config.model.share_embedding:
        # 언어별 분리 어휘집: 내용이 다르므로 임베딩 공유가 무의미하다.
        logger.warning("Separate per-language vocabularies — disabling share_embedding")
        config.model.share_embedding = False
    config.validate()

    model = Transformer(config).to(device)
    logger.info("Model parameters: %s", format_count(count_parameters(model)))

    # 확정된 설정을 체크포인트 옆에 정확히 저장해둔다.
    config.save_yaml(Path(config.checkpoint.save_dir) / "config.yaml")

    # ------------------------------------------------------------ 학습
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        device=device,
        tgt_vocab=tgt_vocab,  # 검증 중 생성 기반 valid BLEU 로깅에 사용
    )
    trainer.fit()

    # ------------------------------------------------- 최종 테스트 평가
    # (teacher-forced 지표; 생성 기반 BLEU 평가는 test.py에서 수행)
    best_path = trainer.checkpoints.best_path
    if best_path.exists():
        logger.info("Loading best checkpoint for final test evaluation: %s", best_path)
        state = CheckpointManager.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    else:
        logger.warning("No best checkpoint found — evaluating the final weights")

    test_metrics = evaluate(
        model, loaders["test"], device, pad_id=config.model.pad_token_id, progress=True
    )
    logger.info(
        "TEST (teacher-forced) — %s",
        " | ".join(f"{name}={value:.4f}" for name, value in test_metrics.items()),
    )
    logger.info("Run `python test.py` for beam-search translation + sacreBLEU.")


if __name__ == "__main__":
    main()
