"""
===============================================================================
 파일: test.py
 목적:
    학습된 체크포인트를 Multi30k의 표준 테스트 세트 전체에 대해 평가한다:
    test2016, test2017, testcoco, test2018 네 가지 split을 한 번의 실행으로
    모두 처리한다.

    split마다:
      1. teacher-forced 지표 (loss / perplexity / 토큰·시퀀스 정확도)
      2. 테스트 세트 전체를 실제로 번역 (빔 서치 또는 greedy)
      3. BPE 제거 후 BLEU · METEOR · COMET 점수 계산
      4. 번역 결과물을 hypo_{split}.txt로 저장

 역할:
    `python train.py` 이후 `python test.py` 한 번으로 네 개 테스트 세트
    모두의 번역 품질을 확인한다:
        python test.py --set inference.beam_size=1        # greedy 디코딩
        python test.py --test-splits test2016 test2017    # 일부만 평가

 입력 / 출력:
    입력 : YAML 설정 (데이터 경로 / 디코딩 옵션) + 체크포인트.
    출력 : 지표와 BLEU를 콘솔에 출력하고, 체크포인트와 같은 디렉터리에
        - test.log                    (전체 실행 개요: 시작/종료, split별 요약)
        - test_{split}.log            (split마다 하나씩 — 상세 실행 로그)
        - hypo_{split}.txt            (split별 번역 결과물, 한 줄에 한 문장)
        도 함께 저장한다.

 구현 세부사항:
    - 모델과 아키텍처 설정은 체크포인트에서, 데이터 경로와 디코딩 옵션은
      CLI 설정에서 온다 (Translator.from_checkpoint 참고).
    - split마다 별도의 로거(`test.{split}`)를 만들어 test_{split}.log에
      기록하므로, 한 split의 로그만 보고 싶을 때 그 파일만 열면 된다.
      전체 개요는 최상위 test.log에 별도로 남는다.
    - 어떤 split을 평가할지는 --test-splits로 바꿀 수 있으며, 기본값은
      DEFAULT_TEST_SPLITS(test2016/test2017/testcoco/test2018) 네 개다.
    - 한 split의 원본 파일을 찾지 못하면(FileNotFoundError) 그 split만
      건너뛰고 나머지는 계속 진행한다 — 전체 평가가 한 파일 때문에
      중단되지 않는다.
    - BLEU는 표준 sacreBLEU corpus_bleu를 사용한다 (fairseq --scoring
      sacrebleu 방식). 참조는 원문(detok) 그대로 넘기고 sacreBLEU의 기본
      13a 토크나이저가 hyp/ref를 동일하게 토큰화한다. 학습 코퍼스가
      소문자화되어 모델 출력도 소문자이므로 lowercase=True(uncased) BLEU다.
    - 번역은 --batch-size 단위로 배치 처리되며 tqdm으로 진행률을
      표시한다.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from config.config import Config
from dataset import build_split_dataloader
from inference.translator import Translator
from preprocess import find_raw_file
from trainer.evaluator import evaluate
from utils.data_paths import default_checkpoint_path
from utils.image import resolve_image_path
from utils.logger import get_logger
from utils.misc import get_device
from utils.mt_metrics import compute_bleu, compute_comet, compute_meteor
from utils.seed import set_seed

# Multi30k 표준 테스트 세트 네 가지 (--test-splits로 이 목록을 좁힐 수 있다).
DEFAULT_TEST_SPLITS: tuple[str, ...] = ("test2016", "test2017", "testcoco", "test2018")


def parse_args() -> argparse.Namespace:
    """CLI: 설정, 체크포인트, device, 평가할 split 목록, 배치 크기."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer on Multi30k")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint path (default: {save_dir}/{src}-{tgt}/ensemble.pt from the config)",
    )
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="config overrides (repeatable), e.g. --set inference.beam_size=8",
    )
    parser.add_argument(
        "--test-splits", nargs="+", default=list(DEFAULT_TEST_SPLITS), metavar="SPLIT",
        help=f"test splits to evaluate (default: {' '.join(DEFAULT_TEST_SPLITS)})",
    )
    parser.add_argument("--device", default=None, help="force a device (cuda|mps|cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="translation batch size")
    parser.add_argument(
        "--comet-model", default="Unbabel/wmt22-comet-da",
        help="COMET model name (default: 논문 표준 wmt22-comet-da)",
    )
    parser.add_argument(
        "--no-comet", action="store_true",
        help="skip COMET (빠른 반복용; 기본은 BLEU+METEOR+COMET 모두 계산)",
    )
    parser.add_argument("--comet-batch-size", type=int, default=8, help="COMET inference batch size")
    return parser.parse_args()


def format_metrics(metrics: dict[str, float | None]) -> str:
    """지표 dict를 'BLEU=.. | METEOR=.. | COMET=..' 문자열로 만든다 (None은 n/a)."""
    labels = (("bleu", "BLEU"), ("meteor", "METEOR"), ("comet", "COMET"))
    parts = []
    for key, label in labels:
        value = metrics.get(key)
        parts.append(f"{label}={value:.2f}" if value is not None else f"{label}=n/a")
    return " | ".join(parts)


def evaluate_split(
    split: str,
    translator: Translator,
    config: Config,
    checkpoint_dir: Path,
    device: torch.device,
    batch_size: int,
    comet_model: str = "Unbabel/wmt22-comet-da",
    run_comet: bool = True,
    comet_batch_size: int = 8,
) -> dict[str, float | None]:
    """한 테스트 split에 대해 teacher-forced 평가 + 번역 + BLEU를 수행한다.

    이 split 전용 로거(`test.{split}`)를 만들어 모든 진행 상황을
    ``checkpoint_dir/test_{split}.log``에 기록하고, 번역 결과물을
    ``checkpoint_dir/hypo_{split}.txt``에 (한 줄에 한 문장) 저장한다.

    Args:
        split: 평가할 split 이름 (예: "test2016").
        translator: 체크포인트로부터 이미 로드된 Translator.
        config: translator.config (모델은 체크포인트, 데이터/디코딩은 CLI).
        checkpoint_dir: 체크포인트가 있는 디렉터리 (로그/결과 저장 위치).
        device: 평가/번역에 사용할 디바이스.
        batch_size: 번역 시 배치 크기.
        comet_model: COMET 모델 이름 (기본: 논문 표준 wmt22-comet-da).
        run_comet: False면 COMET 계산을 건너뛴다 (빠른 반복용).
        comet_batch_size: COMET 추론 배치 크기.

    Returns:
        ``{"bleu", "meteor", "comet"}`` 지표 dict. METEOR/COMET은 의존성/
        다운로드가 필요하므로 사용 불가 시 해당 값이 ``None``이다.

    Raises:
        FileNotFoundError: 이 split의 원본 코퍼스 또는 id 파일이 없을 때
            (호출자가 잡아서 해당 split만 건너뛸 수 있다).
    """
    split_logger = get_logger(f"test.{split}", log_file=checkpoint_dir / f"test_{split}.log")
    d = config.dataset
    split_logger.info("=== Evaluating split '%s' (%s-%s) ===", split, d.src_lang, d.tgt_lang)

    # ------------------------------------------------ 1) teacher-forced 지표
    loader = build_split_dataloader(
        config, split,
        pad_id=translator.src_vocab.pad_id,
        bos_id=translator.tgt_vocab.bos_id,
        eos_id=translator.tgt_vocab.eos_id,
    )
    metrics = evaluate(
        translator.model, loader, device, pad_id=config.model.pad_token_id, progress=True
    )
    split_logger.info(
        "teacher-forced — %s",
        " | ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
    )

    # ------------------------------------------------ 2) 테스트 세트 번역
    src_file = find_raw_file(d.raw_dir, split, d.src_lang)
    ref_file = find_raw_file(d.raw_dir, split, d.tgt_lang)
    with open(src_file, "r", encoding="utf-8") as fh:
        sources = [line.strip() for line in fh]
    with open(ref_file, "r", encoding="utf-8") as fh:
        # 참조는 원문(detok) 그대로 사용한다 — sacreBLEU 내부 13a 토크나이저가
        # hyp/ref 양쪽을 동일하게 토큰화한다 (fairseq --scoring sacrebleu 방식).
        references = [line.strip() for line in fh]

    # Multimodal: split별 이미지 리스트를 코퍼스와 인덱스 정렬해 로드한다.
    image_paths: list[str] | None = None
    if config.multimodal.use_image:
        image_list = Path(config.multimodal.image_dir) / f"{split}.txt"
        with open(image_list, "r", encoding="utf-8") as fh:
            image_paths = [str(resolve_image_path(config.multimodal.image_dir, line)) for line in fh]
        if len(image_paths) != len(sources):
            raise ValueError(
                f"Image/corpus mismatch for split '{split}': {len(image_paths)} images "
                f"but {len(sources)} sentences."
            )

    strategy = f"beam={config.inference.beam_size}" if config.inference.beam_size > 1 else "greedy"
    split_logger.info("Translating %d sentences (%s) ...", len(sources), strategy)
    hypotheses: list[str] = []
    for start in tqdm(range(0, len(sources), batch_size), desc=f"translate[{split}]"):
        batch = sources[start : start + batch_size]
        batch_images = image_paths[start : start + batch_size] if image_paths is not None else None
        hypotheses.extend(translator.translate_batch(batch, images=batch_images))

    # ------------------------------------------------ 3) 번역 결과물 저장
    # 한 줄에 한 문장(BPE 제거된 일반 텍스트); 소스와 인덱스 정렬.
    hypo_path = checkpoint_dir / f"hypo_{split}.txt"
    with open(hypo_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(hypotheses) + "\n")
    split_logger.info("Saved %d translations -> %s", len(hypotheses), hypo_path)

    # ------------------------------------------------ 4) 자동 평가 지표
    # BLEU(sacreBLEU 표준 13a), METEOR(NLTK), COMET(unbabel-comet)을 함께
    # 계산·기록한다. METEOR/COMET은 graceful — 사용 불가 시 None으로 건너뛴다.
    bleu_score, bleu_sig = compute_bleu(hypotheses, references)
    split_logger.info("BLEU   = %.2f  (%s)", bleu_score, bleu_sig)

    meteor_score = compute_meteor(hypotheses, references)
    if meteor_score is not None:
        split_logger.info("METEOR = %.2f", meteor_score)

    comet_score = None
    if run_comet:
        split_logger.info("Computing COMET (%s) — 최초 실행 시 모델 다운로드 ...", comet_model)
        comet_score = compute_comet(
            sources, hypotheses, references,
            model_name=comet_model, batch_size=comet_batch_size,
        )
        if comet_score is not None:
            split_logger.info("COMET  = %.2f  (%s)", comet_score, comet_model)

    return {"bleu": bleu_score, "meteor": meteor_score, "comet": comet_score}


def main() -> None:
    """설정된 모든 테스트 split을 순회하며 평가하고 요약을 test.log에 남긴다."""
    args = parse_args()
    cli_config = Config.from_yaml(args.config, overrides=args.set)
    set_seed(cli_config.training.seed)

    # 체크포인트 경로를 먼저 정해야 그 옆에 로그/결과를 쓸 디렉터리를 안다
    # (train.py가 train.log를 checkpoint.save_dir에 남기는 것과 동일한 규칙).
    # --checkpoint 미지정 시 기본값은 ensemble.pt (default_checkpoint_path 참고).
    checkpoint_path = args.checkpoint or str(
        default_checkpoint_path(
            cli_config.checkpoint.save_dir,
            cli_config.dataset.src_lang,
            cli_config.dataset.tgt_lang,
        )
    )
    checkpoint_dir = Path(checkpoint_path).parent
    # 전체 실행 개요 로그 (split별 상세 로그는 evaluate_split이 별도로 남김).
    logger = get_logger("test", log_file=checkpoint_dir / "test.log")

    device = get_device(args.device or cli_config.training.device)
    logger.info("Using device: %s", device)
    logger.info("Checkpoint: %s", checkpoint_path)

    try:
        translator = Translator.from_checkpoint(
            checkpoint_path, config_override=cli_config, device=device
        )
    except FileNotFoundError as error:
        logger.error("%s", error)
        sys.exit(1)
    config = translator.config

    # sacrebleu는 모든 split을 다 도는 도중에 없는 것을 발견하면 손해가
    # 크므로, 루프 시작 전에 미리 확인한다.
    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        logger.error("sacrebleu is not installed — run `pip install -r requirements.txt`")
        sys.exit(1)

    logger.info(
        "Evaluating %d split(s) for %s-%s: %s",
        len(args.test_splits), config.dataset.src_lang, config.dataset.tgt_lang,
        ", ".join(args.test_splits),
    )

    # ------------------------------------------------ split별 평가 루프
    all_results: dict[str, dict[str, float | None]] = {}
    for split in args.test_splits:
        try:
            all_results[split] = evaluate_split(
                split, translator, config, checkpoint_dir, device, args.batch_size,
                comet_model=args.comet_model,
                run_comet=not args.no_comet,
                comet_batch_size=args.comet_batch_size,
            )
        except FileNotFoundError as error:
            # 이 split의 원본/id 파일이 없어도 나머지 split 평가는 계속한다.
            logger.error("[%s] skipped — %s", split, error)
            continue
        logger.info("[%s] %s", split, format_metrics(all_results[split]))

    if not all_results:
        logger.error("No split could be evaluated — check --test-splits and data-bin artifacts.")
        sys.exit(1)

    # ------------------------------------------------ 요약 (test.log에 기록)
    for split, metrics in all_results.items():
        logger.info("summary [%s] — %s", split, format_metrics(metrics))


if __name__ == "__main__":
    main()
