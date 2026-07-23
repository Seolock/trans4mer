"""
===============================================================================
 파일: translate.py
 목적:
    단일 문장 번역 CLI — 학습된 체크포인트로 임의의 문장을 즉시 번역한다.

 역할:
    사용 예:
        python translate.py \\
            --checkpoint checkpoints/en-de/ensemble.pt \\
            --sentence "A man is riding a bicycle."

    출력:
        German Translation : ein mann fährt fahrrad .

 입력 / 출력:
    입력 : --sentence (소스 언어 원본 문장), --checkpoint.
    출력 : 표준 출력으로 번역 문장 한 줄.

 구현 세부사항:
    - 실제 작업(전처리 -> BPE -> 디코딩 -> BPE 제거)은 전부
      inference/translator.py 의 Translator가 수행한다; 이 파일은 얇은
      CLI 래퍼다.
    - 디코딩 전략은 체크포인트에 저장된 inference 설정을 따르며
      --set 으로 오버라이드할 수 있다:
        --set inference.beam_size=1     # greedy
        --set inference.beam_size=8     # 더 넓은 빔
===============================================================================
"""

from __future__ import annotations

import argparse
import sys

from utils.logger import get_logger
from utils.misc import get_device

logger = get_logger("translate")

# 언어 코드 -> 출력 라벨용 언어 이름 (미등록 코드는 코드 그대로 표시).
LANGUAGE_NAMES: dict[str, str] = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "cs": "Czech",
}


def parse_args() -> argparse.Namespace:
    """CLI: 체크포인트, 번역할 문장, 선택적 디코딩 오버라이드."""
    parser = argparse.ArgumentParser(description="Translate a sentence with a trained model")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint path (default: {save_dir}/{src}-{tgt}/ensemble.pt from the config)",
    )
    parser.add_argument("--sentence", required=True, help="source sentence to translate")
    parser.add_argument(
        "--image", default=None,
        help="image path for Multimodal (MMT) checkpoints; matched to the sentence",
    )
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="config overrides (repeatable), e.g. --set inference.beam_size=1",
    )
    parser.add_argument("--device", default=None, help="force a device (cuda|mps|cpu)")
    return parser.parse_args()


def main() -> None:
    """문장 하나를 번역해 'German Translation : ...' 형식으로 출력한다."""
    args = parse_args()

    from config.config import Config
    from inference.translator import Translator  # torch 임포트 비용은 여기서만
    from utils.data_paths import default_checkpoint_path

    # 체크포인트를 지정하지 않으면 설정의 활성 쌍(src_lang-tgt_lang)으로
    # 기본 경로(ensemble.pt)를 해석한다. --set 오버라이드도 경로 결정에
    # 반영된다 (예: --set dataset.tgt_lang=fr -> checkpoints/en-fr/ensemble.pt).
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        cli_config = Config.from_yaml(args.config, overrides=args.set)
        checkpoint_path = str(
            default_checkpoint_path(
                cli_config.checkpoint.save_dir,
                cli_config.dataset.src_lang,
                cli_config.dataset.tgt_lang,
            )
        )
        logger.info("Checkpoint: %s", checkpoint_path)

    try:
        translator = Translator.from_checkpoint(
            checkpoint_path, device=get_device(args.device)
        )
    except FileNotFoundError as error:
        logger.error("%s", error)
        sys.exit(1)

    if args.set:
        translator.config.apply_overrides(args.set)

    # Multimodal 체크포인트인데 이미지를 주지 않으면 이미지 없이(텍스트 경로로)
    # 번역된다 — 사용자가 알 수 있도록 경고한다.
    if translator.use_image and args.image is None:
        logger.warning("Checkpoint is multimodal but no --image given; translating text-only.")

    translation = translator.translate(args.sentence, image=args.image)

    tgt_lang = translator.config.dataset.tgt_lang
    language = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    print(f"{language} Translation : {translation}")


if __name__ == "__main__":
    main()
