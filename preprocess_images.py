"""
===============================================================================
 파일: preprocess_images.py
 목적:
    Multimodal MT용 이미지를 미리 리사이즈해 split별 uint8 텐서 파일로
    패킹하는 전처리 진입점. 학습 때 매 스텝 JPEG를 디코딩/리사이즈하는
    비용(데이터 로딩 병목)을 제거한다.

    data/image/raw/*.jpg  +  data/image/{split}.txt
        ↓  (read_image RGB -> Resize(image_size), uint8 유지)
    data/image/cache/{split}_{image_size}.npy   # shape (N, 3, S, S) uint8

 역할:
    `python preprocess_images.py` 한 번으로 각 split의 이미지 리스트를
    순서대로 읽어 하나의 memmap .npy로 저장한다. 정규화(float 변환 + mean/std)는
    저장하지 않는다 — 로드 시점에 dataset.py가 적용하므로 캐시 용량을
    1/4로 줄이면서도 즉석 로드 경로와 바이트 단위로 동일한 결과를 낸다
    (Resize는 uint8에 대해 결정적).

 입력 / 출력:
    입력 : config/default.yaml (multimodal.image_dir / image_size /
           image_channels / image_cache_dir) + data/image/{split}.txt + 원본 jpg.
    출력 : {image_cache_dir}/{split}_{image_size}.npy  + 콘솔 통계.

 구현 세부사항:
    - open_memmap으로 디스크에 (N, C, S, S) uint8 배열을 만들고 한 장씩
      채워 넣는다 — 전체 이미지를 메모리에 올리지 않는 스트리밍 방식이라
      데이터셋이 커도 안전하다.
    - 이미지는 리스트({split}.txt) 순서대로 저장되므로, 캐시의 row 인덱스가
      곧 코퍼스의 원본 줄 인덱스다 (dataset.py가 이 인덱스로 조회한다).
    - 파일명에 image_size가 들어가므로 해상도를 바꾸면 다른 캐시가 생성되고,
      dataset.py는 현재 image_size에 맞는 캐시만 사용한다.
    - 이미 존재하는 캐시는 재사용하며 --force 로 강제 재생성한다.
===============================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm

from config.config import Config
from utils.image import build_resize_transform, resolve_image_path
from utils.logger import get_logger

logger = get_logger("preprocess_images")

# 이미지 리스트가 존재할 수 있는 표준 Multi30k split (실제로 {split}.txt가
# 있는 것만 처리된다).
DEFAULT_SPLITS: tuple[str, ...] = (
    "train",
    "val",
    "test2016",
    "test2017",
    "test2018",
    "testcoco",
)


def parse_args() -> argparse.Namespace:
    """CLI: 설정, 점(dot) 오버라이드, 처리할 split, 강제 재생성."""
    parser = argparse.ArgumentParser(description="Pre-resize MMT images into packed uint8 caches")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="config overrides (repeatable), e.g. --set multimodal.image_size=224",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(DEFAULT_SPLITS), metavar="SPLIT",
        help=f"splits to cache (default: {' '.join(DEFAULT_SPLITS)})",
    )
    parser.add_argument("--force", action="store_true", help="regenerate even if the cache exists")
    return parser.parse_args()


def cache_path(cache_dir: str | Path, split: str, image_size: int) -> Path:
    """split별 캐시 파일 경로 규칙 (image_size를 파일명에 포함)."""
    return Path(cache_dir) / f"{split}_{image_size}.npy"


def build_split_cache(
    split: str,
    image_dir: str,
    cache_dir: str,
    image_size: int,
    channels: int,
    force: bool,
) -> bool:
    """한 split의 이미지를 (N, C, S, S) uint8 memmap으로 패킹한다.

    Args:
        split: split 이름 (예: "train").
        image_dir: 이미지 루트 (raw/ 와 {split}.txt 를 담음).
        cache_dir: 캐시 .npy 저장 디렉터리.
        image_size: 리사이즈 한 변 크기.
        channels: 채널 수 (RGB=3).
        force: True면 기존 캐시가 있어도 다시 만든다.

    Returns:
        캐시를 생성했으면 True, split이 없거나 건너뛰었으면 False.
    """
    list_path = Path(image_dir) / f"{split}.txt"
    if not list_path.exists():
        logger.info("[%s] no image list (%s) — skipping", split, list_path)
        return False

    with open(list_path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    num_images = len(lines)

    out_path = cache_path(cache_dir, split, image_size)
    if out_path.exists() and not force:
        logger.info("[%s] cache exists (%s) — skipping (use --force to rebuild)", split, out_path)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    resize = build_resize_transform(image_size)

    # 디스크에 memmap을 만들고 한 장씩 채운다 (전체를 메모리에 올리지 않음).
    array = open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint8,
        shape=(num_images, channels, image_size, image_size),
    )
    for row, line in enumerate(tqdm(lines, desc=f"cache[{split}]")):
        path = resolve_image_path(image_dir, line)
        # read_image: (C, H, W) uint8, RGB 강제 -> Resize (dtype 유지) -> uint8.
        resized = resize(read_image(str(path), mode=ImageReadMode.RGB))
        array[row] = resized.numpy()
    array.flush()
    del array  # memmap을 닫아 버퍼를 확실히 디스크로 내린다.

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("[%s] wrote %d images -> %s (%.1f MB)", split, num_images, out_path, size_mb)
    return True


def main() -> None:
    """설정된 image_size로 요청된 모든 split의 이미지 캐시를 생성한다."""
    args = parse_args()
    config = Config.from_yaml(args.config, overrides=args.set)
    mm = config.multimodal

    logger.info(
        "Caching images: size=%d, channels=%d, dir=%s -> %s",
        mm.image_size, mm.image_channels, mm.image_dir, mm.image_cache_dir,
    )

    built = 0
    for split in args.splits:
        if build_split_cache(
            split,
            image_dir=mm.image_dir,
            cache_dir=mm.image_cache_dir,
            image_size=mm.image_size,
            channels=mm.image_channels,
            force=args.force,
        ):
            built += 1

    logger.info(
        "Done — %d/%d split(s) cached. Enable with `--set multimodal.use_image_cache=true`.",
        built, len(args.splits),
    )


if __name__ == "__main__":
    main()
