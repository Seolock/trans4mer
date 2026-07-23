"""
===============================================================================
 파일: utils/image.py
 목적:
    Multimodal MT를 위한 이미지 로딩/전처리 헬퍼. 디스크의 JPEG를 읽어
    모델이 바로 먹을 수 있는 정규화된 텐서로 변환한다. 또한 미리 리사이즈해
    저장한 uint8 캐시(preprocess_images.py 산출물)를 로드-시점에 정규화하는
    경로도 제공한다.

 역할:
    Dataset(dataset.py의 TranslationDataset), 추론 래퍼
    (inference/translator.py의 Translator), 이미지 캐시 스크립트
    (preprocess_images.py)가 동일한 전처리 규칙을 공유하도록 한 곳에서
    관리한다 — 그래야 학습/추론/캐시 결과가 절대 어긋나지 않는다.

 전처리 파이프라인 (두 경로가 완전히 동일한 결과를 낸다):
    (A) JPEG 즉석 로드:  read_image(uint8) -> Resize(uint8) -> float -> Normalize
    (B) 캐시 사용:        preprocess_images.py가 (A)의 Resize까지 미리 수행해
                          uint8로 저장 -> 로드 시 float -> Normalize 만 적용
    Resize는 uint8에 대해 결정적(antialias bilinear)이므로 (A)와 (B)의 최종
    텐서는 바이트 단위로 동일하다.

 입력 / 출력:
    resolve_image_path(image_dir, list_line) -> Path   (raw/ 아래 실제 파일)
    build_image_transform(image_size)   -> (C,H,W)uint8 -> (C,S,S)정규화 float
    build_resize_transform(image_size)  -> (C,H,W)uint8 -> (C,S,S)uint8 (캐시 생성용)
    build_normalize_transform()         -> (C,S,S)uint8 -> (C,S,S)정규화 float (캐시 로드용)
    load_image_tensor(path, ...)        -> (C,S,S) 정규화 float 텐서

 구현 세부사항:
    - torchvision.io.read_image + torchvision.transforms만 사용한다
      (pretrained/완성 Vision Backbone인 torchvision.models는 사용하지
      않는다 — 이미지 인코더는 scratch로 직접 구현).
    - 정규화는 채널별 mean=std=0.5로 [-1, 1] 범위에 맞춘다. 사전학습
      backbone을 쓰지 않으므로 ImageNet 통계 대신 단순 정규화를 쓴다.
    - split별 이미지 리스트({split}.txt)의 한 줄은 보통 "<id>.jpg" 이지만
      testcoco.txt는 "COCO_..._000000117071.jpg#367178" 처럼 '#'이 붙는다;
      resolve_image_path가 '#' 이후를 잘라 실제 파일명을 얻는다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

# 정규화 통계: pretrained backbone을 쓰지 않으므로 단순 0.5/0.5 ([-1, 1]).
# 전처리의 유일한 근원이므로 세 경로(JPEG/캐시 생성/캐시 로드)가 공유한다.
_IMAGE_MEAN = [0.5, 0.5, 0.5]
_IMAGE_STD = [0.5, 0.5, 0.5]


def resolve_image_path(image_dir: str | Path, list_line: str) -> Path:
    """이미지 리스트 한 줄을 ``raw/`` 아래 실제 파일 경로로 해석한다.

    Args:
        image_dir: 이미지 루트 (예: ``data/image``); 실제 파일은 그 아래
            ``raw/``에 있다.
        list_line: ``{split}.txt``의 한 줄. "<name>.jpg" 또는
            "<name>.jpg#<id>"(testcoco) 형식.

    Returns:
        ``{image_dir}/raw/{name}.jpg`` 경로. '#' 이후 접미사는 제거된다.
    """
    name = list_line.strip().split("#", 1)[0]
    return Path(image_dir) / "raw" / name


def build_resize_transform(image_size: int) -> Resize:
    """리사이즈만 하는 transform (dtype 보존: uint8 -> uint8).

    캐시 생성(preprocess_images.py)에서 사용한다. antialias bilinear는 uint8
    입력에 대해 결정적이므로, 이 결과를 저장해두면 즉석 로드와 동일해진다.

    Args:
        image_size: 정사각형으로 리사이즈할 한 변의 크기.

    Returns:
        ``(C, H, W)`` -> ``(C, image_size, image_size)`` (dtype 유지) transform.
    """
    return Resize((image_size, image_size), antialias=True)


def build_normalize_transform() -> Compose:
    """uint8 이미지를 float으로 바꾸고 정규화하는 transform.

    캐시 로드(dataset.py)에서 미리 리사이즈된 uint8 텐서에 적용한다.

    Returns:
        ``(C, H, W)`` uint8 -> ``(C, H, W)`` 정규화 float32 transform.
    """
    return Compose(
        [
            ConvertImageDtype(torch.float32),  # uint8 [0,255] -> float [0,1]
            Normalize(mean=_IMAGE_MEAN, std=_IMAGE_STD),
        ]
    )


def build_image_transform(image_size: int) -> Compose:
    """전체 전처리 파이프라인: resize -> float 변환 -> 정규화.

    JPEG 즉석 로드 경로(캐시 미사용)에서 사용한다.

    Args:
        image_size: 정사각형으로 리사이즈할 한 변의 크기.

    Returns:
        ``(C, H, W)`` uint8 텐서를 ``(C, image_size, image_size)`` 정규화
        float 텐서로 바꾸는 transform.
    """
    return Compose([build_resize_transform(image_size), build_normalize_transform()])


def load_image_tensor(
    path: str | Path,
    image_size: int | None = None,
    transform: Compose | None = None,
) -> Tensor:
    """이미지 하나를 읽어 정규화된 텐서로 반환한다 (JPEG 즉석 로드 경로).

    ``transform``을 직접 넘기면 재사용하고(권장 — 매 호출마다 새로 만들지
    않음), 없으면 ``image_size``로 그때그때 만든다.

    Args:
        path: 이미지 파일 경로.
        image_size: transform이 없을 때 사용할 리사이즈 크기.
        transform: 미리 만들어둔 :func:`build_image_transform` 결과.

    Returns:
        ``(C, image_size, image_size)`` float32 텐서.

    Raises:
        ValueError: image_size와 transform이 모두 없을 때.
    """
    if transform is None:
        if image_size is None:
            raise ValueError("load_image_tensor requires either image_size or transform")
        transform = build_image_transform(image_size)
    # read_image: (C, H, W) uint8. 흑백/투명 이미지도 RGB 3채널로 강제한다.
    image = read_image(str(path), mode=ImageReadMode.RGB)
    return transform(image)
