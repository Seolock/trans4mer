"""
===============================================================================
 파일: utils/image.py
 목적:
    Multimodal MT를 위한 이미지 로딩/전처리 헬퍼. 디스크의 JPEG를 읽어
    모델이 바로 먹을 수 있는 정규화된 텐서로 변환한다.

 역할:
    Dataset(dataset.py의 TranslationDataset)과 추론 래퍼
    (inference/translator.py의 Translator)가 동일한 방식으로 이미지를
    로드하도록 한 곳에서 규칙을 관리한다 — 그래야 학습과 추론의 전처리가
    절대 어긋나지 않는다.

 입력 / 출력:
    resolve_image_path(image_dir, list_line) -> Path   (raw/ 아래 실제 파일)
    load_image_tensor(path, transform)        -> (C, H, W) float 텐서
    build_image_transform(image_size)         -> torchvision transform

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


def build_image_transform(image_size: int) -> Compose:
    """정규화 파이프라인을 만든다: resize -> float 변환 -> 정규화.

    Args:
        image_size: 정사각형으로 리사이즈할 한 변의 크기.

    Returns:
        ``(C, H, W)`` uint8 텐서를 ``(C, image_size, image_size)`` float
        텐서(채널별 mean=std=0.5로 정규화, 대략 [-1, 1])로 바꾸는 transform.
    """
    return Compose(
        [
            Resize((image_size, image_size), antialias=True),
            ConvertImageDtype(torch.float32),  # uint8 [0,255] -> float [0,1]
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def load_image_tensor(
    path: str | Path,
    image_size: int | None = None,
    transform: Compose | None = None,
) -> Tensor:
    """이미지 하나를 읽어 정규화된 텐서로 반환한다.

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
