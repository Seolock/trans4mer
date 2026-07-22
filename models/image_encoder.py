"""
===============================================================================
 파일: models/image_encoder.py
 목적:
    설정(multimodal.image_encoder)에 따라 사용할 이미지 인코더를 선택하는
    Wrapper. ViTEncoder(패치 기반)와 CNNEncoder(ResNet 스타일)를 동일한
    인터페이스로 감싸, 다른 코드가 인코더 종류에 무관하게 이미지 feature를
    얻도록 한다 (models/positional_encoding.py의 factory 패턴과 동일한 방식).

 역할:
    Transformer가 소유하는 단일 이미지 인코딩 진입점. 두 scratch 인코더
    중 하나를 만들고, 선택적으로 파라미터를 동결한다. 텍스트 인코더와
    마찬가지로 하나의 독립적인 모듈이다.

 입력 / 출력:
    forward(image):
        image: (batch, channels, image_size, image_size)
        ->     (batch, num_patches, d_model)  patch/feature 시퀀스
    이미지 피처의 마지막 차원은 항상 model.d_model이라 Image
    Cross-Attention이 디코더와 같은 폭으로 어텐션할 수 있다.

 구현 세부사항:
    - 완성된 Vision Backbone을 불러오지 않는다; ViTEncoder / CNNEncoder는
      둘 다 scratch 구현이다.
    - freeze_image_encoder=true면 파라미터의 requires_grad를 꺼서 이미지
      인코더를 동결한다 (기본은 false — 번역 손실로 함께 end-to-end 학습).
===============================================================================
"""

from __future__ import annotations

from torch import Tensor, nn

from config.config import Config
from models.cnn_encoder import CNNEncoder
from models.vit_encoder import ViTEncoder


class ImageEncoder(nn.Module):
    """설정으로 ViT/CNN 이미지 인코더를 선택하는 Wrapper.

    Args:
        config: 전체 프로젝트 설정 (multimodal.image_encoder가 종류를 결정).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        mm = config.multimodal
        if mm.image_encoder == "vit":
            self.encoder: nn.Module = ViTEncoder(config)
        elif mm.image_encoder == "cnn":
            self.encoder = CNNEncoder(config)
        else:
            raise ValueError(
                f"Unknown image_encoder '{mm.image_encoder}'. Available: ['vit', 'cnn']"
            )

        # 선택적 동결: 기본은 false (번역 손실로 함께 학습).
        if mm.freeze_image_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def forward(self, image: Tensor) -> Tensor:
        """이미지 배치를 feature 시퀀스로 인코딩한다.

        Args:
            image: ``(batch, channels, image_size, image_size)``.

        Returns:
            ``(batch, num_patches, d_model)`` 이미지 feature 시퀀스.
        """
        return self.encoder(image)
