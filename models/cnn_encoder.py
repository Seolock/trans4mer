"""
===============================================================================
 파일: models/cnn_encoder.py
 목적:
    CNN 기반 이미지 인코더를 scratch로 구현한다. pretrained이나 완성된
    Vision Backbone(torchvision.models 등)을 일절 쓰지 않고 Conv / BatchNorm /
    Activation / Pooling / Residual Block을 모두 직접 조립한다 (ResNet 스타일).

 역할:
    이미지를 컨볼루션 스택으로 인코딩해 공간 feature map을 만들고, 그것을
    펼쳐 디코더의 Image Cross-Attention이 어텐션할 feature 시퀀스로 바꾼다.
    ViTEncoder와 교체 가능한 또 하나의 이미지 인코더다.

 입력 / 출력:
    forward(image):
        image: (batch, channels, image_size, image_size)  실수 텐서
        ->     (batch, num_patches, d_model)  공간 feature 시퀀스
               num_patches = 최종 feature map의 H' * W'

 구현 세부사항:
    - Stem: Conv2d(stride 2) + BatchNorm + ReLU + MaxPool(stride 2) 로 해상도를
      1/4로 줄인다.
    - 이후 `image_layers`개의 스테이지: 각 스테이지는 residual block 2개.
      첫 스테이지를 제외하고 각 스테이지의 첫 block이 stride 2로 다운샘플하며
      채널 수를 늘린다 (다운샘플/채널 변경 시 1x1 conv shortcut).
    - 최종 feature map (B, C, H', W')을 (B, H'*W', C)로 펼치고, 선형 투영으로
      d_model에 맞춘 뒤 학습 가능한 위치 임베딩을 더한다.
    - 모든 conv 가중치는 Transformer가 init_xavier로 random 초기화하며,
      BatchNorm의 게인/바이어스는 기본값(1/0)을 유지한다.
===============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from config.config import Config


class _ResidualBlock(nn.Module):
    """ResNet BasicBlock: (Conv-BN-ReLU-Conv-BN) + shortcut, 이후 ReLU.

    Args:
        in_channels: 입력 채널 수.
        out_channels: 출력 채널 수.
        stride: 첫 conv의 stride (2면 공간 해상도를 절반으로 줄인다).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        # 해상도나 채널이 바뀌면 residual 덧셈을 위해 shortcut을 맞춰준다.
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """residual block 하나를 적용한다.

        Args:
            x: ``(batch, in_channels, H, W)``.

        Returns:
            ``(batch, out_channels, H/stride, W/stride)``.
        """
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.activation(out)


class CNNEncoder(nn.Module):
    """이미지를 공간 feature 시퀀스로 인코딩하는 scratch ResNet 스타일 CNN.

    Args:
        config: 전체 프로젝트 설정 (multimodal 섹션이 이미지 하이퍼파라미터를
            제공하고, 출력 폭은 model.d_model에 맞춘다).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        mm, m = config.multimodal, config.model
        num_stages = mm.image_layers

        # 스테이지별 채널 수: 마지막 스테이지가 image_embed_dim이 되도록
        # 뒤에서부터 절반씩 줄여 채운다 (최소 16 채널 보장).
        channels = [
            max(mm.image_embed_dim // (2 ** (num_stages - 1 - i)), 16)
            for i in range(num_stages)
        ]

        # -------------------------------------------------- Stem (해상도 1/4)
        self.stem = nn.Sequential(
            nn.Conv2d(mm.image_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # -------------------------------------------------- Residual 스테이지
        stages: list[nn.Module] = []
        in_ch = channels[0]
        for i in range(num_stages):
            out_ch = channels[i]
            stride = 1 if i == 0 else 2  # 첫 스테이지만 해상도 유지, 이후 절반씩
            stages.append(_ResidualBlock(in_ch, out_ch, stride=stride))
            stages.append(_ResidualBlock(out_ch, out_ch, stride=1))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # -------------------------------------------------- 시퀀스화 + 투영
        # 최종 공간 해상도: stem에서 1/4, 이후 (num_stages-1)번 절반.
        final_grid = (mm.image_size // 4) // (2 ** (num_stages - 1))
        if final_grid < 1:
            raise ValueError(
                f"CNNEncoder: image_size {mm.image_size} is too small for "
                f"image_layers {num_stages} — feature map collapses to 0."
            )
        self.num_patches = final_grid * final_grid
        self.projection = nn.Linear(channels[-1], m.d_model)
        # 공간 위치를 위한 학습 가능한 위치 임베딩 (init_xavier가 random 초기화).
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, m.d_model)
        )
        self.embedding_dropout = nn.Dropout(m.embedding_dropout)

    def forward(self, image: Tensor) -> Tensor:
        """이미지 배치를 공간 feature 시퀀스로 인코딩한다.

        Args:
            image: ``(batch, channels, image_size, image_size)``.

        Returns:
            ``(batch, num_patches, d_model)`` feature 시퀀스.
        """
        # ===================== 1) Conv 스택 =====================
        # (B, C, H, W) -> (B, C', H', W')
        features = self.stages(self.stem(image))

        # ===================== 2) 시퀀스화 =====================
        # (B, C', H', W') -> (B, H'*W', C')
        batch_size, channels = features.size(0), features.size(1)
        features = features.view(batch_size, channels, -1).transpose(1, 2)

        # ===================== 3) d_model 투영 + 위치 임베딩 =====================
        # (B, N, C') -> (B, N, d_model)
        projected = self.projection(features)
        return self.embedding_dropout(projected + self.position_embedding)
