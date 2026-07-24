"""
===============================================================================
 파일: inference/decode.py
 목적:
    소스 토큰 id로부터 타겟 토큰 id를 만드는 재사용 가능한 greedy 디코딩
    함수. 어휘집/BPE에 의존하지 않고 "id -> id" 수준에서만 동작한다.

 역할:
    두 곳에서 공유된다:
      - inference/translator.py (원본 문장 번역의 greedy 경로)
      - trainer/evaluator.py (학습 중 valid BLEU 계산)
    모델의 encode/decode/generator만 사용하므로 텍스트-only와 Multimodal
    (이미지 cross-attention) 모두를 그대로 지원한다.

 입력 / 출력:
    입력 : model, src (batch, src_len) 오른쪽 패딩된 소스 id,
           bos/eos/pad id, 최대/최소 생성 길이, 선택적 image (batch, C, H, W).
    출력 : 예제마다 생성된 타겟 id 리스트 (BOS/EOS/패딩 제외).

 구현 세부사항:
    - 소스(와 이미지)는 한 번만 인코딩하고, 매 스텝 전체 prefix를 다시
      디코딩한다 (KV 캐시 없음 — 명확성 우선). 완료된 행은 배치 형태를
      직사각형으로 유지하기 위해 패딩을 계속 이어 붙인다.
    - step <= min_length 동안에는 EOS logit을 -inf로 막아 최소 길이를
      강제한다.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
    min_length: int = 1,
    image: Optional[Tensor] = None,
) -> list[list[int]]:
    """argmax greedy 디코딩으로 타겟 id 시퀀스를 생성한다.

    Args:
        model: Transformer (make_source_mask / encode / encode_image /
            decode / generator 메서드를 가진 모델).
        src: ``(batch, src_len)`` 오른쪽 패딩된 소스 id (device 위에 있음).
        bos_id: 타겟 시퀀스 시작(BOS) id — 디코딩이 여기서 시작한다.
        eos_id: 타겟 시퀀스 종료(EOS) id — 나오면 해당 행을 종료한다.
        pad_id: 완료된 행을 이어 붙일 패딩 id.
        max_length: 생성할 최대 토큰 개수.
        min_length: 이 길이 이전에는 EOS를 금지한다.
        image: ``(batch, C, H, W)`` 이미지 텐서 (Multimodal), 또는 None.

    Returns:
        예제마다 생성된 토큰 id 리스트 (BOS/EOS/패딩 제외).
    """
    device = src.device
    batch_size = src.size(0)

    # 소스(와 이미지)는 한 번만 인코딩한다.
    src_mask = model.make_source_mask(src)
    memory = model.encode(src, src_mask)
    image_memory = model.encode_image(image) if image is not None else None

    sequences = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    alive = torch.ones(batch_size, dtype=torch.bool, device=device)

    for step in range(1, max_length + 1):
        decoded = model.decode(
            sequences, memory, memory_mask=src_mask, image_memory=image_memory
        )
        logits = model.generator(decoded[:, -1, :]).float()
        if step <= min_length:
            logits[:, eos_id] = float("-inf")  # 최소 길이 강제
        next_token = logits.argmax(dim=-1)
        # 완료된 행은 배치 형태 유지를 위해 계속 패딩을 낸다.
        next_token = torch.where(alive, next_token, torch.full_like(next_token, pad_id))
        sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
        alive = alive & (next_token != eos_id)
        if not alive.any():
            break

    # BOS 제거, EOS/패딩에서 자르기.
    results: list[list[int]] = []
    for row in sequences[:, 1:].tolist():
        tokens: list[int] = []
        for token_id in row:
            if token_id in (eos_id, pad_id):
                break
            tokens.append(token_id)
        results.append(tokens)
    return results
