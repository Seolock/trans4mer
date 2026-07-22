"""
===============================================================================
 파일: datasets/collate.py
 목적:
    배치 결합(collation): 길이가 제각각인 id 시퀀스 리스트를 패딩된
    텐서로 만들고 teacher-forcing 입력/출력 shift를 생성한다.

 역할:
    모든 DataLoader에 `collate_fn`으로 전달된다. 여기서 원본 타겟
    시퀀스 [BOS, y1, ..., yn, EOS]가 다음으로 바뀐다:
        디코더 입력 : [BOS, y1, ..., yn]      (tgt에서 마지막 토큰 제외)
        디코더 타겟 : [y1, ..., yn, EOS]      (tgt에서 첫 토큰 제외)
    이렇게 하면 디코더의 위치 t가 위치 t+1의 토큰을 예측하도록 학습된다.

 입력 / 출력:
    입력 : Dataset에서 온 {"src": list[int], "tgt": list[int]}의 리스트.
           Multimodal 데이터셋이면 각 아이템에 "image": (C, H, W) 텐서가
           추가로 들어있다.
    출력 : {
        "src"       : (batch, max_src_len)      int64, 오른쪽 패딩
        "tgt_input" : (batch, max_tgt_len - 1)  int64, 오른쪽 패딩
        "tgt_output": (batch, max_tgt_len - 1)  int64, 오른쪽 패딩
        "image"     : (batch, C, H, W)          float — 아이템에 이미지가
                      있을 때만 포함 (텍스트-only면 이 키가 없다).
    }

 구현 세부사항:
    - 오른쪽 패딩(끝에 패딩을 붙이는 방식)이 프로젝트 전역에서 가정된다:
      이렇게 하면 모든 행의 위치 0이 BOS로 유지되고, 모든 어텐션 행에
      적어도 하나의 실제 key가 존재하는 것이 보장된다 (models/utils.py의
      마스크 설명 참고).
    - `tgt_output`의 패딩은 ignore_index를 통해 손실 계산에서 무시된다.
    - 이미지는 데이터셋에서 이미 고정 크기로 리사이즈되므로 패딩 없이
      단순 stack한다. 이미지 키는 아이템에 존재할 때만 배치에 넣어,
      텍스트-only/레거시 경로의 배치 형식은 그대로 유지된다.
    - num_workers > 0에서도 피클 가능하도록 (단순 함수가 아니라) 클래스로
      구현했으며, 패딩 id를 한 번만 바인딩한다.
===============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor


class Seq2SeqCollator:
    """배치를 패딩하고 teacher-forcing shift를 만든다.

    Args:
        pad_id: 모든 시퀀스를 오른쪽 패딩할 때 쓰는 패딩-토큰 id.
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def _pad_batch(self, sequences: list[list[int]]) -> Tensor:
        """id 리스트들의 리스트를 하나의 (batch, max_len) 텐서로 오른쪽 패딩한다."""
        max_len = max(len(seq) for seq in sequences)
        batch = torch.full((len(sequences), max_len), self.pad_id, dtype=torch.long)
        for row, seq in enumerate(sequences):
            batch[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return batch

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, Tensor]:
        """배치 하나를 결합한다.

        Args:
            examples: :class:`Seq2SeqDataset`이 만들어낸 아이템들.

        Returns:
            패딩된 ``src``, ``tgt_input``, ``tgt_output``을 담은 dict.
        """
        src = self._pad_batch([ex["src"] for ex in examples])
        tgt = self._pad_batch([ex["tgt"] for ex in examples])
        batch = {
            "src": src,
            # 패딩 *이후에* shift한다: 마지막 컬럼을 버리면 패딩이나 마지막
            # EOS 둘 중 하나가 제거되는데, 이는 정확히 입력에서 없어져야
            # 할 것이다.
            "tgt_input": tgt[:, :-1].contiguous(),
            "tgt_output": tgt[:, 1:].contiguous(),
        }
        # Multimodal: 이미지가 있는 데이터셋이면 고정 크기 이미지를 stack한다.
        # (C, H, W) 텐서들 -> (batch, C, H, W). 이미지 키가 없으면 위 딕셔너리를
        # 그대로 반환하므로 텍스트-only 배치 형식은 변하지 않는다.
        if examples and "image" in examples[0]:
            batch["image"] = torch.stack([ex["image"] for ex in examples], dim=0)
        return batch
