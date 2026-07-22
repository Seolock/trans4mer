"""
===============================================================================
 파일: inference/beam_search.py
 목적:
    GNMT 길이 정규화를 적용한 배치 빔 서치(beam search) 디코딩.

 역할:
    고품질 디코딩 전략. 매 스텝마다 단 하나의 최선 토큰에 확정하는(greedy)
    대신, `beam_size`개의 가장 확률 높은 부분 가설(hypothesis)을 계속
    살려두고 병렬로 확장한다. 이렇게 하면 초반 토큰들이 국소적으로는
    최선이 아니어 보이는 시퀀스도 놓치지 않고 회복할 수 있다.

 입력 / 출력:
    search(src):
        src: (batch, src_len) int64 소스 id (오른쪽 패딩)
        ->   길이가 `batch`인 토큰-id 리스트들 — 예제마다 최선의 가설
             (앞의 BOS와 뒤의 EOS는 제외).

 구현 세부사항 (전형적인 "2K 후보" 공식):
    - 소스는 한 번만 인코딩된다; memory 행들이 `beam_size`번 반복되어
      모든 예제의 모든 빔이 하나의 텐서로 디코딩된다.
    - 점수는 누적 로그 확률이다. 0번째 스텝에서는 한 예제의 모든 빔이
      동일하므로, 빔 0을 제외한 나머지는 -1e9에서 시작한다 — 그렇지
      않으면 top-k가 같은 토큰의 K개 복사본을 골라버릴 것이다.
    - 매 스텝마다 예제당 상위 2K개의 후보를 취한다: 이 중 K개가 EOS로
      끝나 완료된 목록으로 물러나더라도, K개의 살아있는 빔이 남는다.
    - 완료된 가설은 score / lp(length)로 순위가 매겨지며
      lp(n) = ((5 + n) / 6)^alpha (Wu et al., 2016) — 이것이 없으면
      더 긴 출력이 추가 로그 확률 항 때문에 부당하게 불이익을 받는다.
    - `min_length` 이전에는 EOS가 금지된다.
    - KV 캐시 없음: 매 스텝마다 전체 prefix에 대해 디코더를 다시 실행한다.
      명확하지만 정석적인 방법이다; 캐싱이 자연스러운 첫 번째
      최적화 지점이다 (README 참고).
===============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor

from models.transformer import Transformer


class BeamSearchDecoder:
    """학습된 Transformer에 대한 길이 정규화 배치 빔 서치.

    Args:
        model: 학습된 모델 (eval 모드; 이 클래스가 직접 no_grad를 호출함).
        bos_id: 시퀀스 시작 토큰 id (디코딩이 이 토큰에서 시작함).
        eos_id: 시퀀스 종료 토큰 id (가설을 완료시킴).
        pad_id: 물러난 예제의 행을 패딩할 때 쓰는 패딩 id.
        beam_size: 예제당 유지할 살아있는 가설 개수.
        max_length: 생성할 최대 토큰 개수.
        min_length: 이 개수보다 적게 생성된 시점에서는 EOS가 마스킹됨.
        length_penalty: GNMT alpha; 0이면 길이 정규화를 비활성화.
    """

    def __init__(
        self,
        model: Transformer,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        beam_size: int = 4,
        max_length: int = 64,
        min_length: int = 1,
        length_penalty: float = 0.6,
    ) -> None:
        self.model = model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.beam_size = beam_size
        self.max_length = max_length
        self.min_length = max(1, min_length)
        self.length_penalty = length_penalty

    def _length_normalized(self, score: float, length: int) -> float:
        """GNMT 길이 페널티: score / ((5 + length) / 6) ** alpha."""
        if self.length_penalty == 0.0:
            return score
        return score / (((5.0 + length) / 6.0) ** self.length_penalty)

    @torch.no_grad()
    def search(self, src: Tensor, image: Tensor | None = None) -> list[list[int]]:
        """배치 내 모든 소스 시퀀스에 대해 최선의 가설을 디코딩한다.

        Args:
            src: ``(batch, src_len)`` 오른쪽 패딩된 소스 id.
            image: ``(batch, C, H, W)`` 이미지 텐서 (Multimodal), 또는 None.

        Returns:
            예제마다 최선의 토큰-id 리스트 (BOS/EOS 제외).
        """
        self.model.eval()
        device = src.device
        batch_size, beam_size = src.size(0), self.beam_size
        vocab_size = self.model.generator.out_features

        # ---------------------------------------------------- 한 번만 인코딩
        src_mask = self.model.make_source_mask(src)
        memory = self.model.encode(src, src_mask)
        # 확장 이후 행 배치: 예제 b는 [b*K, (b+1)*K) 범위의 행들을 소유한다.
        memory = memory.repeat_interleave(beam_size, dim=0)
        memory_mask = src_mask.repeat_interleave(beam_size, dim=0)

        # 이미지도 한 번만 인코딩하고 memory와 동일하게 빔 개수만큼 반복한다
        # (같은 행 정렬 규칙을 유지해야 각 빔이 자기 이미지를 보게 된다).
        image_memory = None
        if image is not None:
            image_memory = self.model.encode_image(image)
            image_memory = image_memory.repeat_interleave(beam_size, dim=0)

        # ------------------------------------------------------ 빔 상태
        sequences = torch.full(
            (batch_size * beam_size, 1), self.bos_id, dtype=torch.long, device=device
        )
        # 0번째 스텝에서의 중복 빔 억제 (파일 상단 설명 참고).
        scores = torch.full((batch_size, beam_size), -1e9, device=device)
        scores[:, 0] = 0.0
        scores = scores.view(-1)  # (batch * beam,)

        # 예제별 완료된 가설: (정규화된_점수, 토큰들) 튜플의 리스트.
        finished: list[list[tuple[float, list[int]]]] = [[] for _ in range(batch_size)]
        example_done = [False] * batch_size

        for step in range(1, self.max_length + 1):
            # -------------------------------------------- 모든 빔을 확장
            decoded = self.model.decode(
                sequences, memory, memory_mask=memory_mask, image_memory=image_memory
            )
            # 이번 스텝에서는 가장 최근 위치의 분포만 중요하다.
            logits = self.model.generator(decoded[:, -1, :])
            log_probs = torch.log_softmax(logits.float(), dim=-1)  # (B*K, V)

            if step <= self.min_length:
                log_probs[:, self.eos_id] = -1e9  # 아직 완료하기엔 너무 짧음

            # 모든 (빔, 토큰) 확장에 대한 누적 후보 점수.
            candidates = (scores.unsqueeze(1) + log_probs).view(batch_size, beam_size * vocab_size)

            # 2K개의 후보는 K개가 EOS로 완료되더라도 K개의 살아있는 빔을 보장한다.
            top_scores, top_indices = candidates.topk(2 * beam_size, dim=1)
            source_beam = top_indices // vocab_size  # 어느 빔을 확장하는지
            token = top_indices % vocab_size         # 어느 토큰을 붙이는지

            # -------------------------------- 예제별로 생존자 선택
            next_rows: list[Tensor] = []      # 살아있는 빔당 (이전 행, 새 토큰)
            next_scores: list[float] = []
            for b in range(batch_size):
                if example_done[b]:
                    # 텐서 형태를 직사각형으로 유지하기 위해 자리표시자
                    # 행을 남겨둔다.
                    for k in range(beam_size):
                        next_rows.append(
                            torch.cat(
                                [sequences[b * beam_size + k],
                                 torch.tensor([self.pad_id], device=device)]
                            )
                        )
                        next_scores.append(-1e9)
                    continue

                live = 0
                for candidate in range(2 * beam_size):
                    if live >= beam_size:
                        break
                    beam_k = int(source_beam[b, candidate].item())
                    token_id = int(token[b, candidate].item())
                    score = float(top_scores[b, candidate].item())
                    row = b * beam_size + beam_k

                    if token_id == self.eos_id:
                        # 이 가설을 완료 목록으로 물린다 (BOS는 제거;
                        # EOS는 저장하지 않음).
                        tokens = sequences[row, 1:].tolist()
                        finished[b].append((self._length_normalized(score, len(tokens)), tokens))
                    else:
                        next_rows.append(
                            torch.cat(
                                [sequences[row], torch.tensor([token_id], device=device)]
                            )
                        )
                        next_scores.append(score)
                        live += 1

                # 예제는 충분한 가설을 모으면 완료 처리된다.
                if len(finished[b]) >= beam_size:
                    example_done[b] = True

            sequences = torch.stack(next_rows, dim=0)
            scores = torch.tensor(next_scores, device=device)
            if all(example_done):
                break

        # --------------------------------------------------------- 결과
        results: list[list[int]] = []
        for b in range(batch_size):
            if finished[b]:
                best_score, best_tokens = max(finished[b], key=lambda item: item[0])
                results.append(best_tokens)
            else:
                # EOS 없이 길이가 소진됨: 살아있는 최선의 빔으로 대체한다
                # (해당 행들은 b*K .. b*K+K-1이며, 최고 점수가 이긴다).
                rows = slice(b * beam_size, (b + 1) * beam_size)
                best_row = int(torch.argmax(scores[rows]).item()) + b * beam_size
                results.append(sequences[best_row, 1:].tolist())
        return results
