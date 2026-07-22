"""
===============================================================================
 파일: inference/predict.py
 목적:
    최종 사용자를 위한 추론: 체크포인트 + 토크나이저를 로드하고 greedy
    디코딩, 샘플링(temperature / top-k / top-p / repetition penalty) 또는
    빔 서치를 통해 텍스트를 번역/변환한다.

 역할:
    `Predictor`는 test.py와 아래 CLI가 사용하는 단일 진입점이다. 전략
    선택은 추론 설정을 따른다:
        beam_size > 1                  -> 빔 서치 (inference/beam_search.py)
        beam_size == 1, do_sample 꺼짐 -> 순수 greedy (argmax)
        beam_size == 1, do_sample 켜짐 -> 필터링된 샘플링

 입력 / 출력:
    predict(text: str) -> str   (문장 하나 입력, 문장 하나 출력)
    predict_batch(list[str]) -> list[str]
    CLI: python -m inference.predict --checkpoint checkpoints/best.pt \
             --tokenizer data/tokenizer.json --text "..."

 구현 세부사항:
    - 모델은 체크포인트에 *내장된* 설정 dict로부터 재구성되므로, 아키텍처
      플래그가 학습된 가중치와 절대 어긋날 수 없다.
    - greedy/샘플링 루프는 매 스텝마다 전체 prefix를 다시 디코딩한다
      (KV 캐시 없음 — 명확성을 우선함; 최적화 경로는 README 참고).
    - repetition penalty는 CTRL(Keskar et al., 2019)을 따른다: 이미
      생성된 토큰의 양수 logit은 penalty로 나누고, 음수는 곱한다.
      top-k와 top-p 필터는 softmax 전에 logit을 -inf로 마스킹한다.
    - 생성 길이는 inference.max_length와 positional encoding의
      model.max_seq_length 둘 다에 의해 상한이 걸린다.
===============================================================================
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch import Tensor

from config.config import Config
from datasets.tokenizer import Tokenizer
from inference.beam_search import BeamSearchDecoder
from models.transformer import Transformer
from trainer.checkpoint import CheckpointManager
from utils.misc import get_device


def apply_repetition_penalty(logits: Tensor, generated: Tensor, penalty: float) -> Tensor:
    """이미 생성된 prefix에 등장하는 토큰들을 억제한다.

    Args:
        logits: ``(batch, vocab)`` 다음 토큰 logits (제자리에서 수정되지 않음).
        generated: ``(batch, prefix_len)`` 지금까지 생성된 토큰들.
        penalty: 1.0보다 큰 값은 반복에 불이익을 준다; 1.0이면 아무 효과 없음.

    Returns:
        조정된 logits.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for row in range(logits.size(0)):
        seen = generated[row].unique()
        row_logits = logits[row, seen]
        # CTRL 규칙: 음수 logit을 나누면 오히려 확률이 *올라가므로*,
        # 음수는 곱하는 방식으로 처리한다.
        logits[row, seen] = torch.where(
            row_logits > 0, row_logits / penalty, row_logits * penalty
        )
    return logits


def filter_top_k(logits: Tensor, top_k: int) -> Tensor:
    """행마다 가장 높은 ``top_k``개의 logit만 유지한다 (0이면 비활성화)."""
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Nucleus 필터링: 누적 확률이 ``top_p``를 넘는 가장 작은 토큰 집합만
    남긴다 (1.0이면 비활성화)."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # 오른쪽으로 시프트해서 임계값을 처음 넘는 토큰도 유지되도록 한다.
    remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    remove_original = remove.scatter(dim=-1, index=sorted_indices, src=remove)
    return logits.masked_fill(remove_original, float("-inf"))


class Predictor:
    """학습된 모델을 로드하고 설정된 전략으로 텍스트를 디코딩한다.

    Args:
        model: ``device``에 있는 학습된 Transformer.
        tokenizer: 모델이 학습에 사용한 토크나이저.
        config: 전체 설정 (inference 섹션이 디코딩을 제어함).
        device: 디코딩할 디바이스.
    """

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        config: Config,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        # positional encoding이 지원하는 길이를 절대 넘지 않도록 한다.
        self.max_length = min(config.inference.max_length, config.model.max_seq_length - 1)

    # ------------------------------------------------------------- 팩토리
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[torch.device] = None,
        config_override: Optional[Config] = None,
    ) -> "Predictor":
        """체크포인트 파일로부터 모델 + 설정을 재구성한다.

        Args:
            checkpoint_path: Trainer가 기록한 .pt 파일 경로.
            tokenizer_path: 토크나이저 JSON 경로.
            device: 대상 디바이스 (생략하면 자동 감지).
            config_override: 체크포인트의 것 대신 이 설정의
                inference/dataset 섹션을 사용한다 (모델 섹션은 항상
                가중치와 맞도록 체크포인트에서 온다).

        Returns:
            바로 사용 가능한 :class:`Predictor`.
        """
        device = device or get_device()
        state = CheckpointManager.load(checkpoint_path, map_location=device)
        config = Config.from_dict(state["config"])
        if config_override is not None:
            # 디코딩과 데이터 옵션은 조정될 수 있지만 아키텍처는 안 된다.
            config.inference = config_override.inference
            config.dataset = config_override.dataset
        model = Transformer(config).to(device)
        model.load_state_dict(state["model"])
        tokenizer = Tokenizer.load(tokenizer_path)
        return cls(model, tokenizer, config, device)

    # -------------------------------------------------------------- 디코딩
    @torch.no_grad()
    def greedy_search(self, src: Tensor) -> list[list[int]]:
        """greedy / 샘플링 디코딩 (전략은 ``do_sample``에 따라 결정됨).

        Args:
            src: ``(batch, src_len)`` 오른쪽 패딩된 소스 id.

        Returns:
            예제마다 생성된 토큰 id (BOS/EOS 제외).
        """
        inf = self.config.inference
        batch_size = src.size(0)
        bos, eos, pad = self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id

        src_mask = self.model.make_source_mask(src)
        memory = self.model.encode(src, src_mask)

        sequences = torch.full((batch_size, 1), bos, dtype=torch.long, device=self.device)
        alive = torch.ones(batch_size, dtype=torch.bool, device=self.device)

        for step in range(1, self.max_length + 1):
            decoded = self.model.decode(sequences, memory, memory_mask=src_mask)
            logits = self.model.generator(decoded[:, -1, :]).float()

            logits = apply_repetition_penalty(logits, sequences, inf.repetition_penalty)
            if step <= inf.min_length:
                logits[:, eos] = float("-inf")  # 최소 길이를 강제한다

            if inf.do_sample:
                # temperature -> top-k -> top-p -> 샘플링
                logits = logits / max(inf.temperature, 1e-6)
                logits = filter_top_p(filter_top_k(logits, inf.top_k), inf.top_p)
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            else:
                next_token = logits.argmax(dim=-1)

            # 완료된 행은 형태를 직사각형으로 유지하기 위해 계속 패딩을 낸다.
            next_token = torch.where(alive, next_token, torch.full_like(next_token, pad))
            sequences = torch.cat([sequences, next_token.unsqueeze(1)], dim=1)
            alive = alive & (next_token != eos)
            if not alive.any():
                break

        # BOS 제거, EOS에서 자르기, 패딩 제거.
        results: list[list[int]] = []
        for row in sequences[:, 1:].tolist():
            tokens: list[int] = []
            for token_id in row:
                if token_id in (eos, pad):
                    break
                tokens.append(token_id)
            results.append(tokens)
        return results

    @torch.no_grad()
    def beam_search(self, src: Tensor) -> list[list[int]]:
        """빔 서치 디코딩 (inference/beam_search.py 참고)."""
        inf = self.config.inference
        decoder = BeamSearchDecoder(
            self.model,
            bos_id=self.tokenizer.bos_id,
            eos_id=self.tokenizer.eos_id,
            pad_id=self.tokenizer.pad_id,
            beam_size=inf.beam_size,
            max_length=self.max_length,
            min_length=inf.min_length,
            length_penalty=inf.length_penalty,
        )
        return decoder.search(src)

    # ------------------------------------------------------------ 공개 API
    def predict_batch(self, texts: list[str]) -> list[str]:
        """원본 문자열 배치를 번역한다.

        Args:
            texts: 입력 문장들.

        Returns:
            디코딩된 출력 문장들 (설정에 따라 전략 선택).
        """
        encoded = [
            self.tokenizer.encode(text)[: self.config.model.max_seq_length] for text in texts
        ]
        max_len = max(len(ids) for ids in encoded)
        src = torch.full(
            (len(texts), max_len), self.tokenizer.pad_id, dtype=torch.long, device=self.device
        )
        for row, ids in enumerate(encoded):
            src[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

        use_beam = self.config.inference.beam_size > 1
        outputs = self.beam_search(src) if use_beam else self.greedy_search(src)
        return [self.tokenizer.decode(ids) for ids in outputs]

    def predict(self, text: str) -> str:
        """문자열 하나를 번역한다."""
        return self.predict_batch([text])[0]


# --------------------------------------------------------------------- CLI
def main() -> None:
    """커맨드라인 진입점: 학습된 체크포인트로 --text를 디코딩한다."""
    parser = argparse.ArgumentParser(description="Decode text with a trained Transformer")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="checkpoint path")
    parser.add_argument("--tokenizer", default="data/tokenizer.json", help="tokenizer JSON path")
    parser.add_argument("--text", required=True, help="input sentence to decode")
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="inference overrides (repeatable), e.g. --set inference.beam_size=1",
    )
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(args.checkpoint, args.tokenizer)
    if args.set:
        predictor.config.apply_overrides(args.set)
    print(predictor.predict(args.text))


if __name__ == "__main__":
    main()
