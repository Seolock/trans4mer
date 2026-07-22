"""
===============================================================================
 파일: trainer/metrics.py
 목적:
    검증과 테스트 중에 사용하는 순수 지표 함수들: 토큰 정확도, 시퀀스
    정확도, perplexity, corpus BLEU.

 역할:
    모델, dataloader, 디바이스에 의존하지 않도록 유지해서 각 함수를 단위
    테스트하기 쉽고 재사용하기 쉽게 한다. Evaluator(trainer/evaluator.py)가
    배치들에 걸쳐 *개수(count)* 변형을 집계하여 정확한 corpus-level 값을
    계산한다.

 입력 / 출력:
    - token_correct / sequence_correct는
        predictions: (batch, seq_len) 정수 토큰 id (이미 argmax된 것)
        targets    : (batch, seq_len) 패딩이 포함된 정수 토큰 id
      를 받아 (n_correct, n_total) 정수 카운트를 반환한다.
    - perplexity는 평균 cross-entropy(토큰당 nats)를 exp(loss)로
      매핑한다.
    - corpus_bleu는 토큰화된 hypotheses/references(토큰 리스트의 리스트)를
      받아 표준 BLEU-4를 [0, 1] 범위로 반환한다.

 구현 세부사항:
    - 모든 토큰 지표는 `pad_id`를 통해 패딩 위치를 *제외*한다.
    - perplexity는 (loss <= 20 -> ppl <= ~4.85e8) 클램핑되어 학습 초반의
      오버플로를 막는다.
    - BLEU: clipping이 적용된 modified n-gram precision + brevity penalty
      (Papineni et al., 2002). 0인 precision은 아주 작은 epsilon으로
      바닥을 깔아서, 학습 초기나 짧은 출력에서 에러 대신 0에 가까운
      점수를 낸다.
===============================================================================
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from torch import Tensor


def token_correct(predictions: Tensor, targets: Tensor, pad_id: int) -> tuple[int, int]:
    """올바르게 예측된 패딩이 아닌 토큰 개수를 센다.

    Args:
        predictions: ``(batch, seq_len)`` 예측된 토큰 id.
        targets: ``(batch, seq_len)`` 정답 토큰 id (패딩 포함).
        pad_id: 카운트에서 제외할 패딩 id.

    Returns:
        실제(패딩이 아닌) 타겟 위치에 대한 ``(n_correct, n_total)``.
    """
    mask = targets != pad_id
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


def sequence_correct(predictions: Tensor, targets: Tensor, pad_id: int) -> tuple[int, int]:
    """패딩이 아닌 *모든* 토큰이 올바르게 예측된 시퀀스 개수를 센다.

    Args:
        predictions: ``(batch, seq_len)`` 예측된 토큰 id.
        targets: ``(batch, seq_len)`` 정답 토큰 id (패딩 포함).
        pad_id: 패딩 id; 패딩 위치는 시퀀스를 부적격으로 만들지 않는다.

    Returns:
        ``(n_exactly_correct_sequences, batch_size)``.
    """
    mask = targets != pad_id
    # 위치가 일치하거나 패딩이면 "괜찮음"; 한 행은 전부 괜찮아야 한다.
    fine = (predictions == targets) | ~mask
    correct = fine.all(dim=1).sum().item()
    return int(correct), int(targets.size(0))


def token_accuracy(predictions: Tensor, targets: Tensor, pad_id: int) -> float:
    """올바르게 예측된 패딩이 아닌 토큰의 비율 (비어있으면 0.0)."""
    correct, total = token_correct(predictions, targets, pad_id)
    return correct / total if total > 0 else 0.0


def sequence_accuracy(predictions: Tensor, targets: Tensor, pad_id: int) -> float:
    """완전히 정확하게 재현된 시퀀스의 비율 (배치가 비었으면 0.0)."""
    correct, total = sequence_correct(predictions, targets, pad_id)
    return correct / total if total > 0 else 0.0


def perplexity(mean_cross_entropy: float) -> float:
    """exp(평균 토큰 cross-entropy), 학습 초반에 발산하지 않도록 클램핑됨.

    Args:
        mean_cross_entropy: 토큰당 평균 negative log-likelihood (nats).

    Returns:
        corpus perplexity.
    """
    return math.exp(min(mean_cross_entropy, 20.0))


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    """토큰 시퀀스에서 차수 ``n``인 n-gram의 다중집합(multiset)."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    """clipping이 적용된 n-gram precision과 brevity penalty를 쓰는 corpus-level BLEU.

    teacher-forced 지표와 함께 생성 품질을 추적할 수 있도록 제공된다
    (BLEU를 연결하는 방법은 Evaluator docstring 참고).

    Args:
        hypotheses: 예제마다 하나씩 토큰화된 모델 출력.
        references: 예제마다 하나씩 토큰화된 정답(단일 참조).
        max_n: 가장 높은 n-gram 차수 (4 = 표준 BLEU-4).

    Returns:
        ``[0, 1]`` 범위의 BLEU 점수.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    if not hypotheses:
        return 0.0

    matched = [0] * max_n  # 차수별로 clipping된 n-gram 일치 개수
    possible = [0] * max_n  # 차수별 hypothesis n-gram 총 개수
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngram_counts(hyp, n)
            ref_ngrams = _ngram_counts(ref, n)
            # "Clipped" precision: hypothesis의 n-gram은 참조에 등장하는
            # 횟수만큼만 유효한 것으로 센다.
            matched[n - 1] += sum(min(count, ref_ngrams[g]) for g, count in hyp_ngrams.items())
            possible[n - 1] += max(sum(hyp_ngrams.values()), 0)

    # n-gram precision들의 기하평균 (log(0)을 피하기 위한 epsilon 바닥).
    log_precision_sum = 0.0
    for n in range(max_n):
        precision = matched[n] / possible[n] if possible[n] > 0 else 0.0
        log_precision_sum += math.log(max(precision, 1e-9))
    geo_mean = math.exp(log_precision_sum / max_n)

    # brevity penalty는 참조보다 짧은 hypothesis에 불이익을 준다.
    brevity = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(hyp_len, 1))
    return brevity * geo_mean
