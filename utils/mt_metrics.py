"""
===============================================================================
 파일: utils/mt_metrics.py
 목적:
    번역 품질 자동 평가 지표를 계산하는 재사용 헬퍼:
    BLEU(sacreBLEU) / METEOR(NLTK) / COMET(unbabel-comet).

 역할:
    test.py가 테스트 세트를 번역한 뒤 세 지표를 함께 계산·기록하는 데
    사용한다. 최신 MT 논문 관례(BLEU 단독 지양, 신경망 기반 COMET을 주
    지표로, METEOR를 보조로 병기)에 맞춰 지표를 한곳에서 제공한다.

 입력 / 출력 (모든 함수 공통):
    hypotheses : list[str]  모델이 생성한 번역문 (한 줄에 한 문장)
    references : list[str]  정답 참조문 (hypotheses와 인덱스 정렬)
    sources    : list[str]  원문 소스문 (COMET 전용; hyp/ref와 인덱스 정렬)
    -> 지표 점수(float). METEOR/COMET은 의존성/다운로드가 필요하므로
       사용 불가 시 None을 반환한다 (graceful).

 구현 세부사항:
    - sacreBLEU는 필수 의존성이지만 METEOR/COMET은 선택적이다. 무거운
      import는 함수 안에서 lazy로 수행하고, 미설치·데이터 다운로드 실패·
      모델 로드 실패는 모두 잡아 경고를 남기고 None을 반환한다 — 그래야
      한 지표가 없어도 나머지 지표 계산과 test.py 실행이 중단되지 않는다.
    - 점수 스케일은 BLEU(0~100)에 맞춰 METEOR/COMET도 ×100으로 반환한다.
    - 학습 코퍼스가 lowercase + 토큰화 형태라 모델 출력도 그 형태다.
      COMET/METEOR는 자연문 기준이므로 반환 점수는 절대값보다 상대 비교용
      (모델 간 / 에폭 간)으로 해석해야 한다.
===============================================================================
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_bleu(
    hypotheses: list[str], references: list[str], lowercase: bool = True
) -> tuple[float, str]:
    """표준 sacreBLEU corpus BLEU와 재현용 시그니처를 계산한다.

    참조는 원문(detok) 그대로 넘기고 sacreBLEU 기본 13a 토크나이저가 hyp/ref를
    동일하게 토큰화한다 (fairseq --scoring sacrebleu 방식).

    Args:
        hypotheses: 모델 번역문.
        references: 정답 참조문.
        lowercase: True면 대소문자 무시(uncased) BLEU. 학습 코퍼스가
            소문자화되어 모델 출력도 소문자이므로 기본 True.

    Returns:
        ``(score, signature)`` — sacreBLEU 점수(0~100)와 재현용 시그니처
        문자열(예: ``BLEU|nrefs:1|case:lc|tok:13a|smooth:exp|version:2.x``).
    """
    from sacrebleu.metrics import BLEU

    metric = BLEU(lowercase=lowercase)
    score = metric.corpus_score(hypotheses, [references])
    return score.score, metric.get_signature().format()


def compute_meteor(
    hypotheses: list[str], references: list[str], lowercase: bool = True
) -> Optional[float]:
    """NLTK METEOR를 코퍼스 평균(문장별 점수 평균 ×100)으로 계산한다.

    METEOR는 sacreBLEU 같은 표준 토크나이저가 없으므로 공백 분리(``split``)로
    토큰화한다. 동의어 매칭(WordNet)은 영어 전용이라, 타깃이 de/fr이면 사실상
    exact + Porter-stem 매칭만 작동한다 (그래도 관례적으로 병기되는 지표).

    Args:
        hypotheses: 모델 번역문.
        references: 정답 참조문.
        lowercase: True면 토큰화 전에 소문자로 맞춘다 (BLEU와 동일 정책).

    Returns:
        METEOR 점수(0~100), 또는 nltk 미설치·데이터 준비 실패 시 ``None``.
    """
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        logger.warning("nltk 미설치 — METEOR 건너뜀 (`pip install nltk`).")
        return None

    # WordNet / OMW 데이터가 없으면 조용히 내려받는다 (오프라인이면 실패 -> 건너뜀).
    try:
        for resource in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)
    except Exception as error:  # noqa: BLE001 - 어떤 준비 실패든 graceful하게 건너뜀
        logger.warning("nltk 데이터 준비 실패 — METEOR 건너뜀: %s", error)
        return None

    total = 0.0
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = (hyp.lower() if lowercase else hyp).split()
        ref_tokens = (ref.lower() if lowercase else ref).split()
        total += meteor_score([ref_tokens], hyp_tokens)
    return 100.0 * total / max(len(hypotheses), 1)


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    num_workers: int = 2,
) -> Optional[float]:
    """참조 기반 COMET 시스템 점수를 계산한다 (unbabel-comet).

    최초 호출 시 ``model_name`` 체크포인트를 내려받는다(대형). CUDA가 없으면
    ``gpus=0``으로 CPU에서 실행되며 문장 수에 따라 수 분 걸릴 수 있다.

    Args:
        sources: 원문 소스문 (COMET은 src/mt/ref 3요소가 필요).
        hypotheses: 모델 번역문(mt).
        references: 정답 참조문(ref).
        model_name: COMET 모델 이름 (기본: 논문 표준 wmt22-comet-da).
        batch_size: COMET 추론 배치 크기.
        num_workers: DataLoader 워커 수. comet 2.2 + 최신 torch 조합에서는
            ``num_workers=0``이 ``multiprocessing_context`` 오류를 내므로 1 이상
            이어야 한다.

    Returns:
        COMET 시스템 점수(×100), 또는 unbabel-comet 미설치·다운로드·로드·계산
        실패 시 ``None``.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as error:
        logger.warning("unbabel-comet 사용 불가 — COMET 건너뜀 (%s).", error)
        return None
    except Exception as error:  # noqa: BLE001 - numpy/ABI 등 import-time 오류도 graceful
        logger.warning("unbabel-comet import 오류 — COMET 건너뜀: %s", error)
        return None

    try:
        checkpoint_path = download_model(model_name)
        model = load_from_checkpoint(checkpoint_path)
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        # gpus=0: Mac 등 CUDA 없는 환경에서 CPU 실행.
        # num_workers>=1: comet 2.2 + 최신 torch의 multiprocessing_context 오류 회피.
        output = model.predict(
            data, batch_size=batch_size, gpus=0, num_workers=num_workers, progress_bar=False
        )
        return 100.0 * float(output.system_score)
    except Exception as error:  # noqa: BLE001 - 다운로드/로드/계산 어떤 실패든 graceful
        logger.warning("COMET 로드/계산 실패 — COMET 건너뜀: %s", error)
        return None
