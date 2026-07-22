"""
===============================================================================
 파일: utils/text.py
 목적:
    BPE 파이프라인 전반에서 공유되는 텍스트 처리 헬퍼:
    간단한 토큰화와 BPE 분절 제거(de-BPE).

 역할:
    preprocess.py(코퍼스 전처리), inference/translator.py(입력 문장 처리),
    test.py(BLEU 참조 문장 처리)가 모두 이 함수들을 사용하므로, 학습과
    추론의 전처리가 절대 어긋나지 않는다.

 입력 / 출력:
    simple_tokenize : str -> list[str]   (구두점 분리 + 선택적 소문자화)
    remove_bpe      : str -> str         ("@@ " 분절 마커 제거)

 구현 세부사항:
    - simple_tokenize는 유니코드 인식 정규식으로 단어와 구두점을 분리한다
      ("bushes." -> ["bushes", "."]) — Moses 토크나이저 없이도 BPE가
      구두점에 오염되지 않게 하는 최소한의 전처리다.
    - remove_bpe는 subword-nmt의 표준 복원 규칙을 따른다:
      "un@@ believ@@ able" -> "unbelievable". 줄 끝의 "@@"도 처리하기
      위해 공백을 덧붙인 뒤 치환하고 다시 strip하는 고전적 트릭을 쓴다.
===============================================================================
"""

from __future__ import annotations

import re

# 유니코드 단어(\w+: 독일어 움라우트 포함) 또는 단어/공백이 아닌 문자 1개.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """원본 문장을 단어/구두점 토큰으로 분리한다.

    Args:
        text: 원본 문장 (예: "A man is riding a bicycle.").
        lowercase: True면 분리 전에 소문자로 변환한다. preprocess.py와
            translate.py가 반드시 같은 값을 사용해야 한다
            (config: dataset.lowercase).

    Returns:
        토큰 리스트 (예: ["a", "man", "is", "riding", "a", "bicycle", "."]).
    """
    if lowercase:
        text = text.lower()
    return _TOKEN_PATTERN.findall(text)


def remove_bpe(text: str, separator: str = "@@") -> str:
    """BPE 분절 마커를 제거해 서브워드를 원래 단어로 복원한다.

    Args:
        text: BPE가 적용된 공백 구분 문자열
            (예: "ein mann f@@ ährt fahrrad .").
        separator: BPE 분절 마커 (subword-nmt 기본값 "@@").

    Returns:
        복원된 문자열 (예: "ein mann fährt fahrrad .").
    """
    # 끝에 공백을 붙여 줄 끝의 "word@@"까지 한 번의 치환으로 처리한다.
    return (text + " ").replace(separator + " ", "").rstrip()
