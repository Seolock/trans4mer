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
    detokenize      : str -> str         (구두점 재부착; simple_tokenize의 역)

 구현 세부사항:
    - simple_tokenize는 유니코드 인식 정규식으로 단어와 구두점을 분리한다
      ("bushes." -> ["bushes", "."]) — Moses 토크나이저 없이도 BPE가
      구두점에 오염되지 않게 하는 최소한의 전처리다.
    - remove_bpe는 subword-nmt의 표준 복원 규칙을 따른다:
      "un@@ believ@@ able" -> "unbelievable". 줄 끝의 "@@"도 처리하기
      위해 공백을 덧붙인 뒤 치환하고 다시 strip하는 고전적 트릭을 쓴다.
    - detokenize는 simple_tokenize의 역연산이다. 모델 출력은 학습 코퍼스와
      같은 토큰화 형태("... velo .")로 나오므로, 구두점을 인접 단어에 다시
      붙여 원문(raw) 참조와 같은 자연문("... velo.")으로 복원한다. 그래야
      sacreBLEU/COMET/METEOR가 hyp와 raw ref를 같은 기준으로 비교한다.
===============================================================================
"""

from __future__ import annotations

import re

# 유니코드 단어(\w+: 독일어 움라우트 포함) 또는 단어/공백이 아닌 문자 1개.
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# detokenize 규칙용 문자 집합.
# 앞 토큰에 붙는 문자(닫는 구두점/괄호/인용부호): "word ." -> "word."
_ATTACH_LEFT = frozenset(".,!?;:%)]}…”»。」")
# 뒤 토큰에 붙는 문자(여는 괄호/인용부호): "( word" -> "(word"
_ATTACH_RIGHT = frozenset("([{“«¿¡「")
# 양쪽에 붙는 어포스트로피(프랑스어 엘리전 l'homme, 영어 축약 don't).
_APOSTROPHES = frozenset("'’")


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


def detokenize(text: str) -> str:
    """simple_tokenize의 역연산: 분리된 구두점을 인접 단어에 다시 붙인다.

    모델 출력은 학습 코퍼스와 같은 토큰화 형태라 구두점 앞에 공백이 남아
    있다("un homme fait du velo ."). 이 함수는 원문(raw) 참조와 같은 자연문
    ("un homme fait du velo.")으로 복원해, sacreBLEU/COMET/METEOR가 hyp와
    raw ref를 동일한 기준으로 비교하도록 한다 (그렇지 않으면 tokenized hyp
    vs raw ref의 비대칭 비교가 되어 sacreBLEU가 경고를 낸다).

    규칙 (Moses detokenizer를 최소한으로 흉내낸 것; 대소문자는 건드리지 않음):
      - 닫는 구두점/괄호(. , ! ? ; : ) ] } % ...)는 앞 토큰에 붙인다.
      - 여는 괄호/인용부호(( [ { « ...)는 뒤 토큰에 붙인다.
      - 어포스트로피(' ’)는 양쪽에 붙인다 ("l ' homme" -> "l'homme").
      - 곧은 큰따옴표(")는 여닫이를 번갈아 처리한다.
      - 하이픈(-)은 양쪽이 모두 단어(영숫자)일 때만 붙인다
        ("nord - est" -> "nord-est"; 독립적인 대시는 그대로 둔다).

    Args:
        text: 공백으로 구분된 토큰 문자열 (보통 remove_bpe를 거친 상태).

    Returns:
        구두점이 재부착된 자연문 문자열.
    """
    tokens = text.split()
    if not tokens:
        return ""

    pieces: list[str] = [tokens[0]]
    # 다음 토큰을 앞 토큰에 붙일지 여부 (여는 괄호/여는 따옴표가 설정).
    attach_next = tokens[0] == '"' or tokens[0] in _ATTACH_RIGHT
    dquote_open = tokens[0] == '"'

    for i in range(1, len(tokens)):
        token = tokens[i]

        if token in _APOSTROPHES:
            attach_here, next_attaches = True, True
        elif token == '"':
            # 곧은 큰따옴표는 여닫이가 없으므로 등장 순서로 토글한다.
            attach_here, next_attaches = dquote_open, not dquote_open
            dquote_open = not dquote_open
        elif token == "-":
            # 양쪽이 모두 단어일 때만 합성어 하이픈으로 보고 붙인다.
            prev_word = tokens[i - 1][-1:].isalnum()
            next_word = i + 1 < len(tokens) and tokens[i + 1][:1].isalnum()
            attach_here = next_attaches = prev_word and next_word
        else:
            attach_here = token in _ATTACH_LEFT
            next_attaches = token in _ATTACH_RIGHT

        pieces.append(token if (attach_next or attach_here) else " " + token)
        attach_next = next_attaches

    return "".join(pieces)
