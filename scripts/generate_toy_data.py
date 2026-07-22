"""
===============================================================================
 파일: scripts/generate_toy_data.py
 목적:
    어떤 데이터셋도 다운로드하지 않고 전체 파이프라인(토크나이저 ->
    학습 -> 평가 -> 빔 서치)이 종단간(end-to-end)으로 동작하도록 합성
    sequence-to-sequence 코퍼스를 생성한다.

 역할:
    datasets/dataset.py가 기대하는 TSV 형식("소스<TAB>타겟")으로
    data/train.tsv, data/valid.tsv, data/test.tsv를 작성한다. 난이도가
    점점 높아지는 세 가지 작업을 사용할 수 있다:
        copy    — 타겟이 소스와 동일
        reverse — 타겟은 소스를 뒤집은 것
        sort    — 타겟은 소스를 오름차순 정렬한 것 (기본값)

 입력 / 출력:
    CLI 입력, TSV 파일 출력. 예:
        python scripts/generate_toy_data.py --task sort --train-size 10000

 구현 세부사항:
    토큰은 단어("17")로 렌더링된 정수이므로, 단어 단위 토크나이저는 작은
    닫힌 어휘집을 만들어내고 몇 epoch만으로도 작업을 학습하기에 충분하다
    — 구현을 검증하기 위한 좋은 종단간 sanity check다.
===============================================================================
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# 프로젝트 루트에서 `python scripts/...`로 실행하면 프로젝트 루트가
# sys.path에 있다; 어차피 여기서는 프로젝트 import가 필요 없다.

TASKS = ("copy", "reverse", "sort")


def make_pair(rng: random.Random, task: str, min_len: int, max_len: int, max_value: int) -> str:
    """요청된 작업에 맞는 '소스<TAB>타겟' 한 줄을 만든다."""
    length = rng.randint(min_len, max_len)
    numbers = [rng.randint(0, max_value) for _ in range(length)]
    if task == "copy":
        target = list(numbers)
    elif task == "reverse":
        target = list(reversed(numbers))
    elif task == "sort":
        target = sorted(numbers)
    else:
        raise ValueError(f"Unknown task '{task}'")
    source_text = " ".join(str(n) for n in numbers)
    target_text = " ".join(str(n) for n in target)
    return f"{source_text}\t{target_text}"


def write_split(path: Path, size: int, rng: random.Random, args: argparse.Namespace) -> None:
    """`size`개의 예제로 이루어진 TSV 분할 하나를 작성한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for _ in range(size):
            fh.write(make_pair(rng, args.task, args.min_len, args.max_len, args.max_value) + "\n")
    print(f"wrote {size:6d} examples -> {path}")


def main() -> None:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="Generate a toy seq2seq corpus")
    parser.add_argument("--out-dir", default="data", help="output directory")
    parser.add_argument("--task", choices=TASKS, default="sort", help="target transformation")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--min-len", type=int, default=3, help="min tokens per sequence")
    parser.add_argument("--max-len", type=int, default=12, help="max tokens per sequence")
    parser.add_argument("--max-value", type=int, default=99, help="largest integer token")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    write_split(out / "train.tsv", args.train_size, rng, args)
    write_split(out / "valid.tsv", args.valid_size, rng, args)
    write_split(out / "test.tsv", args.test_size, rng, args)


if __name__ == "__main__":
    main()
