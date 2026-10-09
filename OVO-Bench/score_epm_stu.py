import argparse
import json
from collections import defaultdict
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", required=True)
parser.add_argument("--model", default="MMDuet2")
args = parser.parse_args()

model_dir = Path(args.result_dir) / args.model

if not model_dir.is_dir():
    raise FileNotFoundError(
        f"结果目录不存在：{model_dir}"
    )

records = []

for result_file in sorted(model_dir.glob("*.json")):
    with result_file.open("r", encoding="utf-8") as f:
        result = json.load(f)

    records.extend(result.get("backward", []))
    records.extend(result.get("realtime", []))

scores = defaultdict(list)

for item in records:
    task = item["task"]

    if task not in {"EPM", "STU"}:
        continue

    response = item.get("response")
    ground_truth = item.get("ground_truth")

    if response is None:
        score = 0
    else:
        # 与 OVO-Bench 官方 backward/realtime 评分逻辑保持一致
        score = int(ground_truth in response)

    scores[task].append(score)

all_scores = []

for task in ("EPM", "STU"):
    values = scores.get(task, [])

    if not values:
        print(f"{task}: 没有结果")
        continue

    accuracy = 100.0 * sum(values) / len(values)
    all_scores.extend(values)

    print(
        f"{task}: "
        f"{sum(values)}/{len(values)}, "
        f"Acc={accuracy:.2f}"
    )

if all_scores:
    total_accuracy = (
        100.0 * sum(all_scores) / len(all_scores)
    )
    print(f"总体样本准确率：{total_accuracy:.2f}")
else:
    print("没有找到可评分的 EPM/STU 结果。")
