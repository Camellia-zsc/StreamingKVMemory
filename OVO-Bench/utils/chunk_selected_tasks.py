import argparse
import json
import math
import os

from moviepy.editor import VideoFileClip
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Chunk videos for selected OVO-Bench tasks"
)
parser.add_argument(
    "--anno_path",
    default="data/ovo_bench_new.json",
)
parser.add_argument(
    "--video_dir",
    default="data/src_videos",
)
parser.add_argument(
    "--output_dir",
    default="data/chunked_videos_selected",
)
parser.add_argument(
    "--tasks",
    nargs="+",
    required=True,
)

args = parser.parse_args()

selected_tasks = set(args.tasks)
os.makedirs(args.output_dir, exist_ok=True)

with open(args.anno_path, "r", encoding="utf-8") as f:
    annotations = json.load(f)

annotations = [
    item
    for item in annotations
    if item["task"] in selected_tasks
]

print("选择的任务：", sorted(selected_tasks))
print("需要生成的切片数：", len(annotations))

success_count = 0
skip_count = 0
error_count = 0

for item in tqdm(annotations):
    video_id = item["id"]
    source_path = os.path.join(
        args.video_dir,
        item["video"],
    )
    output_path = os.path.join(
        args.output_dir,
        f"{video_id}.mp4",
    )

    # 已经成功生成过就跳过，支持断点续跑
    if (
        os.path.isfile(output_path)
        and os.path.getsize(output_path) > 0
    ):
        skip_count += 1
        continue

    if not os.path.isfile(source_path):
        print(f"\n源视频不存在：{source_path}")
        error_count += 1
        continue

    video = None
    clip = None

    try:
        end_time = math.ceil(float(item["realtime"]))

        video = VideoFileClip(source_path)
        end_time = min(end_time, video.duration)

        clip = video.subclip(0, end_time)
        clip.write_videofile(
            output_path,
            logger=None,
        )

        success_count += 1

    except Exception as exc:
        error_count += 1
        print(
            f"\n处理失败：id={video_id}, "
            f"task={item['task']}, "
            f"error={exc}"
        )

        # 删除未完成的损坏文件
        if os.path.isfile(output_path):
            os.remove(output_path)

    finally:
        if clip is not None:
            clip.close()
        if video is not None:
            video.close()

print("\n切片任务结束")
print("新生成：", success_count)
print("已存在并跳过：", skip_count)
print("失败：", error_count)
print("输出目录：", args.output_dir)
