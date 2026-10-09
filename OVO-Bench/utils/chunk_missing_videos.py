import argparse
import json
import math
import os
from pathlib import Path

from moviepy.editor import VideoFileClip
from tqdm import tqdm


FORWARD_TASKS = {"REC", "SSR", "CRR"}


def is_completed(path: Path) -> bool:
    """文件存在且大小大于0时，视为已经完成。"""
    return path.is_file() and path.stat().st_size > 0


def build_jobs(annotations, video_dir: Path, output_dir: Path):
    """将标注展开成实际需要生成的所有切片任务。"""
    jobs = []

    for item in annotations:
        task = item["task"]
        source_path = video_dir / item["video"]

        if task in FORWARD_TASKS:
            for index, test_item in enumerate(item["test_info"]):
                output_path = output_dir / f"{item['id']}_{index}.mp4"

                jobs.append(
                    {
                        "id": item["id"],
                        "task": task,
                        "source_path": source_path,
                        "output_path": output_path,
                        "end_time": float(test_item["realtime"]),
                    }
                )
        else:
            output_path = output_dir / f"{item['id']}.mp4"

            jobs.append(
                {
                    "id": item["id"],
                    "task": task,
                    "source_path": source_path,
                    "output_path": output_path,
                    "end_time": float(item["realtime"]),
                }
            )

    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Generate only missing OVO-Bench video chunks."
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
        default="data/chunked_videos",
    )
    args = parser.parse_args()

    anno_path = Path(args.anno_path)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with anno_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    jobs = build_jobs(
        annotations=annotations,
        video_dir=video_dir,
        output_dir=output_dir,
    )

    completed_before = sum(
        is_completed(job["output_path"])
        for job in jobs
    )

    print("=" * 70)
    print("理论切片总数：", len(jobs))
    print("运行前已完成：", completed_before)
    print("本次需要补齐：", len(jobs) - completed_before)
    print("输出目录：", output_dir)
    print("=" * 70)

    generated = 0
    skipped = 0
    failed = 0
    failures = []

    for job in tqdm(jobs, desc="Chunking OVO videos"):
        source_path = job["source_path"]
        output_path = job["output_path"]

        if is_completed(output_path):
            skipped += 1
            continue

        if not source_path.is_file():
            failed += 1
            failures.append(
                {
                    "id": job["id"],
                    "task": job["task"],
                    "source": str(source_path),
                    "output": str(output_path),
                    "error": "source video does not exist",
                }
            )
            continue

        # 先输出到临时文件。
        # 只有完整编码结束后才替换成正式文件，避免中断后留下损坏文件。
        temp_path = output_path.with_name(
            output_path.stem + ".part.mp4"
        )

        if temp_path.exists():
            temp_path.unlink()

        video = None
        clip = None

        try:
            video = VideoFileClip(str(source_path))

            end_time = math.ceil(job["end_time"])
            end_time = min(float(end_time), float(video.duration))

            if end_time <= 0:
                raise ValueError(
                    f"invalid end time: {end_time}"
                )

            clip = video.subclip(0, end_time)

            clip.write_videofile(
                str(temp_path),
                logger=None,
            )

            os.replace(temp_path, output_path)
            generated += 1

        except Exception as exc:
            failed += 1

            failures.append(
                {
                    "id": job["id"],
                    "task": job["task"],
                    "source": str(source_path),
                    "output": str(output_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            if temp_path.exists():
                temp_path.unlink()

            print(
                f"\n失败：task={job['task']}, "
                f"id={job['id']}, "
                f"error={type(exc).__name__}: {exc}"
            )

        finally:
            if clip is not None:
                clip.close()

            if video is not None:
                video.close()

    failure_path = output_dir / "chunk_failures.json"

    with failure_path.open("w", encoding="utf-8") as file:
        json.dump(
            failures,
            file,
            ensure_ascii=False,
            indent=2,
        )

    completed_after = sum(
        is_completed(job["output_path"])
        for job in jobs
    )

    print("\n" + "=" * 70)
    print("切片任务结束")
    print("本次新生成：", generated)
    print("已有并跳过：", skipped)
    print("本次失败：", failed)
    print("当前完成数：", completed_after)
    print("理论总数：", len(jobs))
    print("失败记录：", failure_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
