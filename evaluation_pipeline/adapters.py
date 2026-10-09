import ast
import csv
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .schema import CanonicalSample, Question


LOG = logging.getLogger("evaluate_all")

# Official OVO-Bench task grouping (single source of truth, reused by the
# kv_metrics analysis layer instead of hard-coding task names there).
OVO_TASK_GROUPS = {
    "backward": ["EPM", "ASI", "HLD"],
    "realtime": ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"],
    "forward": ["REC", "SSR", "CRR"],
}

STREAMING_TASK_CODES = {
    "Object Perception": "OP",
    "Action Perception": "AP",
    "Text-Rich Understanding": "TRU",
    "Clips Summarize": "CS",
    "Attribute Perception": "ATP",
    "Spatial Understanding": "SPU",
    "Counting": "CNT",
    "Event Understanding": "EU",
    "Causal Reasoning": "CR",
    "Prospective Reasoning": "PR",
    "Emotion Recognition": "ER",
    "Multimodal Alignment": "MA",
    "Scene Understanding": "SCU",
    "Source Discrimination": "SD",
    "Misleading Context Recognition": "MCR",
    "Anomaly Context Understanding": "ACU",
    "Sequential Question Answering": "SQA",
    "Proactive Output": "PO",
    "Active Output": "PO",
}


def parse_options(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [item.strip() for item in str(value).split("|") if item.strip()]


def parse_timestamp(value) -> float:
    if value is None or value == "":
        return 0.0
    parts = str(value).split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(value)


def option_letter(value: str) -> str:
    match = re.search(r"\b([A-Z])\b", str(value).upper())
    return match.group(1) if match else str(value).strip().upper()


class BenchmarkAdapter(ABC):
    name = ""
    mode = "offline"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._selected_question_ids = set()
        self._last_samples = []

    @abstractmethod
    def load_samples(self) -> Iterable[CanonicalSample]:
        raise NotImplementedError

    @abstractmethod
    def run_sample(self, runtime, sample: CanonicalSample) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError

    def preflight(self, samples: List[CanonicalSample]) -> Dict[str, Any]:
        """Validate the local input set before a model is loaded."""
        paths = {}
        for sample in samples:
            paths.setdefault(sample.video_path, []).append(sample.sample_id)
        missing = sorted(path for path in paths if not Path(path).is_file())
        question_count = sum(len(sample.questions) for sample in samples)
        return {
            "samples": len(samples),
            "questions": question_count,
            "videos": len(paths),
            "missing_videos": missing,
        }


class OVOBenchAdapter(BenchmarkAdapter):
    name = "ovobench"
    mode = "streaming"

    def load_samples(self):
        path = Path(self.config["annotation"])
        data = json.loads(path.read_text(encoding="utf-8"))
        root = Path(self.config.get("chunked_dir", "OVO-Bench/data/chunked_videos"))
        allowed_video_ids = self.config.get("_video_ids")
        if allowed_video_ids is not None:
            allowed_video_ids = {str(v) for v in allowed_video_ids}
        for item in data:
            if allowed_video_ids is not None and str(item["id"]) not in allowed_video_ids:
                continue
            task = item["task"]
            questions = []
            if task in {"REC", "SSR", "CRR"}:
                for index, test in enumerate(item.get("test_info", [])):
                    questions.append(
                        Question(
                            question_id=f"{item['id']}:{index}",
                            text="",
                            timestamp=float(test["realtime"]),
                            task_type=task,
                            answer=test.get("count", test.get("type")),
                            response_policy="may_defer",
                            metadata={"test_index": index, "test_info": test},
                        )
                    )
            else:
                questions.append(
                    Question(
                        question_id=str(item["id"]),
                        text=item["question"],
                        options=item.get("options", []),
                        timestamp=float(item.get("realtime", 0.0)),
                        task_type=task,
                        answer=chr(65 + int(item["gt"])),
                        response_policy="reply_required",
                    )
                )
            first_video = (
                f"{item['id']}_{questions[0].metadata['test_index']}.mp4"
                if task in {"REC", "SSR", "CRR"}
                else f"{item['id']}.mp4"
            )
            yield CanonicalSample(
                benchmark=self.name,
                sample_id=str(item["id"]),
                video_path=str(root / first_video),
                questions=questions,
                mode="streaming",
                metadata=item,
            )

    def run_sample(self, runtime, sample):
        source = sample.metadata
        results = []
        # Experimental OVO Backward/Realtime MCQ switches. Defaults (True/True)
        # are the official behavior; see configs/eval_ovo_full.yaml / the
        # top-level `ovo:` config section.
        ovo_must_reply = bool(self.config.get("must_reply", True))
        ovo_constrain_mcq = bool(
            self.config.get("constrain_mcq_first_token", True)
        )
        for question in sample.questions:
            if question.question_id in self.config.get("_skip_question_ids", set()):
                continue
            task = question.task_type
            video_path = sample.video_path
            is_ovo_mcq = question.response_policy == "reply_required"
            if task in {"REC", "SSR", "CRR"}:
                test = question.metadata["test_info"]
                video_path = str(
                    Path(self.config.get("chunked_dir", "OVO-Bench/data/chunked_videos"))
                    / f"{sample.sample_id}_{question.metadata['test_index']}.mp4"
                )
                prompt = runtime.model.build_prompt(
                    task=task, question=None, options=None, _anno_=source,
                    index=question.metadata["test_index"],
                )
            else:
                prompt = runtime.model.build_prompt(
                    task=task, question=question.text, options=question.options,
                    _anno_=None, index=None,
                )
            run_question = {
                "task": task,
                "timestamp": question.timestamp,
                "sample_id": sample.sample_id,
                "test_index": question.metadata.get("test_index"),
                "options": question.options,
                "response_policy": question.response_policy,
            }
            if is_ovo_mcq:
                # Only OVO Backward/Realtime MCQ consumes these switches.
                run_question["must_reply"] = ovo_must_reply
                run_question["constrain_mcq_first_token"] = ovo_constrain_mcq
            raw = runtime.run_ovo(
                video_path,
                prompt,
                run_question,
            )
            record = {
                "benchmark": self.name,
                "sample_id": sample.sample_id,
                "question_id": question.question_id,
                "prediction": raw,
                "raw_output": raw,
                "ground_truth": question.answer,
                "task_type": task,
                "video_path": video_path,
                "question_timestamp": question.timestamp,
                "response_policy": question.response_policy,
                "protocol": {
                    "context_mode": None,
                    "proactive_protocol": None,
                },
                "metadata": question.metadata,
            }
            if is_ovo_mcq:
                # Report the decode policy that was ACTUALLY used. The
                # Qwen2.5-VL backend owns its own MCQ mode (official by
                # default = free decode + option extraction).
                if getattr(runtime, "backend", "mmduet2") == "qwen2_5_vl":
                    record["ovo_must_reply"] = False
                    record["ovo_mcq_constrained"] = (
                        getattr(runtime, "mcq_decode_mode", "official") == "constrained"
                    )
                else:
                    record["ovo_must_reply"] = ovo_must_reply
                    record["ovo_mcq_constrained"] = ovo_constrain_mcq
                record["mcq_decode_mode"] = getattr(
                    runtime, "mcq_decode_mode", "official"
                )
            results.append(record)
            if (
                question.response_policy == "reply_required"
                and runtime._is_no_reply(raw)
            ):
                results[-1]["status"] = "error"
                results[-1]["error_type"] = "unexpected_no_reply"
        return results

    def evaluate(self, predictions):
        by_task = {}
        for item in predictions:
            if item.get("status") == "failed":
                continue
            task = item["task_type"]
            response = str(item.get("prediction", ""))
            gt = str(item.get("ground_truth", ""))
            if task in {"REC", "SSR", "CRR"}:
                if task == "REC":
                    correct = "".join(re.findall(r"\d+", response)) == gt
                else:
                    expected = "No" if str(gt) == "0" else "Yes"
                    correct = option_letter(response) == option_letter(expected)
            else:
                correct = gt in response
            by_task.setdefault(task, []).append(int(correct))
        metrics = {task: sum(values) / len(values) for task, values in by_task.items()}
        for group, tasks in OVO_TASK_GROUPS.items():
            values = [metrics[t] for t in tasks if t in metrics]
            if values:
                metrics[f"{group}_average"] = sum(values) / len(values)
        # Official OVO-Bench Total Avg: unweighted mean of the three category
        # averages, i.e. (backward_average + realtime_average + forward_average)
        # / 3 when all three groups are present. This is deliberately NOT the
        # mean over the 12 task accuracies, which double-weights realtime.
        group_averages = [
            metrics[f"{group}_average"]
            for group in OVO_TASK_GROUPS
            if f"{group}_average" in metrics
        ]
        overall = (
            sum(group_averages) / len(group_averages)
            if group_averages
            else None
        )
        return {"overall": overall, "tasks": metrics}


class StreamingBenchAdapter(BenchmarkAdapter):
    name = "streamingbench"
    mode = "streaming"

    @staticmethod
    def _category_name(category):
        name = category.lower()
        if name.startswith("real_time"):
            return "real_time_visual"
        if name.startswith("omni_source"):
            return "omni_source"
        if name.startswith("contextual") or name.startswith("sequential"):
            return "contextual"
        if name.startswith("proactive"):
            return "proactive"
        return name

    @staticmethod
    def _video_id(qid):
        parts = qid.rsplit("_", 2)
        if len(parts) >= 3 and parts[-2].isdigit():
            return parts[-2]
        return qid

    @staticmethod
    def _video_subdir(category, task_type):
        category_name = str(category).lower()
        task_name = str(task_type).lower()
        if category_name.startswith("real_time"):
            return Path("raw/real")
        if category_name.startswith("sequential"):
            return Path("raw/sqa")
        if category_name.startswith("proactive"):
            return Path("raw/proactive")

        omni_dirs = {
            "anomaly context understanding": "anomaly",
            "misleading context recognition": "misleading",
            "misleading context understanding": "misleading",
            "emotion recognition": "emotion",
            "multimodal alignment": "multimodal",
            "scene understanding": "scene",
            "source discrimination": "source",
        }
        for task_name_key, subdir in omni_dirs.items():
            if task_name == task_name_key:
                return Path("raw/omni") / subdir
        raise ValueError(
            f"Cannot map StreamingBench video directory: "
            f"category={category!r}, task_type={task_type!r}"
        )

    def load_samples(self):
        root = Path(self.config.get("video_root", "datasets/StreamingBench"))
        annotations = self.config.get("annotations", {})
        if self.config.get("input_modality", "visual_only") == "visual_only":
            LOG.warning(
                "StreamingBench Omni-Source tasks ER/SCU/SD/MA are evaluated "
                "with visual-only MMDuet2 input; audio is not provided."
            )
        for category, annotation in annotations.items():
            path = Path(annotation)
            is_sqa = category.lower().startswith("sequential")
            allowed_video_ids = self.config.get("_video_ids")
            if allowed_video_ids is not None:
                allowed_video_ids = {str(v) for v in allowed_video_ids}
            grouped = {}
            order = []
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    qid = row["question_id"]
                    video_id = self._video_id(qid)
                    if allowed_video_ids is not None and str(video_id) not in allowed_video_ids:
                        continue
                    task_type = row.get("task_type", category)
                    video_path = (
                        root
                        / self._video_subdir(category, task_type)
                        / f"sample_{video_id}"
                        / "video.mp4"
                    )
                    is_proactive = category.lower().startswith("proactive")
                    timestamp_key = "ground_truth_time_stamp" if is_proactive else "time_stamp"
                    policy = "standing_trigger" if is_proactive else "reply_required"
                    question = Question(
                        question_id=qid,
                        text=row["question"],
                        options=parse_options(row.get("options", [])),
                        timestamp=parse_timestamp(row.get("time_stamp")),
                        task_type=task_type,
                        answer=row.get("answer", row.get("ground_truth_output")),
                        response_policy=policy,
                        metadata={**row, "category": category, "video_id": video_id,
                                  "category_name": self._category_name(category),
                                  "task_code": STREAMING_TASK_CODES.get(
                                      row.get("task_type", category),
                                      row.get("task_type", category),
                                  ),
                                  "ground_truth_timestamp": parse_timestamp(row.get(timestamp_key)) if row.get(timestamp_key) else None},
                    )
                    if is_sqa:
                        if video_id not in grouped:
                            grouped[video_id] = []
                            order.append(video_id)
                        grouped[video_id].append((video_path, question))
                    else:
                        yield CanonicalSample(
                            benchmark=self.name,
                            sample_id=qid,
                            video_path=str(video_path),
                            questions=[question],
                            mode="streaming",
                            metadata={
                                "category": category,
                                "category_name": self._category_name(category),
                                "context_mode": self.config.get("context_mode", "main_60s"),
                                "proactive_protocol": self.config.get(
                                    "proactive_protocol", "official_polling"
                                ),
                                "input_modality": self.config.get(
                                    "input_modality", "visual_only"
                                ),
                            },
                        )
            if is_sqa:
                for video_id in order:
                    values = sorted(grouped[video_id], key=lambda item: item[1].timestamp or 0.0)
                    yield CanonicalSample(
                        benchmark=self.name,
                        sample_id=f"sqa:{video_id}",
                        video_path=str(values[0][0]),
                        questions=[item[1] for item in values],
                        mode="streaming",
                        metadata={
                            "category": category,
                            "category_name": self._category_name(category),
                            "context_mode": self.config.get("context_mode", "main_60s"),
                            "proactive_protocol": self.config.get(
                                "proactive_protocol", "official_polling"
                            ),
                            "input_modality": self.config.get(
                                "input_modality", "visual_only"
                            ),
                        },
                    )

    def run_sample(self, runtime, sample):
        results = []
        history = []
        is_sqa = sample.metadata.get("category", "").lower().startswith("sequential")
        is_proactive = sample.metadata.get("category", "").lower().startswith("proactive")
        for q in sample.questions:
            only = self.config.get("_only_question_ids")
            if only is not None and q.question_id not in set(only):
                continue
            if q.question_id in self.config.get("_skip_question_ids", set()):
                if is_sqa and q.answer is not None:
                    history.append((q, q.answer))
                continue
            context_mode = sample.metadata.get("context_mode", "main_60s")
            context_time = (
                float(self.config.get("context_time", 60))
                if context_mode == "main_60s"
                else 0.0
            )
            query_time = float(q.timestamp or 0.0)
            # Official PO starts with the annotated prefix timestamp and
            # polls from timestamp + 1. It is not a 60-second contextual QA.
            # ``full_prefix`` = [0, query_time] (KV-compaction experiment only;
            # the official ``main_60s`` default is unchanged).
            start_time = (
                query_time
                if is_proactive
                else (
                    0.0
                    if context_mode == "full_prefix"
                    else max(0.0, query_time - context_time)
                )
            )
            options_text = "\n".join(q.options)
            history_text = ""
            if is_sqa:
                history_text = (
                    "Here are the contextual information related to the video. "
                    "Please answer based on it:\n"
                    + "\n".join(
                        f"At timestamp {old.timestamp_text if hasattr(old, 'timestamp_text') else old.metadata.get('time_stamp')}, "
                        f"Question: {old.text}; Answer: {answer};"
                        for old, answer in history
                    )
                    + "\n"
                )
            # ``prompt_style`` selects the MCQ prompt sent to the model.
            #   pipeline  -> legacy MMDuet2-era text (default, unchanged)
            #   official  -> benchmark question + options with the upstream
            #                "Please select the correct answer." instruction
            # The Qwen2.5-VL config opts into ``official``; legacy configs keep
            # ``pipeline`` so no prior result changes.
            prompt_style = self.config.get("prompt_style", "pipeline")
            instruction = (
                "Please select the correct answer."
                if prompt_style == "official"
                else "Answer with the option letter only."
            )
            prompt_source = (
                "official_benchmark"
                if prompt_style == "official"
                else "pipeline_mmduet2_mcq"
            )
            prompt = (
                history_text
                + f"Question: {q.text}\nOptions:\n{options_text}\n"
                + instruction
            )
            protocol = {
                "context_mode": context_mode,
                "context_start": start_time,
                "context_end": q.timestamp,
                "proactive_protocol": sample.metadata.get("proactive_protocol"),
                "input_modality": sample.metadata.get("input_modality", "visual_only"),
                "prompt_style": prompt_style,
                "prompt_source": prompt_source,
            }
            if is_proactive:
                if sample.metadata.get("proactive_protocol", "official_polling") == "mmduet_native":
                    record = runtime.run_proactive_native(
                        sample.video_path, q.text, q, start_time, sample.sample_id
                    )
                else:
                    record = runtime.run_proactive_polling(
                        sample.video_path, q.text, q, start_time, sample.sample_id
                    )
                raw = record["prediction"]
            else:
                raw = runtime.run_streaming_mcq(
                    sample.video_path,
                    prompt,
                    {
                        "timestamp": q.timestamp,
                        "start_time": start_time,
                        "sample_id": sample.sample_id,
                        "options": q.options,
                        "response_policy": q.response_policy,
                        "task": q.task_type,
                        "prompt_source": prompt_source,
                    },
                )
                record = {
                    "prediction": raw,
                    "predicted_timestamp": q.timestamp,
                    "ground_truth_timestamp": None,
                }
                if q.response_policy == "reply_required" and runtime._is_no_reply(raw):
                    record["status"] = "error"
                    record["error_type"] = "unexpected_no_reply"
            if is_proactive and (
                record.get("predicted_timestamp") is None
                or runtime._is_no_reply(raw)
            ):
                record["status"] = "error"
                record["error_type"] = "missed_proactive_trigger"
            result = {
                "benchmark": self.name,
                "sample_id": sample.sample_id,
                "question_id": q.question_id,
                "prediction": raw,
                "raw_output": raw,
                "ground_truth": q.answer or q.metadata.get("ground_truth_output"),
                "task_type": q.task_type,
                "task_code": q.metadata.get("task_code", q.task_type),
                "video_path": sample.video_path,
                "question_timestamp": q.timestamp,
                "predicted_timestamp": record.get("predicted_timestamp"),
                "ground_truth_timestamp": q.metadata.get("ground_truth_timestamp"),
                "time_difference": (
                    record.get("predicted_timestamp") - q.metadata.get("ground_truth_timestamp")
                    if record.get("predicted_timestamp") is not None
                    and q.metadata.get("ground_truth_timestamp") is not None
                    else None
                ),
                "response_policy": q.response_policy,
                "protocol": protocol,
                "inference": {
                    "sample_fps": runtime.model.sample_fps,
                    "frames_per_turn": runtime.model.frames_per_turn,
                    "max_side": runtime.model.max_side,
                },
                "metadata": q.metadata,
            }
            if record.get("status"):
                result["status"] = record["status"]
                result["error_type"] = record.get("error_type")
            results.append(result)
            if is_sqa:
                history.append((q, q.answer))
        return results

    @staticmethod
    def score_proactive(predicted_timestamp, ground_truth_timestamp, prediction, ground_truth):
        """Match the official scorer's separate timing and answer checks."""
        if predicted_timestamp is None or ground_truth_timestamp is None:
            return {
                "time_difference": None,
                "time_correct": False,
                "answer_correct": False,
            }
        time_difference = float(predicted_timestamp) - float(ground_truth_timestamp)
        time_correct = abs(time_difference) <= 2
        answer_correct = (
            time_correct
            and str(ground_truth or "").strip().lower()
            in str(prediction or "").strip().lower()
        )
        return {
            "time_difference": time_difference,
            "time_correct": time_correct,
            "answer_correct": answer_correct,
        }

    def evaluate(self, predictions):
        categories = {}
        tasks = {}
        timing = []
        proactive_total = 0
        proactive_time_correct = 0
        proactive_answer_correct = 0
        proactive_differences = []
        for item in predictions:
            if item.get("status") == "failed":
                continue
            meta = item.get("metadata", {})
            category = meta.get("category_name", meta.get("category", "unknown"))
            task = item.get("task_type", "unknown")
            task_code = item.get("task_code", task)
            prediction = str(item.get("prediction", "")).strip()
            ground_truth = str(item.get("ground_truth", "")).strip()
            is_proactive = task_code == "PO" or category.lower().startswith("proactive")
            if is_proactive:
                proactive_total += 1
                proactive_score = self.score_proactive(
                    item.get("predicted_timestamp"),
                    item.get("ground_truth_timestamp"),
                    prediction,
                    ground_truth,
                )
                if proactive_score["time_difference"] is not None:
                    proactive_differences.append(proactive_score["time_difference"])
                    timing.append(proactive_score["time_difference"])
                proactive_time_correct += int(proactive_score["time_correct"])
                proactive_answer_correct += int(proactive_score["answer_correct"])
                # Keep the existing scalar task/category accuracy shape as
                # answer correctness; official timing remains separate below.
                correct = proactive_score["answer_correct"]
            else:
                correct = option_letter(prediction) == option_letter(ground_truth)
            categories.setdefault(category, []).append(int(correct))
            tasks.setdefault(task_code, []).append(int(correct))
            if not is_proactive and item.get("time_difference") is not None:
                timing.append(float(item["time_difference"]))
        values = {k: sum(v) / len(v) for k, v in categories.items()}
        task_values = {k: sum(v) / len(v) for k, v in tasks.items()}
        all_values = [x for v in categories.values() for x in v]
        proactive_metrics = {
            "total": proactive_total,
            "time_correct": proactive_time_correct,
            "time_accuracy": (
                proactive_time_correct / proactive_total
                if proactive_total else 0.0
            ),
            "answer_correct": proactive_answer_correct,
            "answer_accuracy": (
                proactive_answer_correct / proactive_total
                if proactive_total else 0.0
            ),
            "mean_time_difference": (
                sum(proactive_differences) / len(proactive_differences)
                if proactive_differences else None
            ),
            "answered": sum(
                item.get("predicted_timestamp") is not None
                for item in predictions
                if item.get("task_code") == "PO"
            ),
        }
        return {
            "overall": sum(all_values) / len(all_values) if all_values else None,
            "tasks": task_values,
            "categories": values,
            "timing": {
                "mean_time_difference": sum(timing) / len(timing) if timing else None,
                "answered": len(timing),
            },
            "proactive_output": proactive_metrics,
            "input_modality": "visual_only",
        }


class VideoMMEAdapter(BenchmarkAdapter):
    name = "videomme"
    mode = "offline"

    def load_samples(self):
        root = Path(self.config.get("video_root", "datasets/Video-MME/videos"))
        annotation = Path(self.config["annotation"])
        allowed_video_ids = self._load_allowed_video_ids()
        grouped = {}
        order = []
        with annotation.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                benchmark_video_id = str(item.get("video_id", "")).strip()
                if allowed_video_ids is not None and benchmark_video_id not in allowed_video_ids:
                    continue
                video_id = item.get("videoID", item.get("video_id"))
                if video_id not in grouped:
                    grouped[video_id] = []
                    order.append(video_id)
                question = Question(
                    question_id=str(item["question_id"]),
                    text=item["question"],
                    options=item.get("options", []),
                    task_type=item.get("task_type"),
                    answer=item.get("answer"),
                    metadata=item,
                )
                grouped[video_id].append(question)

        for video_id in order:
            questions = grouped[video_id]
            item = questions[0].metadata
            yield CanonicalSample(
                benchmark=self.name,
                sample_id=str(item.get("video_id", video_id)),
                video_path=str(root / f"{video_id}.mp4"),
                duration=None,
                questions=questions,
                mode="offline",
                metadata=item,
            )

    def _load_allowed_video_ids(self):
        map_path = self.config.get("archive_map")
        if not map_path:
            return None

        archive_map = json.loads(Path(map_path).read_text(encoding="utf-8"))
        archives = archive_map.get("archives", archive_map)
        requested = self.config.get("allowed_archives")
        if requested is None:
            requested = list(archives)

        allowed = set()
        for archive_name in requested:
            if archive_name not in archives:
                raise ValueError(
                    f"Archive {archive_name!r} is not present in {map_path}; "
                    f"available={sorted(archives)}"
                )
            for entry in archives[archive_name]:
                if isinstance(entry, dict):
                    allowed.add(str(entry["video_id"]).strip())
                else:
                    allowed.add(str(entry).strip())
        if not allowed:
            raise ValueError(f"No video IDs found in selected archive map: {map_path}")
        return allowed

    def run_sample(self, runtime, sample):
        results = []
        for q in sample.questions:
            if q.question_id in self.config.get("_skip_question_ids", set()):
                continue
            try:
                runtime.reset()
                prompt = q.text + "\n" + "\n".join(q.options) + "\nAnswer with the option letter only."
                raw = runtime.run_offline(sample.video_path, prompt, {})
                results.append({
                    "benchmark": self.name,
                    "sample_id": sample.sample_id,
                    "question_id": q.question_id,
                    "prediction": raw,
                    "raw_output": raw,
                    "ground_truth": q.answer,
                    "task_type": q.task_type,
                    "task_code": q.metadata.get("task_code", q.task_type),
                    "video_path": sample.video_path,
                    "duration": q.metadata.get("duration"),
                    "response_policy": q.response_policy,
                    "protocol": {
                        "context_mode": None,
                        "proactive_protocol": None,
                        "input_modality": "visual_only",
                    },
                    "metadata": q.metadata,
                    "inference": dict(runtime.last_inference_metadata),
                    "status": "ok",
                })
            except Exception as error:
                results.append({
                    "benchmark": self.name,
                    "sample_id": sample.sample_id,
                    "question_id": q.question_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "video_path": sample.video_path,
                    "metadata": q.metadata,
                })
        return results

    def evaluate(self, predictions):
        groups = {}
        values = []
        failed = 0
        for item in predictions:
            if item.get("status") == "failed":
                failed += 1
                continue
            pred = option_letter(item.get("prediction", ""))
            gt = option_letter(item.get("ground_truth", ""))
            correct = int(pred == gt)
            values.append(correct)
            groups.setdefault(item.get("duration", "unknown"), []).append(correct)
        result = {
            "overall": sum(values) / len(values) if values else None,
            **{key: sum(value) / len(value) for key, value in groups.items()},
        }
        expected = sum(len(sample.questions) for sample in self._last_samples)
        result["pipeline"] = {
            "videos": len(self._last_samples),
            "questions": expected,
            "answered": len(values),
            "failed": failed,
            "coverage": len(values) / expected if expected else None,
        }
        return result


BENCHMARK_REGISTRY = {
    "ovobench": OVOBenchAdapter,
    "streamingbench": StreamingBenchAdapter,
    "videomme": VideoMMEAdapter,
}
