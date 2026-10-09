import unittest
from pathlib import Path

from evaluation_pipeline.adapters import (
    OVOBenchAdapter,
    STREAMING_TASK_CODES,
    StreamingBenchAdapter,
    VideoMMEAdapter,
)
from evaluation_pipeline.evaluate import load_config, make_run_id


ROOT = Path(__file__).resolve().parents[1]


class EvaluationPipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(ROOT / "configs/eval_all.yaml"))

    def test_streamingbench_has_18_task_codes(self):
        adapter = StreamingBenchAdapter(self.config["benchmarks"]["streamingbench"])
        samples = list(adapter.load_samples())
        codes = {
            question.metadata["task_code"]
            for sample in samples
            for question in sample.questions
        }
        self.assertEqual(len(codes), 18)
        self.assertEqual(len(codes), len(set(codes)))
        self.assertIn("SQA", codes)
        self.assertIn("PO", codes)

    def test_streamingbench_video_paths_use_raw_task_directories(self):
        adapter = StreamingBenchAdapter(self.config["benchmarks"]["streamingbench"])
        samples = list(adapter.load_samples())
        paths = {sample.video_path for sample in samples}
        self.assertTrue(paths)
        self.assertTrue(all("/raw/" in path for path in paths))
        self.assertTrue(
            any("/raw/real/sample_" in path for path in paths)
        )
        self.assertTrue(
            any("/raw/sqa/sample_" in path for path in paths)
        )
        self.assertTrue(
            any("/raw/proactive/sample_" in path for path in paths)
        )
        self.assertTrue(
            any("/raw/omni/emotion/sample_" in path for path in paths)
        )
        self.assertTrue(
            any("/raw/omni/source/sample_" in path for path in paths)
        )
        self.assertTrue(all(Path(path).is_file() for path in paths))

    def test_streamingbench_protocol_policies(self):
        adapter = StreamingBenchAdapter(self.config["benchmarks"]["streamingbench"])
        samples = list(adapter.load_samples())
        questions = [q for sample in samples for q in sample.questions]
        self.assertTrue(
            all(
                q.response_policy == "standing_trigger"
                for q in questions
                if q.task_type in {"Proactive Output", "Active Output"}
            )
        )
        self.assertTrue(
            all(
                q.response_policy == "reply_required"
                for q in questions
                if q.task_type not in {"Proactive Output", "Active Output"}
            )
        )

    def test_videomme_uses_only_first_five_archives(self):
        adapter = VideoMMEAdapter(self.config["benchmarks"]["videomme"])
        samples = list(adapter.load_samples())
        self.assertEqual(len(samples), 73)
        self.assertEqual(sum(len(sample.questions) for sample in samples), 219)

    def test_run_id_separates_memory_backends_and_budgets(self):
        config = load_config(str(ROOT / "configs/eval_all.yaml"))
        ids = []
        for method, budget in (
            ("full_cache", None),
            ("hermes", 6000),
            ("hermes", 3000),
        ):
            config["memory"]["method"] = method
            config["memory"]["compression"]["budget"] = budget
            ids.append(
                make_run_id(
                    config,
                    "videomme",
                    config["benchmarks"]["videomme"],
                )[0]
            )
        self.assertEqual(len(set(ids)), 3)

    def test_ovo_forward_policy_is_defer(self):
        adapter = OVOBenchAdapter(self.config["benchmarks"]["ovobench"])
        questions = [
            q
            for sample in adapter.load_samples()
            for q in sample.questions
            if q.task_type in {"REC", "SSR", "CRR"}
        ]
        self.assertTrue(questions)
        self.assertTrue(all(q.response_policy == "may_defer" for q in questions))

    def test_proactive_official_timing_and_answer_scoring(self):
        score = StreamingBenchAdapter.score_proactive(98, 100, "output 20", "20")
        self.assertTrue(score["time_correct"])
        self.assertTrue(score["answer_correct"])

        self.assertTrue(
            StreamingBenchAdapter.score_proactive(102, 100, "20", "20")[
                "time_correct"
            ]
        )
        self.assertFalse(
            StreamingBenchAdapter.score_proactive(103, 100, "20", "20")[
                "time_correct"
            ]
        )
        self.assertFalse(
            StreamingBenchAdapter.score_proactive(None, 100, "20", "20")[
                "time_correct"
            ]
        )
        self.assertFalse(
            StreamingBenchAdapter.score_proactive(100, 100, "wrong", "20")[
                "answer_correct"
            ]
        )

    def test_shared_video_sampler_rounding(self):
        from evaluation_pipeline.video_sampling import sample_video_frames
        import cv2
        import numpy as np

        path = "/tmp/evaluation_pipeline_sampler_test.mp4"
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (16, 16),
        )
        for index in range(100):
            writer.write(np.full((16, 16, 3), index, dtype=np.uint8))
        writer.release()
        sampled = sample_video_frames(path, 4)
        self.assertEqual(sampled["frame_indices"], [0, 33, 66, 99])
        self.assertEqual(len(sampled["frames"]), 4)

    def test_reply_required_mcq_uses_option_count(self):
        from evaluation_pipeline.runtime import ModelRuntime

        class Tokenizer:
            def __call__(self, text, add_special_tokens=False):
                token = text.strip()
                return type("Encoded", (), {"input_ids": [ord(token[0])]})()

        runtime = ModelRuntime({"memory": {"method": "full_cache"}})
        runtime.model = type(
            "Model",
            (),
            {"client": type("Client", (), {"processor": type("P", (), {"tokenizer": Tokenizer()})()})()},
        )()
        ids = runtime._allowed_mcq_token_ids(
            {"response_policy": "reply_required", "options": ["a", "b", "c", "d"]}
        )
        self.assertEqual({chr(value) for value in ids}, {"A", "B", "C", "D"})
        self.assertIsNone(
            runtime._allowed_mcq_token_ids(
                {"response_policy": "may_defer", "options": ["a", "b"]}
            )
        )

    def test_efficiency_includes_completed_error_predictions(self):
        from evaluation_pipeline.evaluate import efficiency_metrics

        metrics = efficiency_metrics(
            [
                {
                    "status": "error",
                    "runtime": {
                        "latency": 2.0,
                        "peak_gpu_memory_mb": 100.0,
                        "peak_kv_tokens": 20,
                        "compression_events": 0,
                        "total_removed_tokens": 0,
                        "mean_kv_retention_ratio": None,
                        "mean_kv_compression_ratio": None,
                    },
                },
                {"status": "failed", "runtime": {"latency": 99.0}},
            ]
        )
        self.assertEqual(metrics["mean_latency"], 2.0)
        self.assertEqual(metrics["peak_gpu_memory_mb"], 100.0)


if __name__ == "__main__":
    unittest.main()
