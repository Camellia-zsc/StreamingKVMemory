import abc
import json
import os
import sys

from tqdm import tqdm

sys.path.append("..")

from constant import (
    BR_PROMPT_TEMPLATE,
    REC_PROMPT_TEMPLATE,
    SSR_PROMPT_TEMPLATE,
    CRR_PROMPT_TEMPLATE,
)


class OVOBenchOnline:
    def __init__(self) -> None:
        pass

    def inference(self):
        pass


class OVOBenchOffline:
    def __init__(self, args):
        self.args = args

    def _call_inference(
        self,
        video_path,
        prompt,
        realtime,
        task,
        sample_id,
    ):
        """
        只有支持时间戳推理的模型才接收额外参数。
        其他模型保持原来的两个参数调用。
        """
        if getattr(
            self,
            "supports_timestamp_inference",
            False,
        ):
            return self.inference(
                video_path,
                prompt,
                start_time=0.0,
                end_time=float(realtime),
                task=task,
                sample_id=sample_id,
            )

        return self.inference(
            video_path,
            prompt,
        )

    def _call_forward_inference(
        self,
        video_path,
        prompt,
        realtime,
        task,
        sample_id,
        test_index,
    ):
        """
        只有支持 Forward 流式推理的模型才调用
        inference_forward()。

        其他模型继续使用原来的 inference()。
        """
        if getattr(
            self,
            "supports_forward_inference",
            False,
        ):
            return self.inference_forward(
                video_file_name=video_path,
                prompt=prompt,
                realtime=float(realtime),
                task=task,
                sample_id=sample_id,
                test_index=test_index,
            )

        return self.inference(
            video_path,
            prompt,
        )

    def eval(
        self,
        anno,
        task_list,
        mode="offline",
    ):
        # ====================================================
        # Backward Tasks
        # ====================================================
        if len(anno["backward"]) > 0:
            backward_results = []

            for item in tqdm(
                anno["backward"],
                desc="Backward Tasks",
            ):
                sample_id = item["id"]
                video = item["video"]
                task = item["task"]
                question = item["question"]
                options = item["options"]
                realtime = item["realtime"]

                assert question is not None
                assert options is not None

                prompt = self.build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                )

                chunk_video_path = os.path.join(
                    self.args.chunked_dir,
                    f"{sample_id}.mp4",
                )

                assert os.path.exists(
                    chunk_video_path
                ), chunk_video_path

                try:
                    response = self._call_inference(
                        video_path=chunk_video_path,
                        prompt=prompt,
                        realtime=realtime,
                        task=task,
                        sample_id=sample_id,
                    )
                except Exception as error:
                    print(
                        "Error during inference: "
                        f"id={sample_id}, "
                        f"task={task}, "
                        f"error={error}"
                    )
                    response = None

                result = {
                    "id": sample_id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
                    "ground_truth": chr(
                        65 + item["gt"]
                    ),
                }

                if getattr(
                    self,
                    "supports_timestamp_inference",
                    False,
                ):
                    result["query_time"] = float(
                        realtime
                    )
                    result["answer_video_time"] = (
                        float(realtime)
                        if response is not None
                        else None
                    )

                backward_results.append(
                    result
                )

        # ====================================================
        # Realtime Tasks
        # ====================================================
        if len(anno["realtime"]) > 0:
            realtime_results = []

            for item in tqdm(
                anno["realtime"],
                desc="Realtime Tasks",
            ):
                sample_id = item["id"]
                video = item["video"]
                task = item["task"]
                question = item["question"]
                options = item["options"]
                realtime = item["realtime"]

                assert question is not None
                assert options is not None

                prompt = self.build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                )

                chunk_video_path = os.path.join(
                    self.args.chunked_dir,
                    f"{sample_id}.mp4",
                )

                assert os.path.exists(
                    chunk_video_path
                ), chunk_video_path

                try:
                    response = self._call_inference(
                        video_path=chunk_video_path,
                        prompt=prompt,
                        realtime=realtime,
                        task=task,
                        sample_id=sample_id,
                    )
                except Exception as error:
                    print(
                        "Error during inference: "
                        f"id={sample_id}, "
                        f"task={task}, "
                        f"error={error}"
                    )
                    response = None

                result = {
                    "id": sample_id,
                    "video": video,
                    "task": task,
                    "question": question,
                    "response": response,
                    "ground_truth": chr(
                        65 + item["gt"]
                    ),
                }

                if getattr(
                    self,
                    "supports_timestamp_inference",
                    False,
                ):
                    result["query_time"] = float(
                        realtime
                    )
                    result["answer_video_time"] = (
                        float(realtime)
                        if response is not None
                        else None
                    )

                realtime_results.append(
                    result
                )

        # ====================================================
        # Forward Tasks
        # ====================================================
        if len(anno["forward"]) > 0:
            forward_results = []

            for item in tqdm(
                anno["forward"],
                desc="Forward Tasks",
            ):
                sample_id = item["id"]
                task = item["task"]
                test_info = item["test_info"]

                for index in range(
                    len(test_info)
                ):
                    prompt = self.build_prompt(
                        task=task,
                        question=None,
                        options=None,
                        _anno_=item,
                        index=index,
                    )

                    realtime = test_info[
                        index
                    ]["realtime"]

                    chunk_video_path = os.path.join(
                        self.args.chunked_dir,
                        f"{sample_id}_{index}.mp4",
                    )

                    assert os.path.exists(
                        chunk_video_path
                    ), chunk_video_path

                    try:
                        response = (
                            self._call_forward_inference(
                                video_path=(
                                    chunk_video_path
                                ),
                                prompt=prompt,
                                realtime=realtime,
                                task=task,
                                sample_id=sample_id,
                                test_index=index,
                            )
                        )
                    except Exception as error:
                        print(
                            "Error during inference: "
                            f"id={sample_id}, "
                            f"task={task}, "
                            f"index={index}, "
                            f"error={error}"
                        )
                        response = None

                    item["test_info"][index][
                        "response"
                    ] = response

                forward_results.append(
                    item
                )

        if len(anno["backward"]) == 0:
            backward_results = []

        if len(anno["realtime"]) == 0:
            realtime_results = []

        if len(anno["forward"]) == 0:
            forward_results = []

        # ====================================================
        # 保存结果
        # ====================================================
        if self.args.save_results:
            output_dir = os.path.join(
                self.args.result_dir,
                self.args.model,
            )

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            output_name = (
                f"{self.args.model}_"
                f"{'_'.join(task_list)}_"
                f"{mode}_1.json"
            )

            output_path = os.path.join(
                output_dir,
                output_name,
            )

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "backward": backward_results,
                        "realtime": realtime_results,
                        "forward": forward_results,
                    },
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

    def build_prompt(
        self,
        task,
        question,
        options,
        _anno_,
        index,
    ):
        if task in [
            "EPM",
            "ASI",
            "HLD",
            "STU",
            "OJR",
            "ATR",
            "ACR",
            "OCR",
            "FPD",
        ]:
            formatted_options = (
                "; ".join(
                    f"{chr(65 + i)}. {option}"
                    for i, option
                    in enumerate(options)
                )
                + ";"
            )

            prompt = BR_PROMPT_TEMPLATE.format(
                question,
                formatted_options,
            )

        elif task == "REC":
            activity = _anno_["activity"]

            question = (
                "How many times did they "
                + activity
                + "?"
            )

            prompt = REC_PROMPT_TEMPLATE.format(
                question
            )

        elif task == "SSR":
            step = _anno_[
                "test_info"
            ][index]["step"]

            prompt = SSR_PROMPT_TEMPLATE.format(
                step
            )

        elif task == "CRR":
            question = _anno_["question"]

            prompt = CRR_PROMPT_TEMPLATE.format(
                question
            )

        else:
            raise ValueError(
                f"Unsupported task: {task}"
            )

        return prompt

    @abc.abstractmethod
    def inference(
        self,
        video_file_name,
        prompt,
        start_time=0,
        end_time=0,
    ):
        pass