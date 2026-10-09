"""Qwen2.5-VL streaming backend subclass that drives InfoKVStream.

This is a **new branch**: it subclasses the existing ``EvalQwen25VLStream`` and
only overrides ``stream_video_chunk`` / ``reset`` so every streamed chunk is
funnelled through :class:`~evaluation_pipeline.memory.qwen25vl_gpu.InfoKVGPUIntegrator`.
The HERMES and Full backends are never touched.

Imported lazily (GPU path only) so the CPU test environment stays torch-free.
"""

from __future__ import annotations

import functools


def _resolve_layers(method_config, num_layers: int):
    from .qwen25vl_gpu import default_layer_choices

    defaults = default_layer_choices(num_layers)
    scoring = method_config.scoring
    layer_a = scoring.layer_a if scoring.layer_a is not None else defaults["layer_a"]
    layer_b = scoring.layer_b if scoring.layer_b is not None else defaults["layer_b"]
    v2v_layer = (
        scoring.v2v_layer if scoring.v2v_layer is not None else defaults["v2v_layer"]
    )
    for name, value in (
        ("layer_a", layer_a),
        ("layer_b", layer_b),
        ("v2v.layer", v2v_layer),
    ):
        if not 0 <= int(value) < num_layers:
            raise ValueError(
                f"{name}={value} out of range for {num_layers} layers"
            )
    return int(layer_a), int(layer_b), int(v2v_layer)


def build_infokv_wrapper_class():
    """Import the OVO-Bench backend and return the InfoKV subclass."""

    from models.Qwen25VL_Stream import EvalQwen25VLStream

    class EvalQwen25VLStreamInfoKV(EvalQwen25VLStream):
        """Streaming backend that additionally maintains InfoKVStream memory."""

        infokv_enabled = True

        def __init__(self, args):
            super().__init__(args)
            self._integrator = None

        # -- attach the memory method + capture + compactor ------------ #
        def attach_infokv(self, method_config):
            from .factory import build_memory_method
            from .qwen25vl_gpu import InfoKVGPUIntegrator

            memory = build_memory_method(method_config=method_config)
            layer_a, layer_b, v2v_layer = _resolve_layers(
                method_config, self.qa_model.num_layers
            )
            self._integrator = InfoKVGPUIntegrator(
                self,
                memory,
                layer_a=layer_a,
                layer_b=layer_b,
                v2v_layer=v2v_layer,
            )
            print(
                "[infokv] layers:",
                f"D=({layer_a},{layer_b})",
                f"V2V={v2v_layer}",
                f"prefix={self.qa_model.visual_start_idx}",
                f"budget={method_config.total_visual_budget}",
                f"recent={method_config.recent.budget}",
                f"long_term={method_config.long_term.budget}",
                f"interval={method_config.compression.interval_chunks}",
            )
            return self._integrator

        # -- streaming funnel ----------------------------------------- #
        @functools.wraps(EvalQwen25VLStream.stream_video_chunk)
        def stream_video_chunk(self, frames, *, seconds=None):
            integrator = self.__dict__.get("_integrator")
            if integrator is None:
                return super().stream_video_chunk(frames, seconds=seconds)
            meta, _ = integrator.observe_chunk(
                frames,
                seconds=seconds,
                encode_fn=super().stream_video_chunk,
            )
            integrator.maybe_compact()
            return meta

        def reset(self):
            super().reset()
            integrator = self.__dict__.get("_integrator")
            if integrator is not None:
                integrator.reset()

        # -- reporting ------------------------------------------------- #
        def infokv_events(self):
            integrator = self.__dict__.get("_integrator")
            return [] if integrator is None else list(integrator.events)

        def infokv_sidecar(self):
            integrator = self.__dict__.get("_integrator")
            return [] if integrator is None else list(integrator.last_sidecar)

        def infokv_stats(self):
            integrator = self.__dict__.get("_integrator")
            if integrator is None:
                return {}
            return integrator.memory.get_stats().as_dict()

    EvalQwen25VLStreamInfoKV.__name__ = "EvalQwen25VLStreamInfoKV"
    return EvalQwen25VLStreamInfoKV
