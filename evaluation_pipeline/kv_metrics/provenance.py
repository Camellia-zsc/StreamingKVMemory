"""KV token provenance / UID tracking.

Every KV position that HERMES manages (the long-term visual KV region) gets an
immutable ``kv_uid``. The tracker mirrors the *shape* of ``qa_model.kv_cache``
(one UID list per layer, in cache position order) so that after every
compaction the ``kept`` tensor and the ``kept`` UID list stay index-aligned.

Design notes
------------
* HERMES builds ``keep_indices`` **per layer** and shares them across all KV
  heads (``prune_kv_cache_by_attention`` averages over heads). Therefore the
  UID store is per layer, not per head. Per-head statistics are produced later
  by expanding the per-layer mask through the GQA map.
* Long-term layers pool the pruned tokens into one *summary* token appended at
  the end of the layer's cache. That summary token has no single original
  position, so it carries ``kind="summary"`` and remembers its parents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# UID kinds
KIND_SYSTEM = "system"
KIND_VISUAL = "visual"
KIND_TEXT = "text"
KIND_SUMMARY = "summary"


@dataclass(frozen=True)
class KVTokenUID:
    """Immutable provenance record for a single cached KV position."""

    uid: str
    layer_id: int
    kind: str
    video_id: Optional[str] = None
    frame_index: Optional[int] = None
    frame_timestamp_sec: Optional[float] = None
    visual_token_index: Optional[int] = None
    turn_id: Optional[int] = None
    token_index: Optional[int] = None
    token_type: Optional[str] = None
    parents: tuple = ()

    def is_visual_identity(self) -> bool:
        # A token that maps back to exactly one original visual position.
        return self.kind == KIND_VISUAL

    def to_dict(self) -> dict:
        data = {
            "uid": self.uid,
            "layer_id": self.layer_id,
            "kind": self.kind,
        }
        if self.video_id is not None:
            data["video_id"] = self.video_id
        if self.frame_index is not None:
            data["frame_index"] = self.frame_index
        if self.frame_timestamp_sec is not None:
            data["frame_timestamp_sec"] = float(self.frame_timestamp_sec)
        if self.visual_token_index is not None:
            data["visual_token_index"] = self.visual_token_index
        if self.turn_id is not None:
            data["turn_id"] = self.turn_id
        if self.token_index is not None:
            data["token_index"] = self.token_index
        if self.token_type is not None:
            data["token_type"] = self.token_type
        if self.parents:
            data["parents"] = list(self.parents)
        return data


class KVProvenanceTracker:
    """Keeps a per-layer UID list that mirrors ``qa_model.kv_cache``."""

    def __init__(
        self,
        video_id: Optional[str] = None,
        summary_token_mode: str = "identity_only",
    ):
        self.video_id = video_id
        self.summary_token_mode = summary_token_mode
        # layer_idx -> list[KVTokenUID]
        self.per_layer: List[List[KVTokenUID]] = []
        self._visual_counter = 0
        self._uid_counter = 0
        # Per-kind counters. UIDs for original positions (system/visual/text)
        # must be *stable across passes*, i.e. independent of how many
        # compactions/summary tokens happened in between. Therefore each kind
        # keeps its own monotonic counter.
        self._kind_counters = {"sys": 0, "vis": 0, "txt": 0, "sum": 0}
        # bookkeeping
        self.compaction_index = 0
        self.kept_masks_per_layer: List[List[bool]] = []
        self.summary_parents_per_layer: List[List[str]] = []

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.per_layer = []
        self._visual_counter = 0
        self._uid_counter = 0
        self._kind_counters = {"sys": 0, "vis": 0, "txt": 0, "sum": 0}
        self.compaction_index = 0
        self.kept_masks_per_layer = []
        self.summary_parents_per_layer = []

    def _new_uid(self, prefix: str) -> str:
        if prefix not in self._kind_counters:
            self._kind_counters[prefix] = 0
        index = self._kind_counters[prefix]
        self._kind_counters[prefix] = index + 1
        return f"{prefix}:{index}"

    def _ensure_layers(self, num_layers: int, prefix_len: int) -> None:
        if self.per_layer:
            return
        self.per_layer = [[] for _ in range(num_layers)]
        for layer_idx in range(num_layers):
            for pos in range(prefix_len):
                self.per_layer[layer_idx].append(
                    KVTokenUID(
                        uid=self._new_uid("sys"),
                        layer_id=layer_idx,
                        kind=KIND_SYSTEM,
                        video_id=self.video_id,
                        token_type="text",
                    )
                )

    def register_init(self, num_layers: int, prefix_len: int) -> None:
        """Register the (shared) system/user prefix tokens as text."""

        self.per_layer = []
        self._visual_counter = 0
        self._uid_counter = 0
        self._kind_counters = {"sys": 0, "vis": 0, "txt": 0, "sum": 0}
        self._ensure_layers(num_layers, prefix_len)

    def register_visual_chunk(
        self,
        per_frame_token_counts: Sequence[int],
        frame_indices: Sequence[int],
        frame_timestamps: Sequence[float],
    ) -> None:
        """Append visual tokens for a chunk of frames to every layer."""

        total = int(sum(per_frame_token_counts))
        if total == 0:
            return
        num_layers = len(self.per_layer)
        # Pre-build the visual UIDs once; they are identical across layers
        # except for the layer_id field.
        per_layer_entries: List[List[KVTokenUID]] = [[] for _ in range(num_layers)]
        visual_index = 0
        for frame_idx, count, ts in zip(
            frame_indices, per_frame_token_counts, frame_timestamps
        ):
            for _ in range(int(count)):
                uid = self._new_uid("vis")
                for layer_idx in range(num_layers):
                    per_layer_entries[layer_idx].append(
                        KVTokenUID(
                            uid=uid,
                            layer_id=layer_idx,
                            kind=KIND_VISUAL,
                            video_id=self.video_id,
                            frame_index=int(frame_idx),
                            frame_timestamp_sec=float(ts),
                            visual_token_index=visual_index,
                            token_type="visual",
                        )
                    )
                visual_index += 1
        self._visual_counter += visual_index
        for layer_idx in range(num_layers):
            self.per_layer[layer_idx].extend(per_layer_entries[layer_idx])

    def register_text_tokens(self, count: int, turn_id: Optional[int] = None) -> None:
        """Append transient text tokens (query prefill). Kept in sync so the
        assertion ``len(uids) == seq_len`` still holds during the anchor."""

        if count <= 0:
            return
        for layer_idx in range(len(self.per_layer)):
            for tok in range(count):
                self.per_layer[layer_idx].append(
                    KVTokenUID(
                        uid=self._new_uid("txt"),
                        layer_id=layer_idx,
                        kind=KIND_TEXT,
                        video_id=self.video_id,
                        turn_id=turn_id,
                        token_index=tok,
                        token_type="text",
                    )
                )

    def truncate_to(self, per_layer_length: Sequence[int]) -> None:
        """Drop trailing UIDs so lengths match the cache after truncation."""

        for layer_idx, length in enumerate(per_layer_length):
            if layer_idx >= len(self.per_layer):
                break
            if length < len(self.per_layer[layer_idx]):
                del self.per_layer[layer_idx][length:]

    # ------------------------------------------------------------------ #
    # compaction
    # ------------------------------------------------------------------ #
    def apply_keep_indices(
        self,
        keep_indices_per_layer: Sequence[Sequence[int]],
        long_term_flags: Sequence[bool],
    ) -> None:
        """Mirror ``_shrink_positions_and_rerotate_keys`` on the UID store.

        ``long_term_flags[layer]`` must be True for layers where HERMES pools
        the pruned tokens into a single appended summary token.
        """

        self.compaction_index += 1
        new_per_layer: List[List[KVTokenUID]] = []
        kept_masks: List[List[bool]] = []
        summary_parents: List[List[str]] = []

        for layer_idx, uids in enumerate(self.per_layer):
            keep = keep_indices_per_layer[layer_idx]
            keep_list = [int(i) for i in keep]
            keep_set = set(keep_list)
            seq_len = len(uids)

            mask = [pos in keep_set for pos in range(seq_len)]
            kept = [uids[pos] for pos in keep_list if 0 <= pos < seq_len]

            pruned_uids: List[str] = []
            if layer_idx < len(long_term_flags) and long_term_flags[layer_idx]:
                pruned_uids = [
                    uids[pos].uid
                    for pos in range(seq_len)
                    if pos not in keep_set
                ]
                if pruned_uids:
                    kept = kept + [
                        KVTokenUID(
                            uid=self._new_uid("sum"),
                            layer_id=layer_idx,
                            kind=KIND_SUMMARY,
                            video_id=self.video_id,
                            token_type="summary",
                            parents=tuple(pruned_uids),
                        )
                    ]

            new_per_layer.append(kept)
            kept_masks.append(mask)
            summary_parents.append(pruned_uids)

        self.per_layer = new_per_layer
        self.kept_masks_per_layer = kept_masks
        self.summary_parents_per_layer = summary_parents

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def lengths(self) -> List[int]:
        return [len(uids) for uids in self.per_layer]

    def layer_uids(self, layer_idx: int) -> List[KVTokenUID]:
        return self.per_layer[layer_idx]

    def assert_sync(self, cache) -> None:
        """Assert ``len(uids) == kv sequence length`` for every layer."""

        if cache is None:
            return
        for layer_idx, entry in enumerate(cache):
            k = entry[0]
            seq_len = int(k.shape[2])
            if layer_idx >= len(self.per_layer):
                raise AssertionError(
                    f"provenance missing for layer {layer_idx} "
                    f"(cache seq={seq_len})"
                )
            have = len(self.per_layer[layer_idx])
            if have != seq_len:
                raise AssertionError(
                    f"provenance desync at layer {layer_idx}: "
                    f"uids={have} cache_seq={seq_len}"
                )

    def visual_region(self, layer_idx: int) -> "VisualRegion":
        """Return the visual slice [start, end) of a layer's UID list."""

        uids = self.per_layer[layer_idx]
        start = None
        end = None
        for pos, uid in enumerate(uids):
            if uid.kind in (KIND_VISUAL, KIND_SUMMARY):
                if start is None:
                    start = pos
                end = pos + 1
        if start is None:
            return VisualRegion(layer_idx=layer_idx, start=0, end=0, uids=[])
        return VisualRegion(
            layer_idx=layer_idx, start=start, end=end, uids=uids[start:end]
        )

    def retained_uid_set(self, layer_idx: int) -> set:
        """UIDs that are considered *retained* for RAM on this layer.

        ``identity_only`` (default): only original visual positions that
        survive verbatim count. Summary tokens and their parents do not.
        ``absorbed``: parents folded into a summary token also count.
        """

        retained = set()
        for uid in self.per_layer[layer_idx]:
            if uid.kind == KIND_VISUAL:
                retained.add(uid.uid)
            elif uid.kind == KIND_SUMMARY and self.summary_token_mode == "absorbed":
                retained.update(uid.parents)
        return retained


@dataclass
class VisualRegion:
    layer_idx: int
    start: int
    end: int
    uids: List[KVTokenUID]

    @property
    def length(self) -> int:
        return self.end - self.start

    def timestamps(self) -> List[Optional[float]]:
        return [uid.frame_timestamp_sec for uid in self.uids]

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_idx,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }
