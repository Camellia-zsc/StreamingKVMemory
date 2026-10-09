"""CPU-only debug CLI for the pluggable memory methods.

Example::

    python -m evaluation_pipeline.memory.debug \
        --method infokv_stream \
        --num_chunks 20 \
        --tokens_per_chunk 8 \
        --budget 64 \
        --recent_budget 16 \
        --compression_interval 4

Everything runs on CPU with synthetic data. No model, no CUDA, no weights.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np

from .factory import available_memory_methods, build_memory_method
from .schemas import (
    CompressionConfig,
    LongTermConfig,
    MemoryMethodConfig,
    Observation,
    RecentConfig,
)


def _build_scenario(
    scenario: str,
    num_chunks: int,
    tokens_per_chunk: int,
    dim: int,
    rng: np.random.Generator,
) -> List[Observation]:
    """Synthesize chunk observations for the three debug scenarios."""

    observations: List[Observation] = []
    base = rng.normal(size=dim).astype(np.float32)
    position = 0
    token_id = 0
    for chunk_id in range(num_chunks):
        if scenario == "static":
            g = base
        elif scenario == "event":
            if chunk_id == 5:
                g = base + 5.0 * rng.normal(size=dim).astype(np.float32)
            else:
                g = base + 0.01 * rng.normal(size=dim).astype(np.float32)
        else:  # "random"
            g = rng.normal(size=dim).astype(np.float32)

        token_embeddings = g[None, :] + 0.05 * rng.normal(
            size=(tokens_per_chunk, dim)
        ).astype(np.float32)
        shallow = rng.normal(size=(tokens_per_chunk, dim)).astype(np.float32)
        # Randomly decide, per token, how much the deep representation moves.
        move = rng.uniform(0.0, 1.2, size=(tokens_per_chunk, 1)).astype(np.float32)
        direction = rng.normal(size=(tokens_per_chunk, dim)).astype(np.float32)
        deep = shallow + move * direction
        attention = rng.uniform(0.0, 1.0, size=tokens_per_chunk).astype(np.float32)
        # Synthetic visual Q/K states so the V2V provider runs for real
        # (shape [n, H, D] with H=1 head for the CPU debug path).
        query_states = rng.normal(size=(tokens_per_chunk, 1, dim)).astype(np.float32)
        key_states = rng.normal(size=(tokens_per_chunk, 1, dim)).astype(np.float32)

        observations.append(
            Observation(
                chunk_id=chunk_id,
                chunk_embedding=g,
                token_ids=list(range(token_id, token_id + tokens_per_chunk)),
                token_embeddings=token_embeddings,
                shallow_hidden=shallow,
                deep_hidden=deep,
                attention_importance=attention,
                query_states=query_states,
                key_states=key_states,
                is_visual=[True] * tokens_per_chunk,
                original_positions=list(
                    range(position, position + tokens_per_chunk)
                ),
                frame_time=float(chunk_id),
            )
        )
        position += tokens_per_chunk
        token_id += tokens_per_chunk
    return observations


def _run(args) -> Dict:
    rng = np.random.default_rng(args.seed)
    if args.method == "infokv_stream":
        long_term_budget = (
            args.long_term_budget
            if args.long_term_budget is not None
            else max(0, int(args.budget) - int(args.recent_budget))
        )
        config = MemoryMethodConfig(
            method="infokv_stream",
            total_visual_budget=int(args.budget),
            recent=RecentConfig(enabled=True, budget=int(args.recent_budget)),
            long_term=LongTermConfig(budget=int(long_term_budget)),
            compression=CompressionConfig(
                interval_chunks=int(args.compression_interval)
            ),
        )
        config.validate()
        memory = build_memory_method(method_config=config)
    else:
        memory = build_memory_method(
            {"memory": {"method": args.method, "compression": {"budget": args.budget}}}
        )

    observations = _build_scenario(
        args.scenario, args.num_chunks, args.tokens_per_chunk, args.dim, rng
    )

    compaction_events: List[Dict] = []
    for observation in observations:
        memory.observe(observation)
        if memory.should_compact():
            result = memory.compact()
            compaction_events.append(
                {
                    "trigger_index": len(compaction_events) + 1,
                    "before": result.extra.get("before", result.tokens_before),
                    "after": result.extra.get("after", result.tokens_after),
                    "evicted": result.extra.get("evicted", 0),
                    "effective": result.extra.get("effective", False),
                    "recent": result.extra.get("recent_size", 0),
                    "long_term": result.extra.get("long_term_size", 0),
                    "retention_ratio": result.retention_ratio,
                }
            )

    stats = memory.get_stats()
    rows = memory.sidecar_rows()
    retained = [r for r in rows if r.get("selected")]
    retained_sorted = sorted(
        retained,
        key=lambda r: (-(r["final_score"] or 0.0), -r["chunk_id"]),
    )

    # Top retained chunks by summed final score.
    by_chunk: Dict[int, float] = {}
    for row in retained:
        by_chunk[row["chunk_id"]] = by_chunk.get(row["chunk_id"], 0.0) + float(
            row["final_score"] or 0.0
        )
    top_chunks = sorted(by_chunk.items(), key=lambda kv: -kv[1])[: args.top_chunks]

    return {
        "memory": memory,
        "stats": stats,
        "compaction_events": compaction_events,
        "retained": retained_sorted,
        "top_chunks": top_chunks,
    }


def _print_report(args, result: Dict) -> None:
    stats = result["stats"]
    stats_dict = stats.as_dict()
    print("=" * 68)
    print(f"Memory method CPU synthetic debug: {args.method}")
    print("=" * 68)
    print(f"method                    : {args.method}")
    print(f"scenario                  : {args.scenario}")
    print(f"num_chunks                : {args.num_chunks}")
    print(f"total_seen_tokens         : {stats_dict.get('total_seen_tokens', 0)}")
    print(f"recent_tokens             : {stats_dict['recent_size']}")
    print(f"long_term_tokens          : {stats_dict['long_term_size']}")
    print(f"pending_candidates        : {stats_dict.get('pending_candidates', 0)}")
    print(f"evicted_tokens            : {stats_dict['evicted_count']}")
    print(
        "compression_trigger_count : "
        f"{stats_dict['compression_trigger_count']}"
    )
    print(
        "effective_compaction_count: "
        f"{stats_dict['effective_compaction_count']}"
    )
    print(
        f"budget (total/recent/long): "
        f"{stats_dict['budget_total']}/"
        f"{stats_dict['budget_recent']}/"
        f"{stats_dict['budget_long_term']}"
    )
    print(
        f"peak_managed_tokens       : {stats_dict['peak_managed_tokens']}"
    )
    print(
        "peak_budget_overshoot     : "
        f"{stats_dict['peak_budget_overshoot_tokens']} "
        f"({stats_dict['peak_budget_overshoot_ratio']:.3f})"
        if stats_dict.get("peak_budget_overshoot_ratio") is not None
        else "peak_budget_overshoot     : 0"
    )

    print("\n--- per-compaction events (before/after = long-term pool size) ---")
    for event in result["compaction_events"]:
        retention = event["retention_ratio"]
        retention_text = "n/a" if retention is None else f"{retention:.3f}"
        print(
            f"trigger #{event['trigger_index']:>2}: "
            f"before={event['before']:>4} after={event['after']:>4} "
            f"evicted={event['evicted']:>3} "
            f"effective={str(event['effective']):>5} "
            f"recent={event['recent']:>3} long_term={event['long_term']:>3} "
            f"retention={retention_text}"
        )

    print("\n--- top retained chunks by summed final score ---")
    for chunk_id, score in result["top_chunks"]:
        print(f"chunk {chunk_id:>3}: score_sum={score:.4f}")

    print("\n--- retained tokens ---")
    print(
        f"{'token_id':>8} {'chunk':>6} {'N':>9} {'D':>10} "
        f"{'A_v2v':>8} {'final':>9}"
    )
    for row in result["retained"]:
        print(
            f"{row['token_id']:>8} {row['chunk_id']:>6} "
            f"{row['novelty_raw']:>9.4f} {row['evolution_raw']:>10.4f} "
            f"{row['v2v_raw']:>8.4f} "
            f"{(row['final_score'] or 0.0):>9.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation_pipeline.memory.debug",
        description="CPU-only synthetic debug for the memory methods.",
    )
    parser.add_argument(
        "--method",
        default="infokv_stream",
        choices=list(available_memory_methods()),
    )
    parser.add_argument("--num_chunks", type=int, default=20)
    parser.add_argument("--tokens_per_chunk", type=int, default=8)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--recent_budget", type=int, default=16)
    parser.add_argument(
        "--long_term_budget",
        type=int,
        default=None,
        help="Defaults to budget - recent_budget.",
    )
    parser.add_argument("--compression_interval", type=int, default=4)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        default="random",
        choices=["random", "static", "event"],
    )
    parser.add_argument("--top_chunks", type=int, default=5)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw result dict as JSON instead of the text report.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = _run(args)
    if args.json:
        payload = {
            "stats": result["stats"].as_dict(),
            "compaction_events": result["compaction_events"],
            "top_chunks": result["top_chunks"],
            "retained_tokens": result["retained"],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_report(args, result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
