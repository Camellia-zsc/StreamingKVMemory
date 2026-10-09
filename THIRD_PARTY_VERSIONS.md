# Third-party provenance

This repository vendors local snapshots of two upstream projects. Both
snapshots contain **local modifications** used by the actual Qwen2.5-VL +
HERMES experiments; they are **not** pristine copies of the upstream release
and must not be described or treated as such.

## HERMES

- Upstream: https://github.com/haowei-freesky/HERMES.git
- Base commit: `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789`

The `HERMES/` tree here is a snapshot of that upstream repository **with local
modifications** used by the real Qwen2.5-VL + HERMES experiments. Do not assume
it is equal to the pristine upstream version at the base commit.

## OVO-Bench

- Upstream: https://github.com/JoeLeelyf/OVO-Bench.git
- Base commit: `c34093f9a072d93f9e1c6b923446dfba71f4528e`

The `OVO-Bench/` tree here is a snapshot of that upstream repository **with
local adapter/evaluation modifications**. Do not assume it is equal to the
pristine upstream version at the base commit.
