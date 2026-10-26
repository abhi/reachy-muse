#!/bin/sh
# Every configuration in bench/motion.md, one after another. MOTION holds models/{0.8b,4b} (bundle downloads)
# and mlx/{0.8b,4b}-{bf16,4bit} (mlx_lm.convert outputs); PY is a Python with the companion-motion extra.
set -eu
: "${MOTION:?set MOTION}" "${PY:?set PY}"
OUT=$(dirname "$0")/motion_runs
run() { name=$1; shift; echo "== $name"; "$PY" "$(dirname "$0")/motion.py" --out "$OUT/$name.json" "$@"; }
for size in 0.8b 4b; do
	run "hf-mps-$size" --bundle "$MOTION/models/$size" --planner hf --device mps --probes
	for q in bf16 4bit; do
		run "mlx-$q-$size" --bundle "$MOTION/models/$size" --planner mlx --mlx-model "$MOTION/mlx/$size-$q" --device mps --probes
	done
done
run hf-cpu-0.8b --bundle "$MOTION/models/0.8b" --planner hf --device cpu --runs 1 --limit 2
run mlx-4bit-0.8b-gencpu --bundle "$MOTION/models/0.8b" --planner mlx --mlx-model "$MOTION/mlx/0.8b-4bit" --device cpu
for size in 0.8b 4b; do
	run "mlx-4bit-$size-contended" --bundle "$MOTION/models/$size" --planner mlx --mlx-model "$MOTION/mlx/$size-4bit" --device mps --runs 1 \
		--contend mlx-community/Qwen3-4B-Instruct-2507-4bit
done
for size in 0.8b 4b; do
	for q in bf16 4bit; do
		run "mlx-$q-$size-formatted" --bundle "$MOTION/models/$size" --planner mlx --mlx-model "$MOTION/mlx/$size-$q" --device mps --runs 1 --formatted
	done
done
run mlx-8bit-4b --bundle "$MOTION/models/4b" --planner mlx --mlx-model "$MOTION/mlx/4b-8bit" --device mps --probes
run mlx-8bit-4b-formatted --bundle "$MOTION/models/4b" --planner mlx --mlx-model "$MOTION/mlx/4b-8bit" --device mps --runs 1 --formatted
