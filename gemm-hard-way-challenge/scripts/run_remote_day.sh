#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run correctness + benchmark for one or more GEMM hard-way challenge days on a Kubernetes GPU pod.

By default this runs the known-good solution kernels and compares each selected
kernel against the torch baseline through max_error and TFLOPS columns.

Usage:
  scripts/run_remote_day.sh --day DAY [options]
  scripts/run_remote_day.sh --all [options]

Options:
  --day DAY           Day number: 1..14.
  --all               Run all available day groups.
  --namespace NS      Kubernetes namespace. Default: $GEMM_NS or kk-flyte-adhoc
  --pod POD           Pod name. Default: $GEMM_POD or fcfd13ebf3b654511bbb-n0-0
  --remote-dir DIR    Remote directory. Default: $GEMM_REMOTE_DIR or /tmp/gemm-hard-way-challenge
  --container NAME    Optional container name for multi-container pods.
  --sizes "S..."      Space-separated matrix sizes. Default depends on dtype/day.
  --warmup N          Warmup iterations. Default: 5
  --iters N           Benchmark iterations. Default: 20
  --include-cutlass   Include CUTLASS days 13-14. Requires uploaded CUTLASS headers.
  --profile           Also run Nsight Compute on the selected kernel/day.
  --profile-set SET   Nsight Compute section set. Default: roofline
  -h, --help          Show this help.

Examples:
  scripts/run_remote_day.sh --day 1
  scripts/run_remote_day.sh --day 9 --sizes "128 256" --iters 50
  scripts/run_remote_day.sh --day 14 --include-cutlass
  scripts/run_remote_day.sh --day 6 --profile
EOF
}

namespace="${GEMM_NS:-kk-flyte-adhoc}"
pod="${GEMM_POD:-fcfd13ebf3b654511bbb-n0-0}"
remote_dir="${GEMM_REMOTE_DIR:-/tmp/gemm-hard-way-challenge}"
container=""
sizes=""
warmup=5
iters=20
include_cutlass=0
profile=0
profile_set="roofline"
all=0
declare -a days=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --day)
      days+=("$2")
      shift 2
      ;;
    --all)
      all=1
      shift
      ;;
    --namespace)
      namespace="$2"
      shift 2
      ;;
    --pod)
      pod="$2"
      shift 2
      ;;
    --remote-dir)
      remote_dir="$2"
      shift 2
      ;;
    --container)
      container="$2"
      shift 2
      ;;
    --sizes)
      sizes="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    --iters)
      iters="$2"
      shift 2
      ;;
    --include-cutlass)
      include_cutlass=1
      shift
      ;;
    --profile)
      profile=1
      shift
      ;;
    --profile-set)
      profile_set="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$all" -eq 1 ]]; then
  days=(1 2 3 4 5 6 7 8 9 10 11 12)
  if [[ "$include_cutlass" -eq 1 ]]; then
    days+=(13 14)
  fi
fi

if [[ "${#days[@]}" -eq 0 ]]; then
  echo "Pass --day DAY or --all" >&2
  usage >&2
  exit 2
fi

kubectl_args=(-n "$namespace")
if [[ -n "$container" ]]; then
  kubectl_args+=(-c "$container")
fi

kernel_for_day() {
  case "$1" in
    1) echo "float32:01_naive" ;;
    2) echo "float32:02_coalesced" ;;
    3) echo "float32:03_shared_mem" ;;
    4) echo "float32:04_blocktiling_1d" ;;
    5) echo "float32:05_blocktiling_2d" ;;
    6) echo "float32:06_vectorize" ;;
    7) echo "float32:07_warptiling" ;;
    8) echo "float16:08_warptiling_fp16" ;;
    9) echo "float16:09_tensorcore_naive_fp16" ;;
    10) echo "float16:10_tensorcore_fp16" ;;
    11) echo "float16:11_tensorcore_db_fp16" ;;
    12) echo "float16:12_tensorcore_async_fp16" ;;
    13) echo "float32:13_cutlass_fp32" ;;
    14) echo "float16:14_cutlass_autotune_fp16_cfg0" ;;
    *)
      echo "Unsupported day: $1" >&2
      return 1
      ;;
  esac
}

default_sizes_for_dtype() {
  case "$1" in
    float32) echo "128 256" ;;
    float16|bfloat16) echo "128 256" ;;
    *) echo "128" ;;
  esac
}

kubectl "${kubectl_args[@]}" exec "$pod" -- \
  bash -lc "test -d '$remote_dir' && test -f '$remote_dir/solutions/python/benchmark_solution.py'"

for day in "${days[@]}"; do
  spec="$(kernel_for_day "$day")"
  dtype="${spec%%:*}"
  kernel="${spec#*:}"
  run_sizes="${sizes:-$(default_sizes_for_dtype "$dtype")}"

  cutlass_arg=""
  if [[ "$day" -ge 13 ]]; then
    if [[ "$include_cutlass" -ne 1 ]]; then
      echo "Day $day requires --include-cutlass and uploaded CUTLASS headers. Skipping." >&2
      continue
    fi
    cutlass_arg="--include-cutlass"
  elif [[ "$include_cutlass" -eq 1 ]]; then
    cutlass_arg="--include-cutlass"
  fi

  echo
  echo "=== Day $day: dtype=$dtype kernel=$kernel sizes=[$run_sizes] ==="
  kubectl "${kubectl_args[@]}" exec "$pod" -- bash -lc "
    set -euo pipefail
    cd '$remote_dir'
    python3 solutions/python/benchmark_solution.py \
      $cutlass_arg \
      --sizes $run_sizes \
      --dtype $dtype \
      --kernels torch \
      --kernels $kernel \
      --warmup $warmup \
      --iters $iters
  "

  if [[ "$profile" -eq 1 ]]; then
    safe_sizes="${run_sizes// /_}"
    report="/tmp/gemm_day${day}_${kernel}_${safe_sizes}"
    echo "Profiling Day $day -> $report.ncu-rep"
    kubectl "${kubectl_args[@]}" exec "$pod" -- bash -lc "
      set -euo pipefail
      cd '$remote_dir'
      ncu --set '$profile_set' \
          --target-processes all \
          --force-overwrite \
          -o '$report' \
          python3 solutions/python/benchmark_solution.py \
            $cutlass_arg \
            --sizes ${run_sizes%% *} \
            --dtype $dtype \
            --kernels $kernel \
            --warmup 1 \
            --iters 1
    "
    echo "Copy profile with:"
    echo "  kubectl -n '$namespace' cp '$pod:$report.ncu-rep' ./$(basename "$report").ncu-rep"
  fi
done

