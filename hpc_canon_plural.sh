#!/usr/bin/env bash
# canon_batch_driver.sh
# Purpose: Pull directories and launch canonicalisation jobs on HPC.
#-----------------------------------------------------------------------------
set -euo pipefail

# Determine directory of this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") -e ENV_NAME [-c CONFIG]

  -e ENV_NAME   (required) Environment name exported to the jobs.
  -c CONFIG     (optional) Config string for run_canonicalisation_hpc.sh (default: soft_inf)
EOF
  exit 1
}

# Parse CLI args
env_name=""
config="soft_inf"
while getopts ":e:c:h" opt; do
  case "${opt}" in
    e) env_name="${OPTARG}" ;;
    c) config="${OPTARG}" ;;
    h|*) usage ;;
  esac
done
shift $((OPTIND-1))
[[ -z "${env_name}" ]] && { echo "Error: -e ENV_NAME is required" >&2; usage; }

# Ensure helper scripts exist
pull_script="${script_dir}/pull_dirs_canon_local.sh"
canon_script="${script_dir}/run_canonicalisation_hpc.sh"
for f in "${pull_script}" "${canon_script}"; do
  [[ -x "${f}" ]] || { echo "[ERROR] Required script '$f' not found or not executable." >&2; exit 4; }
done

# Run pull_dirs4canon locally and wait for it to complete
source $pull_script $env_name

# Check output
out_file="${script_dir}/tmp_out_pull_dirs4canon.out"
[[ -s "${out_file}" ]] || { echo "[ERROR] Directory list '${out_file}' missing/empty" >&2; exit 3; }

# Random suffix
if command -v shuf >/dev/null 2>&1; then suffix=$(shuf -i 0-10000 -n 1); else suffix=$((RANDOM%10001)); fi
echo "[INFO] Using suffix=${suffix}"

# Submit canonicalisation jobs
while IFS= read -r model_dir; do
  [[ -z "${model_dir}" ]] && continue
  vars="env_name=${env_name},model_dir=${model_dir},config=${config},suffix=${suffix}"
  job_id=$(qsub -v "${vars}" "${canon_script}" | awk '{print $1}')
  echo "[INFO] Submitted ${model_dir} → job ${job_id}"
done < "${out_file}"

echo "[INFO] All jobs queued."
