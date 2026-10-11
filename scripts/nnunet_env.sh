# Directory env for nnU-Net and this repo's work tree.
# Keeps values that are already set. Fills the rest with repo defaults.
#
#   source scripts/nnunet_env.sh

if [ -n "${BASH_SOURCE-}" ]; then
  _nnunet_env_here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  _nnunet_env_here="$(cd "$(dirname "$0")/.." && pwd)"
fi

if [ "${BASH_SOURCE-}" = "$0" ]; then
  echo "source this file so the exports stay in your shell: source scripts/nnunet_env.sh" >&2
  exit 1
fi

: "${nnUNet_raw:=$_nnunet_env_here/raw}"
: "${nnUNet_preprocessed:=$_nnunet_env_here/preprocessed}"
: "${nnUNet_results:=$_nnunet_env_here/results}"
: "${WORK_PREFIX:=$_nnunet_env_here/data/work}"

export nnUNet_raw nnUNet_preprocessed nnUNet_results WORK_PREFIX
unset _nnunet_env_here
