#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
simulation_dir="$(cd "${script_dir}/.." && pwd)"
indiflight_root="${INDIFLIGHT_ROOT:-/home/quant/research/Indiflight/CustomIndiflight/indiflight}"
pi_protocol_root="${PI_PROTOCOL_ROOT:-/home/quant/research/Indiflight/pi-protocol}"
local_mk="${indiflight_root}/make/local.mk"
config_mk="${simulation_dir}/config/mockupConfig.mk"
support_venv="${simulation_dir}/../.venv"
mockup_debug="${MOCKUP_DEBUG:-}"
mockup_force_rebuild="${MOCKUP_FORCE_REBUILD:-1}"

if [[ ! -d "${indiflight_root}" ]]; then
  echo "Missing custom Indiflight checkout: ${indiflight_root}" >&2
  exit 1
fi

if [[ ! -d "${pi_protocol_root}" ]]; then
  echo "Missing pi-protocol checkout: ${pi_protocol_root}" >&2
  exit 1
fi

mkdir -p "${indiflight_root}/lib/main"
if [[ -L "${indiflight_root}/lib/main/pi-protocol" ]]; then
  current_target="$(readlink "${indiflight_root}/lib/main/pi-protocol")"
  if [[ "${current_target}" != "${pi_protocol_root}" ]]; then
    echo "Existing pi-protocol symlink points to ${current_target}, expected ${pi_protocol_root}" >&2
    exit 1
  fi
elif [[ -e "${indiflight_root}/lib/main/pi-protocol" ]]; then
  echo "Using existing ${indiflight_root}/lib/main/pi-protocol" >&2
else
  ln -s "${pi_protocol_root}" "${indiflight_root}/lib/main/pi-protocol"
fi

cp "${config_mk}" "${local_mk}"

if [[ -x "${support_venv}/bin/python3" ]]; then
  export PATH="${support_venv}/bin:${PATH}"
fi

make_args=(TARGET=MOCKUP DEBUG="${mockup_debug}")
if [[ "${mockup_force_rebuild}" != "0" ]]; then
  make_args+=(-B)
fi

if [[ -n "${mockup_debug}" ]]; then
  echo "Building MOCKUP with DEBUG=${mockup_debug}"
else
  echo "Building MOCKUP release build"
fi

make -C "${indiflight_root}" "${make_args[@]}"

so_path="${indiflight_root}/obj/main/indiflight_MOCKUP.so"
if [[ ! -f "${so_path}" ]]; then
  echo "Build completed but ${so_path} was not created." >&2
  exit 1
fi

echo "${so_path}"
