#!/usr/bin/env bash
# check_system.sh — Safe system info collector for fresh Linux servers
# Prints OS, kernel, CPU, memory, disk, GPU/driver, CUDA, Python, and network info.
# No installs or changes.

set -u

section() {
  echo
  echo "============================================================"
  echo "== $1"
  echo "============================================================"
}

cmd() {
  local desc="$1"; shift
  echo "- ${desc}:"
  echo "  $*"
  if command -v "$1" >/dev/null 2>&1; then
    "$@"
  else
    echo "  (command not found)"
  fi
  echo
}

section "OS and Kernel"
if command -v lsb_release >/dev/null 2>&1; then
  cmd "lsb_release" lsb_release -a
else
  cmd "cat /etc/os-release" cat /etc/os-release
fi
cmd "uname" uname -a

section "CPU"
cmd "lscpu" lscpu
cmd "nproc" nproc

section "Memory"
cmd "free" free -h

section "Disk"
cmd "df" df -hT
cmd "lsblk" lsblk -o NAME,FSTYPE,SIZE,MOUNTPOINT | sed 's/^/  /'

section "GPU (NVIDIA)"
cmd "nvidia-smi" nvidia-smi
# PCI view of NVIDIA devices
if command -v lspci >/dev/null 2>&1; then
  echo "- lspci | grep -i nvidia:"
  echo "  lspci | grep -i nvidia"
  lspci | grep -i nvidia || true
  echo
fi

section "CUDA Toolkit"
# nvcc version
if command -v nvcc >/dev/null 2>&1; then
  cmd "nvcc --version" nvcc --version
else
  echo "- nvcc: not installed"
fi
# CUDA paths
{
  echo
  echo "- CUDA-related environment:"
  echo "  PATH=$PATH"
  echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"
  echo "  CUDA_HOME=${CUDA_HOME-}"
  echo
} | sed 's/^/  /'

section "Python"
cmd "python3 --version" python3 --version
cmd "which python3" which python3
cmd "pip3 --version" pip3 --version

section "Conda"
cmd "conda --version" conda --version
cmd "conda envs" conda info --envs

section "Network"
cmd "hostname -I" hostname -I
cmd "public ip (curl ifconfig.me)" curl -s ifconfig.me
cmd "DNS (systemd-resolve)" systemd-resolve --status

section "Kernel Modules (NVIDIA)"
cmd "lsmod | grep nvidia" sh -c 'lsmod | grep -i nvidia || true'

section "Summary"
# High-level summary lines (best-effort)
OS=$( (lsb_release -ds 2>/dev/null || grep PRETTY_NAME= /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"') || echo "Unknown OS")
KERNEL=$(uname -r 2>/dev/null || echo "?")
CPU=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/,"",$2); print $2; exit}')
NPROC=$(nproc 2>/dev/null || echo "?")
MEM=$(free -h 2>/dev/null | awk '/Mem:/ {print $2" total"}')
GPU=$( (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd "," -) || echo "No NVIDIA GPU detected")
CUDA_VER=$(nvcc --version 2>/dev/null | awk -F, '/release/ {print $2}' | sed 's/^ //')
PY=$(python3 --version 2>/dev/null | awk '{print $2}')

cat <<EOF | sed 's/^/  /'
OS: ${OS}
Kernel: ${KERNEL}
CPU: ${CPU:-Unknown} (${NPROC} cores)
Memory: ${MEM:-Unknown}
GPU(s): ${GPU}
CUDA (nvcc): ${CUDA_VER:-not installed}
Python3: ${PY:-Unknown}
EOF

echo
echo "Done."
