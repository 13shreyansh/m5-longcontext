# Preparation environment inventory

Observed on 2026-08-26 and re-checked on 2026-08-29 SGT. Identifiers such as serial number, hardware UUID,
user name, and provisioning identifiers are deliberately omitted.

## Host

| Item | Observed value |
|---|---|
| Model | MacBook Pro (`Mac17,9`) |
| Chip | Apple M5 Pro |
| CPU | 18 cores; `system_profiler` reports 6 Super and 12 Performance |
| GPU | Integrated Apple M5 Pro, 20 cores |
| Memory | 64 GB unified memory (68,719,476,736 bytes reported) |
| Published unified-memory bandwidth | 307 GB/s |
| Metal | Metal 4 supported |
| OS | macOS 26.6.2, build 25G83 |
| Kernel/architecture | Darwin 25.6.0, arm64 |
| Repository filesystem | 926 GiB total, 752 GiB available at observation time |

The 307 GB/s figure is a vendor specification, not a locally measured result.
Apple's current [MacBook Pro technical specifications](https://www.apple.com/macbook-pro/specs/)
list the 18-core CPU / 20-core GPU M5 Pro configuration with 307 GB/s unified
memory bandwidth and permit 64 GB unified memory. The locally observed CPU,
GPU, and memory configuration matches that row. Apple does not publish a
conventional fp32 peak-FLOP figure on that page, so an MFU percentage is not
invented from an unsupported denominator.

## Compiler and accelerator toolchains

| Tool/capability | Observed value |
|---|---|
| Apple clang / `cc` / `gcc` / `g++` | 21.0.0 (`clang-2100.1.1.101`), arm64 target |
| GNU Make | 3.81 |
| Command Line Tools | `/Library/Developer/CommandLineTools` |
| Standalone `metal` compiler | Not found by `xcrun` |
| Xcode Instruments / `xctrace` | Stub exists, but full Xcode runtime is not installed; active developer directory is Command Line Tools |
| Application-visible Metal counters | One `timestamp` set containing only `GPUTimestamp`; no utilization, bandwidth, cache or stall counters |
| `powermetrics` GPU sampler | Installed but requires superuser; non-interactive `sudo` reports that a password is required |
| CMake / Ninja | Absent globally; repository-local CMake 4.4.2 and Ninja 1.13.0 installed in the ignored MLX experiment environment on 2026-08-29 |
| `nvidia-smi` / `nvcc` | Not found |
| `hipcc` / `rocminfo` | Not found |

Metal 4 support does not imply CUDA compatibility. No NVIDIA driver version is
applicable on this host, and no separate Apple GPU driver version was exposed
by the recorded `system_profiler` inventory.

## Isolated Python environment

Creation/install commands:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch tensorflow
```

The `.venv/` directory is ignored and contains no project secrets. Recorded
versions:

| Package/runtime | Version |
|---|---|
| System/venv Python | 3.9.6 |
| pip | 26.0.1 |
| PyTorch | 2.8.0 |
| TensorFlow | 2.20.0 |
| Keras | 3.10.0 |
| NumPy | 2.0.2 |
| Triton | Not installed |

Top-level pins are in `requirements-preparation.txt`; the complete environment
snapshot is in `requirements-preparation.lock.txt`.

Post-start experiments also use ignored repository-local runtimes managed by
uv 0.12.7: CPython 3.12.14, 3.13.15 and 3.14.7; PyTorch 2.13.0 and the rejected
official nightly 2.15.0.dev20260829; Triton 3.7.0; MLX 0.32.0; and mlx-mfa
2.62.1. Details and exact commands are in
[`UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md`](UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md).
The nightly wheel checksum, full 13-row result and C++20 compatibility boundary
are recorded in
[`SOLUTION_MILESTONE_84_2026-08-30.md`](SOLUTION_MILESTONE_84_2026-08-30.md).

## Framework-visible devices

PyTorch probe:

```text
torch=2.8.0
torch.cuda.is_available=False
torch.cuda.device_count=0
torch.backends.mps.is_built=True
torch.backends.mps.is_available=True
```

TensorFlow probe:

```text
tensorflow=2.20.0
tf.physical_devices=[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]
tf.gpu_devices=[]
```

TensorFlow imports emit a non-fatal `urllib3` warning because the system Python
is linked against LibreSSL 2.8.3 while urllib3 2.6.3 expects OpenSSL 1.1.1 or
newer. The recorded CPU benchmark still exited successfully.

## Inventory commands

```bash
sw_vers
uname -a
uname -m
sysctl -n machdep.cpu.brand_string
sysctl -n hw.physicalcpu hw.logicalcpu hw.memsize
system_profiler SPHardwareDataType SPDisplaysDataType SPSoftwareDataType
python3 --version
clang --version
xcode-select -p
xcrun --find clang
xcrun --find metal
xcrun metal --version
nvidia-smi --version
nvcc --version
hipcc --version
rocminfo --version
df -h .
vm_stat
```
