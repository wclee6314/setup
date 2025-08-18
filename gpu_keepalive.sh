#!/usr/bin/env bash
# gpu_keepalive.sh — GPU idle 종료 방지용 최소 연산 루프
# torch/cupy/tensorflow/jax가 없어도 libcuda(Driver API)만 있으면 동작합니다.
set -euo pipefail

INTERVAL="${KEEPALIVE_INTERVAL:-300}"
QUIET=0
FORCE_BACKEND="${KEEPALIVE_BACKEND:-}"   # -b 로도 강제 가능 (torch|cupy|tensorflow|jax|cuda)

usage() {
  cat >&2 <<'USAGE'
사용법: ./gpu_keepalive.sh [-i 초] [-q] [-b 백엔드]
  -i N   주기(초), 기본 300
  -q     조용히(정상 로그 숨김), 단 오류는 항상 표시
  -b B   백엔드 강제 지정: torch|cupy|tensorflow|jax|cuda
USAGE
}

# 로그/에러
log() { [[ "$QUIET" -eq 0 ]] && echo "[gpu-keepalive] $(date '+%F %T') $*"; }
die() { echo "[gpu-keepalive] $(date '+%F %T') ERROR: $*" >&2; exit 1; }

# 옵션 파싱
while getopts ":i:qb:h" opt; do
  case "$opt" in
    i) INTERVAL="$OPTARG" ;;
    q) QUIET=1 ;;
    b) FORCE_BACKEND="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

# 간단한 유효성 검사
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [ "$INTERVAL" -lt 1 ]; then
  die "INTERVAL 은 1 이상의 정수여야 합니다: '$INTERVAL'"
fi
if [[ -n "$FORCE_BACKEND" ]] && ! [[ "$FORCE_BACKEND" =~ ^(torch|cupy|tensorflow|jax|cuda)$ ]]; then
  die "지원되지 않는 백엔드: $FORCE_BACKEND (torch|cupy|tensorflow|jax|cuda)"
fi

# 단일 인스턴스 락 (flock 우선, 없으면 디렉터리 방식)
prepare_lock() {
  if command -v flock >/dev/null 2>&1; then
    LOCKFILE="/tmp/gpu_keepalive.lock"
    # 파일을 연 뒤 락을 비차단으로 시도
    exec 9>"$LOCKFILE" || die "락 파일 생성 실패: $LOCKFILE"
    if ! flock -n 9; then
      die "이미 실행 중입니다. ($LOCKFILE)"
    fi
    # 현재 PID 기록 (파일은 FD 9로 열려 있으며, 락은 유지됨)
    printf '%d\n' "$$" 1>&9
    trap 'flock -u 9; rm -f "$LOCKFILE"' EXIT INT TERM
  else
    # flock이 없을 때: 디렉터리 + PID 방식
    LOCKDIR="/tmp/gpu_keepalive.lock.d"
    if mkdir "$LOCKDIR" 2>/dev/null; then
      echo $$ > "$LOCKDIR/pid"
      trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
    else
      pid=""
      [[ -f "$LOCKDIR/pid" ]] && pid="$(cat "$LOCKDIR/pid" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        die "이미 실행 중(PID=$pid)."
      fi
      # 스테일 락 정리 후 재생성
      rm -rf "$LOCKDIR" || die "스테일 락 정리 실패: $LOCKDIR"
      mkdir "$LOCKDIR" || die "락 재생성 실패: $LOCKDIR"
      echo $$ > "$LOCKDIR/pid"
      trap 'rm -rf "$LOCKDIR"' EXIT INT TERM
    fi
  fi
}

detect_backend() {
  if [[ -n "$FORCE_BACKEND" ]]; then
    echo "$FORCE_BACKEND"
    return 0
  fi
python3 - <<'PY'
import importlib, sys, ctypes

def ok_torch():
    try:
        torch = importlib.import_module("torch")
        return bool(getattr(torch.cuda, "is_available", lambda: False)())
    except Exception:
        return False

def ok_cupy():
    try:
        cp = importlib.import_module("cupy")
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

def ok_tf():
    try:
        tf = importlib.import_module("tensorflow")
        return bool(tf.config.list_physical_devices('GPU'))
    except Exception:
        return False

def ok_jax():
    try:
        jax = importlib.import_module("jax")
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False

def ok_cuda_driver():
    # 의존성 0: libcuda만 필요
    for name in ("libcuda.so.1","libcuda.so"):
        try:
            cuda = ctypes.CDLL(name)
            break
        except OSError:
            cuda = None
    if cuda is None:
        return False
    cuda.cuInit.restype = ctypes.c_int
    if cuda.cuInit(0) != 0:
        return False
    cuDeviceGetCount = getattr(cuda, "cuDeviceGetCount", None)
    if cuDeviceGetCount is None:
        return True
    cuDeviceGetCount.restype = ctypes.c_int
    count = ctypes.c_int()
    if cuDeviceGetCount(ctypes.byref(count)) != 0:
        return False
    return count.value > 0

for name, fn in [
    ("torch", ok_torch),
    ("cupy", ok_cupy),
    ("tensorflow", ok_tf),
    ("jax", ok_jax),
    ("cuda", ok_cuda_driver),
]:
    if fn():
        print(name); sys.exit(0)

print("none")
PY
}

gpu_ping() {
  case "$BACKEND" in
    torch)
      python3 - <<'PY'
import torch
a = torch.rand((32,32), device='cuda'); b = torch.rand((32,32), device='cuda')
c = a @ b; _ = float(c.sum().item()); torch.cuda.synchronize(); print("ok")
PY
      ;;
    cupy)
      python3 - <<'PY'
import cupy as cp
a = cp.random.random((32,32)); b = cp.random.random((32,32))
c = a.dot(b); _ = float(c.sum().get()); cp.cuda.Stream.null.synchronize(); print("ok")
PY
      ;;
    tensorflow)
      python3 - <<'PY'
import tensorflow as tf
with tf.device('/GPU:0'):
    a = tf.random.uniform([32,32]); b = tf.random.uniform([32,32])
    c = tf.linalg.matmul(a,b); _ = float(tf.reduce_sum(c).numpy()); print("ok")
PY
      ;;
    jax)
      python3 - <<'PY'
import jax, jax.numpy as jnp
a = jnp.ones((32,32)); b = jnp.ones((32,32)); c = a @ b
_ = float(jax.device_get(jnp.sum(c))); print("ok")
PY
      ;;
    cuda)
      # ✅ 의존성 없는 CUDA Driver API(CTypes) — 작은 memset 실행
      python3 - <<'PY'
import ctypes, sys
from ctypes import byref, c_int, c_size_t, c_void_p, c_uint8

cuda = None
for name in ("libcuda.so.1","libcuda.so"):
    try:
        cuda = ctypes.CDLL(name); break
    except OSError: pass
if cuda is None: sys.exit("no libcuda")

def sym(names):
    for n in names:
        f = getattr(cuda, n, None)
        if f: return f
    raise AttributeError(names[0])

def chk(code, where):
    if code != 0:
        print(f"CUDA error {code} at {where}"); sys.exit(1)

cuda.cuInit.restype = c_int
chk(cuda.cuInit(0), "cuInit")
cuDeviceGet = cuda.cuDeviceGet; cuDeviceGet.restype = c_int
dev = c_int()
chk(cuDeviceGet(byref(dev), 0), "cuDeviceGet")

cuCtxCreate = sym(["cuCtxCreate_v2","cuCtxCreate"]); cuCtxCreate.restype = c_int
cuCtxDestroy = sym(["cuCtxDestroy_v2","cuCtxDestroy"]); cuCtxDestroy.restype = c_int
ctx = c_void_p()
chk(cuCtxCreate(byref(ctx), 0, dev), "cuCtxCreate")

try:
    cuMemAlloc = sym(["cuMemAlloc_v2","cuMemAlloc"]); cuMemAlloc.restype = c_int
    cuMemFree  = sym(["cuMemFree_v2","cuMemFree"]);   cuMemFree.restype  = c_int
    cuMemsetD8 = sym(["cuMemsetD8_v2","cuMemsetD8"]); cuMemsetD8.restype = c_int
    cuCtxSync  = getattr(cuda, "cuCtxSynchronize");    cuCtxSync.restype  = c_int

    dptr = c_void_p()
    nbytes = c_size_t(4096)  # 4KB
    chk(cuMemAlloc(byref(dptr), nbytes.value), "cuMemAlloc")
    try:
        chk(cuMemsetD8(dptr, c_uint8(0), nbytes.value), "cuMemsetD8")
        chk(cuCtxSync(), "cuCtxSynchronize")
    finally:
        cuMemFree(dptr)
    print("ok")
finally:
    cuCtxDestroy(ctx)
PY
      ;;
    *)
      return 1
      ;;
  esac
}

# --- 실행 흐름 ---
prepare_lock

BACKEND="$(detect_backend)"
if [[ "$BACKEND" == "none" ]]; then
  die "GPU 백엔드를 찾지 못했습니다. (torch/cupy/tensorflow/jax/libcuda). 컨테이너에서 GPU 드라이버(libcuda) 노출을 확인하세요."
fi

log "선택된 백엔드: ${BACKEND} / 주기(초): ${INTERVAL}"

while true; do
  if OUT="$(gpu_ping 2>&1)"; then
    log "GPU keep-alive 수행 완료 (${BACKEND})"
  else
    echo "[gpu-keepalive] $(date '+%F %T') ERROR: keep-alive 실패 (${BACKEND}): ${OUT}" >&2
  fi
  sleep "$INTERVAL"
done
