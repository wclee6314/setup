"""
GPU Utilization & Memory Maintenance Script

각 GPU의 utilization과 memory 사용량을 입력된 목표 수치(--util/--mem, 기본 0.25)
이상으로 유지한다.

동작 규칙:
1. 매 사이클 시작 시 우리가 추가한 부하를 모두 풀고, 다른 프로세스의
   사용량을 측정한다.
2. 다른 프로세스 사용량이 이미 목표 수치 이상이면 해당 항목에 추가 부하를
   주지 않고 (불필요한 연산 수행 안 함) CHECK_INTERVAL(기본 10분) 동안
   대기한 뒤 다시 확인한다.
3. 다른 프로세스 사용량이 목표 수치 미만이면 부족분만큼 우리가 부하를
   추가하여 합계가 목표 수치 이상이 되도록 한다. CHECK_INTERVAL 동안 부하를
   유지하고, 다음 사이클 시작 시 다시 측정/결정한다.
4. utilization과 memory는 항목별로 독립 제어된다.

각 GPU는 독립된 스레드에서 동작하며 Ctrl+C로 종료한다.
"""

import torch
import threading
import signal
import time
import subprocess
import argparse

TARGET_UTIL_RATIO = 0.25     # 목표 GPU utilization (0.25 = 25%)
TARGET_MEM_RATIO = 0.25      # 목표 GPU memory (0.25 = 25%)
CHECK_INTERVAL = 10 * 60     # 상태 확인 주기 (초) = 10분
MATRIX_SIZE = 2048           # 연산용 정방행렬 크기 (float32)
COMPUTE_BATCH = 10           # 동기화 전 연속 matmul 횟수
MIN_IDLE = 0.001             # 최소 idle 시간 (초)
SAMPLE_COUNT = 3             # 한 번 측정할 때 평균에 사용할 샘플 수
SAMPLE_INTERVAL = 1.0        # 샘플 간 간격 (초)
QUIESCE_TIME = 3.0           # 측정 직전 우리 부하를 비우고 대기하는 시간 (초)

stop_event = threading.Event()


def _sample_once(gpu_id):
    """nvidia-smi로 (utilization%, memory_used MiB, memory_total MiB) 1회 샘플"""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None, None, None


def _sample_avg(gpu_id):
    """SAMPLE_COUNT회 샘플의 평균 반환"""
    utils, mems, totals = [], [], []
    for i in range(SAMPLE_COUNT):
        u, m, t = _sample_once(gpu_id)
        if u is not None:
            utils.append(u)
            mems.append(m)
            totals.append(t)
        if i < SAMPLE_COUNT - 1:
            if stop_event.wait(SAMPLE_INTERVAL):
                break
    if not utils:
        return None, None, None
    return sum(utils) / len(utils), sum(mems) / len(mems), totals[-1]


def _run_compute(gpu_id, a, b, device, target_util_ratio, duration):
    """duration 동안 duty-cycle 방식으로 target_util_ratio 만큼 utilization 발생.
    target_util_ratio<=0 이거나 텐서가 없으면 idle 상태로 대기만 함."""
    if target_util_ratio <= 0 or a is None or b is None:
        stop_event.wait(duration)
        return

    start = time.time()
    while not stop_event.is_set():
        if time.time() - start >= duration:
            break

        t0 = time.time()
        for _ in range(COMPUTE_BATCH):
            c = torch.matmul(a, b)
        torch.cuda.synchronize(device)
        del c
        busy = time.time() - t0

        if 0 < target_util_ratio < 1:
            idle = busy * (1.0 - target_util_ratio) / target_util_ratio
        else:
            idle = MIN_IDLE

        remain = duration - (time.time() - start)
        if remain <= 0:
            break
        if stop_event.wait(min(max(idle, MIN_IDLE), remain)):
            break


def stress_loop(gpu_id):
    """단일 GPU 독립 제어 루프 (10분 주기로 측정 → 적용 → 유지 반복)"""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.init()

    # CUDA context로 인한 기본 메모리 점유량 측정 (시작 시 1회)
    torch.cuda.empty_cache()
    if stop_event.wait(QUIESCE_TIME):
        return
    _, base_mem, mem_total = _sample_avg(gpu_id)
    if mem_total is None:
        print(f"  [GPU {gpu_id}] nvidia-smi 사용 불가. 스레드 종료.")
        return

    target_util_pct = TARGET_UTIL_RATIO * 100.0
    target_mem_mb = TARGET_MEM_RATIO * mem_total

    print(f"  [GPU {gpu_id}] Ready. CUDA baseline mem≈{base_mem:.0f} MiB / "
          f"total {mem_total} MiB. "
          f"Target util={target_util_pct:.0f}%, mem={target_mem_mb:.0f} MiB. "
          f"Check every {CHECK_INTERVAL:.0f}s.")

    while not stop_event.is_set():
        # 측정 직전 우리 부하 모두 해제 → 다른 프로세스 사용량만 보이도록
        torch.cuda.empty_cache()
        if stop_event.wait(QUIESCE_TIME):
            break

        util, mem_used, _ = _sample_avg(gpu_id)
        ts = time.strftime("%H:%M:%S")
        if util is None:
            print(f"  [GPU {gpu_id}] {ts} 측정 실패. {CHECK_INTERVAL:.0f}s 후 재시도.")
            if stop_event.wait(CHECK_INTERVAL):
                break
            continue

        # 우리는 idle 상태이므로 측정값 ≈ 다른 프로세스 사용량
        # 메모리는 CUDA context baseline 만큼은 우리 것이므로 빼줌
        other_util = util
        other_mem = max(0.0, mem_used - base_mem)
        other_mem_pct = other_mem / mem_total * 100.0

        util_over = other_util >= target_util_pct
        mem_over = other_mem >= target_mem_mb
        util_thr = f">={target_util_pct:.0f}%"
        mem_thr = f">={TARGET_MEM_RATIO*100:.0f}%"

        print(f"  [GPU {gpu_id}] {ts} 다른 프로세스: "
              f"util≈{other_util:.0f}%{f' ({util_thr})' if util_over else ''}, "
              f"mem≈{other_mem:.0f} MiB ({other_mem_pct:.1f}%)"
              f"{f' ({mem_thr})' if mem_over else ''}")

        # 항목별 독립 제어: 목표 수치 이상이면 해당 항목 부하 중지
        my_util_pct = 0.0 if util_over else (target_util_pct - other_util)
        my_mem_mb = 0.0 if mem_over else (target_mem_mb - other_mem)
        my_util_ratio = my_util_pct / 100.0

        if util_over and mem_over:
            state = (f"[IDLE] 목표 이미 충족 - 추가 부하 없이 "
                     f"{CHECK_INTERVAL:.0f}s 대기")
        elif util_over:
            state = f"[PARTIAL] util OFF, mem +{my_mem_mb:.0f} MiB"
        elif mem_over:
            state = f"[PARTIAL] util +{my_util_pct:.1f}%, mem OFF"
        else:
            state = (f"[ACTIVE] util +{my_util_pct:.1f}%, "
                     f"mem +{my_mem_mb:.0f} MiB")
        print(f"  [GPU {gpu_id}] {ts} {state}")

        # 메모리 점유 (mem 항목)
        mem_holder = None
        my_mem_bytes = int(my_mem_mb * 1024 * 1024)
        if my_mem_bytes > 0:
            try:
                n_elems = my_mem_bytes // 4  # float32
                mem_holder = torch.empty(n_elems, dtype=torch.float32, device=device)
            except Exception as e:
                print(f"  [GPU {gpu_id}] {ts} 메모리 할당 실패: {e}")
                mem_holder = None

        # matmul 텐서 (util 항목)
        a = b = None
        if my_util_pct > 0:
            try:
                a = torch.randn(MATRIX_SIZE, MATRIX_SIZE,
                                dtype=torch.float32, device=device)
                b = torch.randn(MATRIX_SIZE, MATRIX_SIZE,
                                dtype=torch.float32, device=device)
                torch.cuda.synchronize(device)
            except Exception as e:
                print(f"  [GPU {gpu_id}] {ts} matmul 텐서 할당 실패: {e}")
                a = b = None

        # CHECK_INTERVAL 동안 부하 유지 (memory는 holder로, util은 duty-cycle로)
        _run_compute(gpu_id, a, b, device, my_util_ratio, CHECK_INTERVAL)

        # 다음 사이클의 측정을 위해 해제
        del a, b
        if mem_holder is not None:
            del mem_holder

    torch.cuda.empty_cache()
    print(f"  [GPU {gpu_id}] Stopped.")


def main():
    global TARGET_UTIL_RATIO, TARGET_MEM_RATIO, MATRIX_SIZE, CHECK_INTERVAL

    parser = argparse.ArgumentParser(
        description="GPU Utilization & Memory Maintenance (per-GPU 25% target)"
    )
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU indices (e.g. '0,2,3'). Default: all GPUs.")
    parser.add_argument("--util", type=float, default=TARGET_UTIL_RATIO,
                        help=f"Target GPU utilization ratio 0~1 (default: {TARGET_UTIL_RATIO})")
    parser.add_argument("--mem", type=float, default=TARGET_MEM_RATIO,
                        help=f"Target GPU memory ratio 0~1 (default: {TARGET_MEM_RATIO})")
    parser.add_argument("--interval", type=float, default=CHECK_INTERVAL,
                        help=f"Check interval in seconds (default: {CHECK_INTERVAL})")
    parser.add_argument("--matrix-size", type=int, default=MATRIX_SIZE,
                        help=f"Matrix size for matmul (default: {MATRIX_SIZE})")
    args = parser.parse_args()

    TARGET_UTIL_RATIO = args.util
    TARGET_MEM_RATIO = args.mem
    MATRIX_SIZE = args.matrix_size
    CHECK_INTERVAL = args.interval

    num_gpus = torch.cuda.device_count()
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        gpu_ids = [g for g in gpu_ids if 0 <= g < num_gpus]
    else:
        gpu_ids = list(range(num_gpus))

    if not gpu_ids:
        print("No valid GPU IDs. Exiting.")
        return

    print(f"\nStarting on GPU(s): {gpu_ids} | "
          f"target util {TARGET_UTIL_RATIO*100:.0f}%, "
          f"mem {TARGET_MEM_RATIO*100:.0f}%, "
          f"check every {CHECK_INTERVAL:.0f}s\n")

    def signal_handler(sig, frame):
        print("\nStopping all GPU stress threads...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    threads = []
    for gid in gpu_ids:
        t = threading.Thread(target=stress_loop, args=(gid,), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All GPU stress threads stopped. Exiting.")


if __name__ == "__main__":
    main()