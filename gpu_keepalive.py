import argparse
import math
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pynvml
import torch


DEFAULT_ACTIVE_SECONDS = 5
DEFAULT_REST_SECONDS = 5
DEFAULT_TARGET_UTIL = 100
DEFAULT_MEMORY_BUDGET_MIB = 10
DEFAULT_IDLE_TRIGGER_SECONDS = 5.0
DEFAULT_BURST_SECONDS = 5.0
DEFAULT_IDLE_UTIL_THRESHOLD = 41
DEFAULT_SLEEP_CYCLES = 300_000_000
DEFAULT_PAUSE_UTIL_THRESHOLD = 1
DEFAULT_PROCESS_UTIL_SAMPLE_MAX_AGE_SECONDS = 3.0


def align_size(value: int, multiple: int = 256) -> int:
    return max(multiple, (value // multiple) * multiple)


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f}GiB"


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def safe_set_torch_cpu_threads() -> None:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def current_process_pid_candidates() -> set[int]:
    pids = {os.getpid()}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(("NSpid:", "NStgid:")):
                    continue
                for token in line.split()[1:]:
                    if token.isdigit():
                        pids.add(int(token))
    except OSError:
        pass
    return pids


def nvml_device_uuid(handle) -> str:
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    if isinstance(uuid, bytes):
        return uuid.decode("utf-8")
    return str(uuid)


def lookup_physical_index(token: str, uuid_to_index: Dict[str, int]) -> Optional[int]:
    token = token.strip()
    if not token or token == "-1":
        return None
    if token.isdigit():
        return int(token)

    upper_token = token.upper()
    for uuid, index in uuid_to_index.items():
        upper_uuid = uuid.upper()
        if upper_uuid == upper_token or upper_uuid.startswith(upper_token):
            return index
    return None


def visible_to_physical_map() -> List[Tuple[int, int, str, str]]:
    visible_count = torch.cuda.device_count()
    physical_count = pynvml.nvmlDeviceGetCount()
    uuid_to_index: Dict[str, int] = {}
    for physical_index in range(physical_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        uuid_to_index[nvml_device_uuid(handle)] = physical_index

    env_tokens = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]

    mapping: List[Tuple[int, int, str, str]] = []
    for visible_index in range(visible_count):
        physical_index: Optional[int] = None

        props = torch.cuda.get_device_properties(visible_index)
        torch_uuid = getattr(props, "uuid", None)
        if torch_uuid:
            physical_index = uuid_to_index.get(str(torch_uuid))

        if physical_index is None and visible_index < len(env_tokens):
            physical_index = lookup_physical_index(env_tokens[visible_index], uuid_to_index)

        if physical_index is None and visible_index < physical_count:
            physical_index = visible_index

        if physical_index is None or not (0 <= physical_index < physical_count):
            raise RuntimeError(
                f"Could not resolve physical GPU index for visible GPU {visible_index}."
            )

        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mapping.append((visible_index, physical_index, str(name), nvml_device_uuid(handle)))

    return mapping


def normalize_used_gpu_memory(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value >= (1 << 63):
        return None
    return int(value)


def read_process_name(pid: int) -> str:
    nvml_get_name = getattr(pynvml, "nvmlSystemGetProcessName", None)
    if nvml_get_name is not None:
        try:
            name = nvml_get_name(pid)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if name:
                return str(name)
        except pynvml.NVMLError:
            pass
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return "unknown"


def nvml_running_process_infos(handle) -> List["ProcessInfo"]:
    process_getters: Sequence[Sequence[str]] = (
        ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses"),
        ("nvmlDeviceGetGraphicsRunningProcesses_v3", "nvmlDeviceGetGraphicsRunningProcesses"),
        ("nvmlDeviceGetMPSComputeRunningProcesses_v3", "nvmlDeviceGetMPSComputeRunningProcesses"),
    )

    labels = ("compute", "graphics", "mps")
    process_infos: Dict[Tuple[int, str], ProcessInfo] = {}
    for getter_names, label in zip(process_getters, labels):
        for getter_name in getter_names:
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                for proc in getter(handle):
                    pid = getattr(proc, "pid", None)
                    if pid is not None:
                        pid = int(pid)
                        process_infos[(pid, label)] = ProcessInfo(
                            pid=pid,
                            used_gpu_memory=normalize_used_gpu_memory(
                                getattr(proc, "usedGpuMemory", None)
                            ),
                            kind=label,
                            name=read_process_name(pid),
                        )
                break
            except (
                pynvml.NVMLError_NoPermission,
                pynvml.NVMLError_NotSupported,
            ):
                break
    return sorted(process_infos.values(), key=lambda item: (item.pid, item.kind))


def snapshot_visible_gpu_pids(
    gpu_mapping: Sequence[Tuple[int, int, str, str]]
) -> Dict[int, set[int]]:
    snapshot: Dict[int, set[int]] = {}
    for visible_index, physical_index, _name, _uuid in gpu_mapping:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        snapshot[visible_index] = {
            proc.pid for proc in nvml_running_process_infos(handle)
        }
    return snapshot


def discover_nvml_self_pids(
    gpu_mapping: Sequence[Tuple[int, int, str, str]],
    baseline_pids: Dict[int, set[int]],
    dtype: torch.dtype,
) -> set[int]:
    probes: List[torch.Tensor] = []
    pid_counts: Dict[int, int] = {}
    visible_count = len(gpu_mapping)

    try:
        for visible_index, _physical_index, _name, _uuid in gpu_mapping:
            device = torch.device(f"cuda:{visible_index}")
            with torch.cuda.device(device):
                probes.append(torch.empty(1, device=device, dtype=dtype))

        for visible_index, _physical_index, _name, _uuid in gpu_mapping:
            torch.cuda.synchronize(torch.device(f"cuda:{visible_index}"))

        time.sleep(0.2)

        for visible_index, physical_index, _name, _uuid in gpu_mapping:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            current_pids = {
                proc.pid for proc in nvml_running_process_infos(handle)
            }
            new_pids = current_pids - baseline_pids.get(visible_index, set())
            for pid in new_pids:
                pid_counts[pid] = pid_counts.get(pid, 0) + 1
    finally:
        probes.clear()
        for visible_index, _physical_index, _name, _uuid in gpu_mapping:
            try:
                with torch.cuda.device(torch.device(f"cuda:{visible_index}")):
                    torch.cuda.empty_cache()
            except RuntimeError:
                pass

    shared_pids = {pid for pid, count in pid_counts.items() if count == visible_count}
    if shared_pids:
        return shared_pids
    return set(pid_counts)


@dataclass
class WorkerSnapshot:
    visible_index: int
    physical_index: int
    state: str
    util: int
    matrix_size: int
    memory_used_bytes: int
    nvml_self_memory_bytes: int
    other_processes: bool
    note: str


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    used_gpu_memory: Optional[int]
    kind: str
    name: str
    sm_util: Optional[int] = None
    mem_util: Optional[int] = None
    timestamp_us: Optional[int] = None


def nvml_process_utilization_samples(handle) -> Dict[int, Tuple[int, int, int]]:
    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
    except (
        pynvml.NVMLError_NoPermission,
        pynvml.NVMLError_NotSupported,
        pynvml.NVMLError_NotFound,
    ):
        return {}

    by_pid: Dict[int, Tuple[int, int, int]] = {}
    for sample in samples:
        pid = getattr(sample, "pid", None)
        if pid is None:
            continue
        pid = int(pid)
        sm_util = int(getattr(sample, "smUtil", 0) or 0)
        mem_util = int(getattr(sample, "memUtil", 0) or 0)
        timestamp_us = int(getattr(sample, "timeStamp", 0) or 0)

        previous = by_pid.get(pid)
        if (
            previous is None
            or timestamp_us > previous[2]
            or (timestamp_us == previous[2] and sm_util > previous[0])
        ):
            by_pid[pid] = (sm_util, mem_util, timestamp_us)
    return by_pid


def recent_process_util_sample(timestamp_us: int, max_age_seconds: float) -> bool:
    if timestamp_us <= 0:
        return True
    age_seconds = time.time() - (timestamp_us / 1_000_000)
    return age_seconds <= max_age_seconds


class BurstCoordinator:
    def __init__(
        self,
        visible_indices: Sequence[int],
        idle_trigger_seconds: float,
        burst_seconds: float,
        initial_burst_seconds: float,
        idle_util_threshold: int,
    ) -> None:
        now = time.monotonic()
        self.visible_indices = list(visible_indices)
        self.idle_trigger_seconds = max(0.1, idle_trigger_seconds)
        self.burst_seconds = max(0.1, burst_seconds)
        self.idle_util_threshold = max(0, idle_util_threshold)
        self._lock = threading.Lock()
        self._util: Dict[int, int] = {index: 0 for index in self.visible_indices}
        self._all_zero_since: Optional[float] = None
        self._burst_until: Optional[float] = now + max(0.1, initial_burst_seconds)

    def _all_util_zero(self) -> bool:
        return all(util <= self.idle_util_threshold for util in self._util.values())

    def _advance(self, visible_index: int, util: int) -> bool:
        now = time.monotonic()

        self._util[visible_index] = util
        if self._burst_until is not None and now < self._burst_until:
            return True
        if self._burst_until is not None:
            self._burst_until = None

        if not self._all_util_zero():
            self._all_zero_since = None
            return False

        if self._all_zero_since is None:
            self._all_zero_since = now
            return False

        if now - self._all_zero_since >= self.idle_trigger_seconds:
            self._burst_until = now + self.burst_seconds
            self._all_zero_since = None
            return True
        return False

    def update(self, visible_index: int, util: int) -> bool:
        with self._lock:
            return self._advance(visible_index, util)


class GpuWorker(threading.Thread):
    def __init__(
        self,
        visible_index: int,
        physical_index: int,
        gpu_name: str,
        gpu_uuid: str,
        args: argparse.Namespace,
        stop_event: threading.Event,
        coordinator: BurstCoordinator,
        self_pids: set[int],
    ) -> None:
        super().__init__(daemon=True)
        self.visible_index = visible_index
        self.physical_index = physical_index
        self.gpu_name = gpu_name
        self.gpu_uuid = gpu_uuid
        self.args = args
        self.stop_event = stop_event
        self.coordinator = coordinator
        self.self_pids = set(self_pids)
        self.self_pids.update(current_process_pid_candidates())
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        self.device = torch.device(f"cuda:{visible_index}")
        self.dtype = dtype_from_name(args.dtype)
        self.dtype_bytes = bytes_per_element(self.dtype)
        self.total_memory_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).total)
        self.max_memory_budget_bytes = min(
            int(self.total_memory_bytes * self.args.max_memory_fraction),
            max(1, self.args.memory_budget_mib * 1024 * 1024),
        )
        self.max_matrix_size = self._compute_max_matrix_size()
        requested_min_matrix_size = align_size(args.min_matrix_size)
        budget_min_matrix = self._matrix_size_for_fraction(
            self.args.min_memory_budget_fraction
        )
        self.min_matrix_size = min(
            max(1, requested_min_matrix_size),
            max(1, budget_min_matrix),
            max(1, self.max_matrix_size),
        )
        self._base_process_memory_bytes = int(torch.cuda.memory_allocated(self.device))
        self.matrix_size = min(
            self.max_matrix_size,
            self._matrix_size_for_fraction(
                args.start_memory_fraction,
                minimum_size=self.min_matrix_size,
            ),
        )
        self.util_upper_margin = args.util_upper_margin
        self.status_lock = threading.Lock()
        self.state = "INIT"
        self.last_util = 0
        self.other_processes = False
        self.note = "starting"
        self.last_memory_used_bytes = 0
        self.last_nvml_self_memory_bytes = 0
        self.blockers: List[ProcessInfo] = []
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None

    def _compute_max_matrix_size(self) -> int:
        max_size = int(math.sqrt(self.max_memory_budget_bytes / (3 * self.dtype_bytes)))
        max_size = align_size(max(1, max_size))
        if self.args.hard_max_matrix_size > 0:
            max_size = min(max_size, align_size(self.args.hard_max_matrix_size))
        return max_size

    def _matrix_size_for_fraction(self, fraction: float, minimum_size: int = 1) -> int:
        fraction = max(0.001, min(1.0, fraction))
        budget_bytes = max(1, int(self.max_memory_budget_bytes * fraction))
        size = int(math.sqrt(budget_bytes / (3 * self.dtype_bytes)))
        return min(self.max_matrix_size, max(int(minimum_size), align_size(size)))

    def _process_memory_used_bytes(self) -> int:
        try:
            return int(torch.cuda.memory_allocated(self.device))
        except RuntimeError:
            return 0

    def _nvml_self_memory_used_bytes(self) -> int:
        total = 0
        for proc in nvml_running_process_infos(self.handle):
            if proc.pid in self.self_pids and proc.used_gpu_memory is not None:
                total += proc.used_gpu_memory
        return total

    def _set_status(
        self,
        *,
        state: Optional[str] = None,
        util: Optional[int] = None,
        other_processes: Optional[bool] = None,
        note: Optional[str] = None,
    ) -> None:
        with self.status_lock:
            if state is not None:
                self.state = state
            if util is not None:
                self.last_util = util
            if other_processes is not None:
                self.other_processes = other_processes
            if note is not None:
                self.note = note
            self.last_memory_used_bytes = self._process_memory_used_bytes()
            self.last_nvml_self_memory_bytes = self._nvml_self_memory_used_bytes()

    def _enforce_memory_budget(self) -> bool:
        used = self._process_memory_used_bytes()
        used = max(0, used - self._base_process_memory_bytes)
        budget_bytes = min(
            self.max_memory_budget_bytes,
            self.args.memory_budget_mib * 1024 * 1024,
        )
        return used <= budget_bytes

    def snapshot(self) -> WorkerSnapshot:
        with self.status_lock:
            return WorkerSnapshot(
                visible_index=self.visible_index,
                physical_index=self.physical_index,
                state=self.state,
                util=self.last_util,
                matrix_size=self.matrix_size if self.state == "ACTIVE" else 0,
                memory_used_bytes=self.last_memory_used_bytes,
                nvml_self_memory_bytes=self.last_nvml_self_memory_bytes,
                other_processes=self.other_processes,
                note=self.note,
            )

    def _gpu_util(self) -> int:
        try:
            return int(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
        except pynvml.NVMLError:
            return -1

    def _blocking_processes(self) -> List[ProcessInfo]:
        threshold_util = int(self.args.pause_util_threshold)
        max_sample_age = float(self.args.process_util_sample_max_age_seconds)
        running_infos: Dict[int, ProcessInfo] = {}
        for proc in nvml_running_process_infos(self.handle):
            running_infos.setdefault(proc.pid, proc)

        blockers = []
        for pid, (sm_util, mem_util, timestamp_us) in nvml_process_utilization_samples(
            self.handle
        ).items():
            if pid in self.self_pids:
                continue
            if pid in self.args.ignore_pid:
                continue
            proc = running_infos.get(pid)
            name = proc.name if proc is not None else read_process_name(pid)
            if name in self.args.ignore_process_name:
                continue
            if sm_util < threshold_util:
                continue
            if not recent_process_util_sample(timestamp_us, max_sample_age):
                continue
            blockers.append(
                ProcessInfo(
                    pid=pid,
                    used_gpu_memory=proc.used_gpu_memory if proc is not None else None,
                    kind=proc.kind if proc is not None else "util",
                    name=name,
                    sm_util=sm_util,
                    mem_util=mem_util,
                    timestamp_us=timestamp_us,
                )
            )
        return blockers

    def _other_process_present(self) -> bool:
        self.blockers = self._blocking_processes()
        return bool(self.blockers)

    def _format_blockers(self) -> str:
        if not self.blockers:
            return "none"
        parts = []
        for proc in self.blockers[:4]:
            sm_util = "unknown"
            if proc.sm_util is not None:
                sm_util = f"{proc.sm_util}%"
            mem_bw = "unknown"
            if proc.mem_util is not None:
                mem_bw = f"{proc.mem_util}%"
            alloc = "unknown"
            if proc.used_gpu_memory is not None:
                alloc = f"{proc.used_gpu_memory / (1024 ** 2):.0f}MiB"
            parts.append(
                f"pid={proc.pid}:{proc.name}:{proc.kind}:sm={sm_util}:"
                f"mem_bw={mem_bw}:alloc={alloc}"
            )
        if len(self.blockers) > 4:
            parts.append(f"+{len(self.blockers) - 4} more")
        return ", ".join(parts)

    def _free_buffers(self) -> None:
        self.a = None
        self.b = None
        self.c = None
        try:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        except RuntimeError:
            pass

    def _allocate_buffers(self) -> None:
        size = self.matrix_size
        while size >= self.min_matrix_size:
            try:
                with torch.cuda.device(self.device):
                    self.a = torch.empty((size, size), device=self.device, dtype=self.dtype)
                    self.b = torch.empty((size, size), device=self.device, dtype=self.dtype)
                    self.c = torch.empty((size, size), device=self.device, dtype=self.dtype)
                    self.a.normal_(mean=0.0, std=1.0)
                    self.b.normal_(mean=0.0, std=1.0)
                if not self._enforce_memory_budget():
                    self._free_buffers()
                    next_size = align_size(int(size * 0.85))
                    if next_size <= self.min_matrix_size:
                        raise RuntimeError(
                            f"GPU {self.visible_index}: cannot fit even the minimum matrix within budget."
                        )
                    size = next_size
                    continue
                self.matrix_size = size
                return
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self._free_buffers()
                size = align_size(int(size * 0.85))
        raise RuntimeError(
            f"GPU {self.visible_index}: could not allocate workload tensors even at the minimum size."
        )

    def _ensure_buffers(self) -> None:
        if self.a is None or self.a.shape[0] != self.matrix_size:
            self._free_buffers()
            self._allocate_buffers()

    def _retune_matrix_size(self, util: int) -> None:
        new_size = self.matrix_size
        if util >= 0 and util < self.args.target_util:
            new_size = min(self.max_matrix_size, align_size(int(self.matrix_size * 1.15)))
        elif util >= self.args.target_util + self.util_upper_margin:
            new_size = max(self.min_matrix_size, align_size(int(self.matrix_size * 0.92)))

        if new_size != self.matrix_size:
            self.matrix_size = new_size
            self._ensure_buffers()

    def _pause_until_clear(self) -> None:
        self._free_buffers()
        self._set_status(
            state="PAUSED",
            util=self._gpu_util(),
            other_processes=True,
            note=f"blocked by {self._format_blockers()}",
        )
        while not self.stop_event.wait(self.args.pause_poll_seconds):
            if not self._other_process_present():
                self._set_status(
                    state="READY",
                    util=self._gpu_util(),
                    other_processes=False,
                    note="external process cleared",
                )
                return

    def _run_active_window(self, seconds: float) -> None:
        if self.args.backend == "sleep":
            self._run_sleep_window(seconds)
            return

        if seconds <= 0:
            return
        try:
            self._ensure_buffers()
        except RuntimeError as exc:
            if "cannot fit even the minimum matrix within budget" in str(exc):
                self._set_status(
                    state="REST",
                    util=self._gpu_util(),
                    other_processes=False,
                    note="skip burst: budget too low",
                )
                return
            raise
        end_time = time.monotonic() + seconds
        next_probe = time.monotonic()

        self._set_status(
            state="ACTIVE",
            util=self._gpu_util(),
            other_processes=False,
            note=f"matrix={self.matrix_size}",
        )

        with torch.inference_mode():
            while time.monotonic() < end_time and not self.stop_event.is_set():
                if self._other_process_present():
                    self._pause_until_clear()
                    return

                try:
                    for _ in range(self.args.ops_per_batch):
                        torch.mm(self.a, self.b, out=self.c)
                    torch.cuda.synchronize(self.device)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    self.matrix_size = max(
                        self.min_matrix_size, align_size(int(self.matrix_size * 0.80))
                    )
                    self._free_buffers()
                    self._ensure_buffers()
                    continue

                now = time.monotonic()
                if now >= next_probe:
                    util = self._gpu_util()
                    self._retune_matrix_size(util)
                    self._set_status(
                        state="ACTIVE",
                        util=util,
                        other_processes=False,
                        note=f"matrix={self.matrix_size}",
                    )
                    next_probe = now + self.args.probe_seconds

    def _run_sleep_window(self, seconds: float) -> None:
        if seconds <= 0:
            return

        end_time = time.monotonic() + seconds
        next_probe = time.monotonic()
        self._set_status(
            state="ACTIVE",
            util=self._gpu_util(),
            other_processes=False,
            note=f"sleep_cycles={self.args.sleep_cycles}",
        )

        while time.monotonic() < end_time and not self.stop_event.is_set():
            if self._other_process_present():
                self._pause_until_clear()
                return

            torch.cuda._sleep(self.args.sleep_cycles)
            torch.cuda.synchronize(self.device)

            now = time.monotonic()
            if now >= next_probe:
                util = self._gpu_util()
                self._set_status(
                    state="ACTIVE",
                    util=util,
                    other_processes=False,
                    note=f"sleep_cycles={self.args.sleep_cycles}",
                )
                next_probe = now + self.args.probe_seconds

    def run(self) -> None:
        torch.cuda.set_device(self.device)
        while not self.stop_event.is_set():
            util = self._gpu_util()
            if self.coordinator.update(self.visible_index, util):
                if self._other_process_present():
                    self._pause_until_clear()
                    continue
                self._run_active_window(self.args.burst_seconds)
                if self.stop_event.is_set():
                    break

            if self._other_process_present():
                self._pause_until_clear()
                continue

            self._free_buffers()
            self._set_status(
                state="REST",
                util=util,
                other_processes=False,
                note="resting",
            )
            if self.stop_event.wait(0.5):
                break

        self._free_buffers()
        self._set_status(
            state="STOPPED",
            util=self._gpu_util(),
            other_processes=False,
            note="stopped",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep all visible GPUs busy in short bursts with low host RAM usage. "
            "By default, run a 5s 100% burst after all GPUs stay idle for 5s, "
            "and pause only when another process is actively using GPU SMs."
        )
    )
    parser.add_argument("--active-seconds", type=float, default=DEFAULT_ACTIVE_SECONDS)
    parser.add_argument("--rest-seconds", type=float, default=DEFAULT_REST_SECONDS)
    parser.add_argument("--target-util", type=int, default=DEFAULT_TARGET_UTIL)
    parser.add_argument("--util-upper-margin", type=int, default=20)
    parser.add_argument("--probe-seconds", type=float, default=0.75)
    parser.add_argument("--pause-poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--pause-util-threshold",
        type=int,
        default=DEFAULT_PAUSE_UTIL_THRESHOLD,
        help=(
            "Pause when another process has per-process GPU SM utilization at or "
            "above this percent. VRAM allocation is ignored for pause decisions."
        ),
    )
    parser.add_argument(
        "--process-util-sample-max-age-seconds",
        type=float,
        default=DEFAULT_PROCESS_UTIL_SAMPLE_MAX_AGE_SECONDS,
        help="Ignore per-process utilization samples older than this many seconds.",
    )
    parser.add_argument("--status-seconds", type=float, default=5.0)
    parser.add_argument("--ops-per-batch", type=int, default=2)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--backend",
        choices=["sleep", "matmul"],
        default="sleep",
        help="sleep uses CUDA busy-wait kernels for high util with near-zero tensor memory.",
    )
    parser.add_argument(
        "--sleep-cycles",
        type=int,
        default=DEFAULT_SLEEP_CYCLES,
        help="GPU clock cycles per CUDA sleep kernel in sleep backend.",
    )
    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=DEFAULT_MEMORY_BUDGET_MIB,
        help="Per-process memory budget in MiB.",
    )
    parser.add_argument(
        "--min-memory-budget-fraction",
        type=float,
        default=0.01,
        help="Minimum memory fraction (0.01-1.0) used for initial matrix size.",
    )
    parser.add_argument("--start-memory-fraction", type=float, default=0.02)
    parser.add_argument("--max-memory-fraction", type=float, default=0.08)
    parser.add_argument("--min-matrix-size", type=int, default=2048)
    parser.add_argument("--hard-max-matrix-size", type=int, default=12288)
    parser.add_argument(
        "--pause-memory-threshold-mib",
        type=int,
        default=0,
        help="Deprecated and ignored; pause decisions use per-process GPU SM utilization.",
    )
    parser.add_argument("--ignore-pid", type=int, action="append", default=[])
    parser.add_argument("--ignore-process-name", action="append", default=[])
    parser.add_argument(
        "--idle-trigger-seconds",
        type=float,
        default=DEFAULT_IDLE_TRIGGER_SECONDS,
        help="Trigger burst after this many seconds of all GPUs being at idle util.",
    )
    parser.add_argument(
        "--idle-util-threshold",
        type=int,
        default=DEFAULT_IDLE_UTIL_THRESHOLD,
        help="Treat GPU util at or below this percent as idle.",
    )
    parser.add_argument(
        "--burst-seconds",
        type=float,
        default=DEFAULT_BURST_SECONDS,
        help="How long each burst runs.",
    )
    parser.add_argument(
        "--initial-burst-seconds",
        type=float,
        default=DEFAULT_BURST_SECONDS,
        help="How long to run burst at startup.",
    )
    args = parser.parse_args()
    if not (0 < args.start_memory_fraction <= args.max_memory_fraction < 1):
        raise ValueError("Memory fractions must satisfy 0 < start <= max < 1.")
    if args.memory_budget_mib <= 0:
        raise ValueError("memory-budget-mib must be greater than 0.")
    if not (0 < args.min_memory_budget_fraction <= 1.0):
        raise ValueError("min-memory-budget-fraction must be in (0, 1].")
    if args.target_util <= 0 or args.target_util > 100:
        raise ValueError("target-util must be in the range 1..100.")
    if args.active_seconds <= 0 or args.rest_seconds < 0:
        raise ValueError("active-seconds must be > 0 and rest-seconds must be >= 0.")
    if args.ops_per_batch <= 0:
        raise ValueError("ops-per-batch must be >= 1.")
    if args.sleep_cycles <= 0:
        raise ValueError("sleep-cycles must be > 0.")
    if args.pause_util_threshold < 1 or args.pause_util_threshold > 100:
        raise ValueError("pause-util-threshold must be in the range 1..100.")
    if args.process_util_sample_max_age_seconds <= 0:
        raise ValueError("process-util-sample-max-age-seconds must be > 0.")
    if args.idle_trigger_seconds < 1:
        raise ValueError("idle-trigger-seconds must be >= 1.")
    if args.idle_util_threshold < 0 or args.idle_util_threshold > 100:
        raise ValueError("idle-util-threshold must be in the range 0..100.")
    if args.burst_seconds <= 0:
        raise ValueError("burst-seconds must be > 0.")
    if args.initial_burst_seconds <= 0:
        raise ValueError("initial-burst-seconds must be > 0.")
    return args


def print_status(workers: Sequence[GpuWorker]) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parts = []
    for worker in workers:
        snap = worker.snapshot()
        part = (
            f"GPU{snap.visible_index}(phys={snap.physical_index}) {snap.state} "
            f"util={snap.util}% work_mem={snap.memory_used_bytes / (1024 ** 2):.1f}MiB "
            f"proc_mem={snap.nvml_self_memory_bytes / (1024 ** 2):.0f}MiB"
        )
        if snap.state == "ACTIVE":
            part += f" {snap.note}"
        if snap.other_processes:
            part += f" external=yes [{snap.note}]"
        parts.append(part)
    print(f"[{timestamp}] " + " | ".join(parts), flush=True)


def install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum, _frame) -> None:
        print(f"Received signal {signum}; stopping workers.", flush=True)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


def main() -> None:
    args = parse_args()
    safe_set_torch_cpu_threads()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:
        raise SystemExit(f"Failed to initialize NVML: {exc}")

    torch.backends.cuda.matmul.allow_tf32 = True

    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    try:
        gpu_mapping = visible_to_physical_map()
        if not gpu_mapping:
            raise SystemExit("No visible GPUs detected.")

        baseline_pids = snapshot_visible_gpu_pids(gpu_mapping)
        self_pids = current_process_pid_candidates()
        self_pids.update(
            discover_nvml_self_pids(
                gpu_mapping,
                baseline_pids,
                dtype_from_name(args.dtype),
            )
        )

        print(
            "Launching GPU workers:",
            ", ".join(
                [
                    (
                        f"visible={visible} phys={physical} "
                        f"name={name} uuid={uuid}"
                    )
                    for visible, physical, name, uuid in gpu_mapping
                ]
            ),
            flush=True,
        )
        print(
            f"Self PID candidates: {sorted(self_pids)}",
            flush=True,
        )
        print(
            f"Cycle: Burst mode enabled (startup: {args.initial_burst_seconds}s, repeat: "
            f"{args.burst_seconds}s every {args.idle_trigger_seconds}s below-threshold), "
            f"trigger util <= {args.idle_util_threshold}%, backend={args.backend}, "
            f"workload memory target <= {args.memory_budget_mib} MiB, "
            f"pause when external SM util >= {args.pause_util_threshold}%.",
            flush=True,
        )

        coordinator = BurstCoordinator(
            visible_indices=[visible for visible, _physical, _name, _uuid in gpu_mapping],
            idle_trigger_seconds=args.idle_trigger_seconds,
            burst_seconds=args.burst_seconds,
            initial_burst_seconds=args.initial_burst_seconds,
            idle_util_threshold=args.idle_util_threshold,
        )
        workers = [
            GpuWorker(
                visible,
                physical,
                name,
                uuid,
                args,
                stop_event,
                coordinator,
                self_pids,
            )
            for visible, physical, name, uuid in gpu_mapping
        ]
        for worker in workers:
            worker.start()

        while not stop_event.wait(args.status_seconds):
            print_status(workers)
    finally:
        stop_event.set()
        for thread in threading.enumerate():
            if isinstance(thread, GpuWorker):
                thread.join(timeout=5)
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()