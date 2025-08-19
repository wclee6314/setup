import torch
import time
import threading
import pynvml

# Initialize NVML to monitor GPU usage
pynvml.nvmlInit()

def get_gpu_handle():
    return pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0

def get_gpu_usage():
    handle = get_gpu_handle()
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

def gpu_task(duration_minutes=5, target_usage=0.1):
    start_time = time.time()
    end_time = start_time + duration_minutes * 60

    # Generate random tensor to keep GPU busy
    tensor_size = int(100000 * target_usage)  # Adjust tensor size to simulate GPU usage
    while time.time() < end_time:
        # Perform a dummy operation to keep the GPU active
        a = torch.rand((tensor_size, tensor_size), device='cuda')
        b = torch.matmul(a, a)
        del b
        time.sleep(0.1)  # Small delay to control GPU utilization

def gpu_utilization_scheduler(interval_minutes=60, duration_minutes=5, target_usage=0.1):
    while True:
        print(f"Starting GPU task for {duration_minutes} minutes.")
        gpu_task(duration_minutes, target_usage)
        print(f"GPU task completed. Next run in {interval_minutes} minutes.")
        time.sleep(interval_minutes * 60)  # Wait for the specified interval

if __name__ == "__main__":
    # Use threading to avoid blocking the main program
    scheduler_thread = threading.Thread(target=gpu_utilization_scheduler, args=(60, 5, 0.1))
    scheduler_thread.daemon = True
    scheduler_thread.start()

    try:
        # Monitor GPU usage in the main thread
        while True:
            gpu_usage = get_gpu_usage()
            print(f"Current GPU usage: {gpu_usage}%")
            time.sleep(10)  # Check GPU usage every 10 seconds
    except KeyboardInterrupt:
        print("Terminating GPU utilization script.")
    finally:
        pynvml.nvmlShutdown()