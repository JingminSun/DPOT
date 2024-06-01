import pynvml
import time
import subprocess

def get_gpu_time(handle):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu

def monitor_gpu_usage(command, max_gpu_hours):
    # Convert hours to seconds
    max_gpu_seconds = max_gpu_hours * 3600

    # Initialize NVML
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU

    # Start the subprocess
    process = subprocess.Popen(command, shell=True)
    start_time = time.time()

    try:
        while True:
            elapsed_time = time.time() - start_time
            gpu_time = get_gpu_time(handle)

            if elapsed_time > max_gpu_seconds:
                print(f"Terminating process {process.pid} after {max_gpu_hours} GPU hours")
                process.terminate()
                break

            if process.poll() is not None:
                break

            time.sleep(10)  # Sleep for 10 seconds before checking again

    except KeyboardInterrupt:
        print("Interrupted by user")
        process.terminate()

    process.wait()
    pynvml.nvmlShutdown()

# Example command to run your script
command = "python train_temporal.py --model DPOT --gpu 1 --use_writer --log_path test --comment M"

# Set the maximum GPU hours
max_gpu_hours = 48  # 1 GPU hour

# Run the monitoring function
monitor_gpu_usage(command, max_gpu_hours)
