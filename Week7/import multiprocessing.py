import multiprocessing
import time

def cpu_stress():
    # An infinite loop that keeps the CPU busy
    while True:
        pass

if __name__ == "__main__":
    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()

    print(f"Stressing {num_cores} CPU cores...")

    # Create a pool of processes, one for each CPU core
    processes = []
    for i in range(num_cores):
        process = multiprocessing.Process(target=cpu_stress)
        processes.append(process)
        process.start()

    # Let it run for a specific time, e.g., 10 seconds
    try:
        time.sleep(1000)
    except KeyboardInterrupt:
        print("Test interrupted.")
    
    # Stop the stress test (this part won't actually be reached unless you stop manually)
    for process in processes:
        process.terminate()

    print("Stress test completed.")
