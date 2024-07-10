import time 
import os 
from multiprocessing import Pool 

def worker_func(i):
    proc_id = os.getpid()
    print(f"starting job {i} with proc_id {proc_id}", flush=True)
    sched_affinity = os.sched_getaffinity(proc_id)
    print(f"CPUs for job {i} is: {sched_affinity}", flush=True)
    time.sleep(2)
    print(f"job {i} is finished, returning to main", flush=True)


def main():
    print("starting main script", flush=True)
    num_processes = 200
    inputs=range(num_processes)
    start_time = time.time() 
    sched_affinity = os.sched_getaffinity(0)
    print(f"CPUs for main process: {sched_affinity}")
    with Pool(processes=num_processes) as pool:
        for i in pool.imap_unordered(worker_func, inputs):
            print(f"worker {i} finished after {time.time() - start_time} seconds", flush=True)


if __name__ == "__main__":
    main()