import psutil
import time
from pathlib import Path
import argparse
import signal
# import pynvml


class Mem:
    def __init__(self, user=[], processes=[]):
        for arg in processes:
            setattr(self, arg, [])
        self.user = user
        self.processes = processes
        self.exit = False
#         pynvml.nvmlInit()
        signal.signal(signal.SIGINT, self._terminate)
        signal.signal(signal.SIGTERM, self._terminate)

    def _terminate(self, *args):
        self.exit = True

    def _get_user_processes(self):
        user_processes = []
        for p in psutil.process_iter():
            try:
                if self.user in p.username().lower():
                    user_processes.append(p)
            except psutil.AccessDenied:
                continue
        return user_processes

    def proc_info(self, fname):
        fname = Path(fname)
        with fname.open("w") as fout:
            fout.write("process,rss,vms,cpu\n")
        while not self.exit:
            for arg in self.processes:
                setattr(self, arg, [])
            for p in psutil.process_iter():
                try:
                    if (
                        (any(name in p.name().lower() for name in self.processes)) and
                        (self.user in p.username().lower())
                    ):
                        getattr(self, p.name().lower().split(".")[0]).append(p)
                except psutil.AccessDenied:
                    continue
                except Exception as e:
                    print(e)
                    continue
            with fname.open("a") as fout:
                for proc in self.processes:
                    rss = []
                    vms = []
                    cpu = []
#                     gpu = []

                    for p in getattr(self, proc):
                        try:
                            rss.append(p.memory_info().rss)
                            vms.append(p.memory_info().vms)
                            cpu.append(p.cpu_percent())
#                             # TODO: Get GPU
#                             gpus = list(range(pynvml.nvmlDeviceGetCount()))
#                             for device_id in gpus:
#                                 hd = pynvml.nvmlDeviceGetHandleByIndex(device_id)
#                                 cps = pynvml.nvmlDeviceGetGraphicsRunningProcesses(hd)
#                                 for ps in cps :
#                                     # check pid values
#                                     if ps.usedGpuMemory:
#                                         gpu.append(ps.usedGpuMemory)
#                                     else:
#                                         gpu.append(0)
                        except Exception as e:
                            # print(e)
                            pass

                    rss = sum(rss) / 1024 / 1024
                    vms = sum(vms) / 1024 / 1024
                    cpu = sum(cpu)
                    # TODO:
#                     gpu = sum(gpu) / 1024 / 1024
                    fout.write(f"{proc},{rss:.2f},{vms:.2f},{cpu:.2f}\n")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to read memory and cpu consumption by user and process."
    )
    parser.add_argument("--user", type=str, default=None, required=True)
    parser.add_argument("--processes", type=str, nargs="*")
    parser.add_argument("--filename", type=str, default="mem_usage")

    args = parser.parse_args()

    user = args.user
    processes = args.processes
    filename = args.filename
    # print(user)
    # print(processes)

    m = Mem(user=user, processes=processes)
    m.proc_info(f"{filename}.txt")
