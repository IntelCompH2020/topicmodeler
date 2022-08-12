import psutil
import time
from pathlib import Path
import argparse
import signal
import pynvml
from socket import gethostname


class Mem:
    use_gpu = False

    def __init__(self, user=[], processes=[], gpu=False):
        for arg in processes:
            setattr(self, arg, [])
        self.user = user
        self.processes = processes
        self.exit = False
        if gpu:
            try:
                pynvml.nvmlInit()
                self.use_gpu = True
            except:
                # Manage use in different OS
                pass

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
        # Check if there are gpus
        if self.use_gpu:
            # try:
            gpus = list(range(pynvml.nvmlDeviceGetCount()))
        # except:
        #     # Manage use in different OS
        #     gpus = []
        #     pass

        # Generate file
        fname = Path(fname)
        line = "process,rss,vms,cpu"
        if self.use_gpu:
            line += ",gpu_memory,gpu"
        with fname.open("w") as fout:
            fout.write(f"{line}\n")

        # Keep measuring
        hname = gethostname().lower()
        while not self.exit:
            for arg in self.processes:
                setattr(self, arg, [])
            for p in psutil.process_iter():
                try:
                    pname = p.name().lower().split(".")[0]
                    puser = p.username().lower().replace(hname, "").strip("\\").strip("/")
                    for name in self.processes:
                        # print(name, pname, self.user, puser)
                        if (pname in name or name in pname) and self.user in puser:
                            getattr(self, name).append(p)
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
                    if self.use_gpu:
                        gpu_mem = []
                        gpu = []

                    for p in getattr(self, proc):
                        try:
                            rss.append(p.memory_info().rss)
                            vms.append(p.memory_info().vms)
                            cpu.append(p.cpu_percent())
                            if self.use_gpu:
                                # # Get GPU
                                # if not gpus:
                                #     gpu_mem.append(0)
                                #     gpu.append(0)
                                # else:
                                for device_id in gpus:
                                    hd = pynvml.nvmlDeviceGetHandleByIndex(
                                        device_id)
                                    use = pynvml.nvmlDeviceGetUtilizationRates(
                                        hd).gpu
                                    gpu_ps = pynvml.nvmlDeviceGetComputeRunningProcesses(
                                        hd)
                                    gpu.append(use)
                                    for gp in gpu_ps:
                                        # TODO: check pid values
                                        if gp.usedGpuMemory and gp.pid == p.pid:
                                            gpu_mem.append(gp.usedGpuMemory)
                                        else:
                                            gpu_mem.append(0)

                        except Exception as e:
                            # print(e)
                            pass

                    rss = sum(rss) / 1024 / 1024
                    vms = sum(vms) / 1024 / 1024
                    cpu = sum(cpu)
                    line = f"{proc},{rss:.2f},{vms:.2f},{cpu:.2f}"
                    if self.use_gpu:
                        gpu_mem = sum(gpu_mem) / 1024 / 1024
                        gpu = sum(gpu)
                        line += f",{gpu_mem:.2f},{gpu:.2f}"
                    fout.write(f"{line}\n")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to read memory and cpu consumption by user and process."
    )
    parser.add_argument("--user", type=str, default=None,
                        required=True, help="Measure information from this user.")
    parser.add_argument("--processes", type=str, nargs="*",
                        help="Measure a list of processes.")
    parser.add_argument("--gpu", type=bool, default=False,
                        help="Measure GPU usage.")
    parser.add_argument("--filename", type=str, default="mem_usage",
                        help="Filename to save information.")

    args = parser.parse_args()

    user = args.user
    processes = args.processes
    filename = args.filename
    gpu = args.gpu
    # print(user)
    # print(processes)

    m = Mem(user=user, processes=processes, gpu=gpu)
    m.proc_info(f"{filename}.txt")
