"""Sample system memory and CUDA free bytes through training and snapshot I/O."""
import json
from pathlib import Path
import threading
import time
import torch


def meminfo():
    return {line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}


class ResourceMonitor:
    def __init__(self, output, *, cuda=True):
        self.output=Path(output)
        self.cuda=cuda and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.init()
            self.initial_gpu_free=torch.cuda.mem_get_info()[0]
            torch.cuda.reset_peak_memory_stats()
        else:
            self.initial_gpu_free=None
        self.minimum_gpu_free=self.initial_gpu_free
        self.minimum_available=meminfo()['MemAvailable']
        self.total_ram=meminfo()['MemTotal']
        self.peak_rss=0
        self.started=time.monotonic()
        self.stop=threading.Event()
        self.samples=0
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.errors=[]

    def sample(self):
        self.samples+=1
        self.minimum_available=min(self.minimum_available,meminfo()['MemAvailable'])
        rss=next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        self.peak_rss=max(self.peak_rss,int(rss.split()[1])*1024)
        if self.cuda:
            self.minimum_gpu_free=min(self.minimum_gpu_free,torch.cuda.mem_get_info()[0])

    def run(self):
        while not self.stop.wait(.2):
            try:
                self.sample()
            except Exception as exc:
                self.errors.append(str(exc))
                break

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop.set()
        self.thread.join()
        self.sample()
        result={'sample_period_seconds':.2,'samples':self.samples,'errors':self.errors,
            'elapsed_seconds':time.monotonic()-self.started,'system_ram_total_bytes':self.total_ram,
            'system_memavailable_min_bytes':self.minimum_available,'process_rss_peak_sampled_bytes':self.peak_rss,
            'gpu_initial_free_bytes':self.initial_gpu_free,'gpu_min_free_bytes':self.minimum_gpu_free,
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated() if self.cuda else None,
            'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved() if self.cuda else None,
            'cpu_margin_pass':self.minimum_available>=.2*self.total_ram,
            'gpu_margin_pass':self.minimum_gpu_free>=.15*self.initial_gpu_free if self.cuda else None,
            'sampling_limit':'Samples may miss short transients; allocator peaks are reported separately.'}
        self.output.write_text(json.dumps(result,indent=2)+'\n')
