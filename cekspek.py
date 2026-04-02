import platform
import sys
import os

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def get_cpu_name():
    if platform.system() == 'Windows':
        return platform.processor() or 'Unknown'
    # macOS/Linux
    try:
        if platform.system() == 'Darwin':
            import subprocess
            out = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
            return out
        # linux
        with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as cpuinfo:
            for line in cpuinfo:
                if 'model name' in line:
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'Unknown'


def get_ram_size_gb():
    if psutil is not None:
        return psutil.virtual_memory().total / (1024 ** 3)
    if hasattr(os, 'sysconf') and 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return pages * page_size / (1024 ** 3)
    return None


def get_gpu_info():
    gpus = []
    if torch is not None and torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            try:
                gpus.append(torch.cuda.get_device_name(i))
            except Exception:
                gpus.append(f'CUDA device #{i} (name unknown)')
    else:
        try:
            import subprocess
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, stderr=subprocess.DEVNULL)
            for line in out.splitlines():
                line = line.strip()
                if line:
                    gpus.append(line)
        except Exception:
            pass
    return gpus


def main():
    print('=== System info ===')
    print('OS:', f"{platform.system()} {platform.release()} ({platform.version()})")
    print('Platform:', platform.platform())
    print('Python:', sys.version.replace('\n', ' '))

    if torch is not None:
        print('PyTorch:', torch.__version__)
        cuda_available = torch.cuda.is_available()
        print('CUDA available:', cuda_available)
        print('CUDA version (torch):', torch.version.cuda)
    else:
        print('PyTorch: not installed')
        cuda_available = False

    gpu_names = get_gpu_info()
    if gpu_names:
        for i, name in enumerate(gpu_names, 1):
            print(f'GPU #{i}:', name)
    else:
        print('GPU: none detected or not accessible')

    cpu_name = get_cpu_name()
    print('CPU:', cpu_name)

    ram_gb = get_ram_size_gb()
    if ram_gb is not None:
        print(f'RAM: {ram_gb:.2f} GB')
    else:
        print('RAM: unknown (install psutil for more accurate info)')

    print('--- Additional details ---')
    print('Machine:', platform.machine())
    print('Processor info (platform):', platform.processor())


if __name__ == '__main__':
    main()
