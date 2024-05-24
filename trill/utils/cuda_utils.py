import subprocess

from openmm.openmm import Platform


def get_available_cuda_devices():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]).decode("utf-8")
        return [str(i) for i in output.strip().split("\n")]
    except subprocess.CalledProcessError:
        return []

def set_platform_properties(args):
    if int(args.GPUs) >= 1:
        available_devices = get_available_cuda_devices()
        num_requested_gpus = int(args.GPUs)

        if num_requested_gpus > len(available_devices):
            print(f"Warning: Requested {num_requested_gpus} GPUs but only {len(available_devices)} are available.")
            num_requested_gpus = len(available_devices)

        selected_devices = ",".join(available_devices[:num_requested_gpus])
        properties = {'DeviceIndex': selected_devices, 'Precision': 'mixed'}
        platform = Platform.getPlatformByName('CUDA')
    else:
        num_threads = args.n_workers  # Number of CPU threads
        properties = {'Threads': str(num_threads)}
        platform = Platform.getPlatformByName('CPU')

    return platform, properties