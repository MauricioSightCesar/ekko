import time
import numpy as np
import torch

def inference_time_cpu(model, input_data, n_reps=500):
    """Compute a pytorch model inference time in a cpu device"""

    model.to("cpu")
    input_data = input_data.to("cpu")

    starter, ender = 0, 0
    timings = np.zeros((n_reps, 1))

    with torch.no_grad():
        for rep in range(n_reps):
            starter = time.time()
            _ = model(input_data)
            ender = time.time()

            elapsed_time = (ender - starter) * 1000  # ms
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)

    return mean_inference_time

def pytorch_inference_time_gpu(model, input_data, n_reps=500, n_gpu_warmups=100):
    """Compute a pytorch model inference time in a gpu device"""
    # References:
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks

    # https://discuss.pytorch.org/t/elapsed-time-units/29951 (time in milliseconds)

    model.to("cuda")
    input_data = input_data.to("cuda")

    # Init timer loggers
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((n_reps, 1))

    # GPU Warm-up
    for _ in range(n_gpu_warmups):
        _ = model(input_data)

    # Measure performance
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            _ = model(input_data)
            ender.record()
            # Wait for gpu to sync
            torch.cuda.synchronize()
            elapsed_time = starter.elapsed_time(ender) # ms
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)

    return mean_inference_time

def pytorch_inference_time_mps(model, input_data, n_reps=500, n_gpu_warmups=100):
    """Compute a pytorch model inference time in a mps device"""
    # References:
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks

    # https://discuss.pytorch.org/t/elapsed-time-units/29951 (time in milliseconds)

    model.to("mps")
    input_data = input_data.to("mps")

    # Init timer loggers
    starter, ender = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
    timings = np.zeros((n_reps, 1))

    # GPU Warm-up
    for _ in range(n_gpu_warmups):
        _ = model(input_data)

    # Measure performance
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            _ = model(input_data)
            ender.record()
            # Wait for gpu to sync
            torch.mps.synchronize()
            elapsed_time = starter.elapsed_time(ender) # ms
            timings[rep] = elapsed_time

    mean_inference_time = np.mean(timings)

    return mean_inference_time

def get_inference_time(model, dummy_input):
    """Get the inference time of a model from all devices"""
    cpu_inference_time = None
    gpu_inference_time = None
    mps_inference_time = None

    # Check if the model is on GPU
    if torch.backends.mps.is_available():
        mps_inference_time = pytorch_inference_time_mps(model, dummy_input)
    
    if torch.cuda.is_available():
        gpu_inference_time = pytorch_inference_time_gpu(model, dummy_input)

    cpu_inference_time = inference_time_cpu(model, dummy_input)

    return cpu_inference_time, gpu_inference_time, mps_inference_time

def pytorch_compute_model_size_mb(model):
    """Compute a pytorch model size in megabytes"""
    # Reference
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_param_buffer_mb = (param_size + buffer_size) / (1024) ** 2

    return size_param_buffer_mb

def get_resource_metrics(model, test_loader):
    first_batch = next(iter(test_loader))
    data = first_batch[0]
    dummy_input = torch.randn_like(data)
    cpu_inference_time, gpu_inference_time, mps_inference_time = get_inference_time(model, dummy_input)
    model_size_mb = pytorch_compute_model_size_mb(model)
    resource_metrics = {
        'cpu_inference_time': cpu_inference_time,
        'gpu_inference_time': gpu_inference_time,
        'mps_inference_time': mps_inference_time,
        'model_size_mb': model_size_mb
    }
    return resource_metrics
