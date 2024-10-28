import torch
import thop
from thop import profile
import time
import argparse
import importlib

def measure_throughput(model, input_tensor, num_runs=100):
    model.eval()
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            model(input_tensor)
        
        start_time = time.time()
        for _ in range(num_runs):
            model(input_tensor)
        end_time = time.time()
        
    throughput = num_runs * input_tensor.size(0) / (end_time - start_time)
    return throughput

def main():
    parser = argparse.ArgumentParser(description="Measure Params, FLOPs, and Throughput of a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model class")
    parser.add_argument("--input_size", type=int, nargs='+', required=True, help="Input size of the model")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs for measuring throughput")
    args = parser.parse_args()

    # Load the trained model
    model_module = importlib.import_module(args.model_name)
    model_class = getattr(model_module, args.model_name.split('.')[-1])
    model = model_class()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Create a random input tensor with the specified input size
    input_tensor = torch.randn(*args.input_size)

    # Measure Params and FLOPs
    macs, params = profile(model, inputs=(input_tensor,))
    print(f"Params: {params}")
    print(f"FLOPs: {macs}")

    # Measure Throughput
    throughput = measure_throughput(model, input_tensor, args.num_runs)
    print(f"Throughput: {throughput} samples/second")

if __name__ == "__main__":
    main()
