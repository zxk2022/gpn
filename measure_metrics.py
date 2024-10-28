import torch
import thop
from thop import profile
import time
import argparse
import importlib
import yaml
import sys
from models import model_mapping
from datasets import ModelNet40Dataset

def parse_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, help='Path to the YAML file')
    parser.add_argument('--model', type=str, help='Name of the model')
    parser.add_argument('--type', type=int, help='Type of training')
    parser.add_argument('--data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--use_bg', type=bool, help='Use background points')
    parser.add_argument('--output', type=str, help='Path to the log file')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    parser.add_argument('--keys', type=str, help='The input characteristic parameter is')
    parser.add_argument('--graph_type', type=str, help='The type of graph')
    parser.add_argument('--undirected', type=bool, help='Convert the graph to an undirected graph')
    args = parser.parse_args()

    args.model = config.get('model')
    args.type = config.get('type')
    args.data_dir = config.get('data_dir')
    args.use_bg = config.get('use_bg')
    args.output = config.get('output')
    args.epochs = config.get('epochs')
    args.batch = config.get('batch')
    args.lr = config.get('lr')
    args.min_lr = config.get('min_lr')
    args.dropout = config.get('dropout')
    args.keys = config.get('keys')
    args.graph_type = config.get('graph_type')
    args.undirected = config.get('undirected')
    args.configs = config.get('configs')

    return args

def measure_throughput(model, input_tensor, num_runs=100):
    model.eval()
    with torch.no_grad():
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
    parser.add_argument("--input_size", type=int, nargs='+', required=True, help="Input size of the model")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of runs for measuring throughput")
    parser.add_argument('--cfgs', type=str, help='Path to the YAML file')
    args = parser.parse_args()

    args = parse_yaml_config(args.cfgs)
    args.configs = args.configs[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_mapping[args.model](3, 40, args.configs).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    dataset = ModelNet40Dataset(root=args.data_dir, graph_type=args.graph_type, train=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    input_tensor = next(iter(data_loader)).to(device)

    macs, params = profile(model, inputs=(input_tensor,))
    print(f"Params: {params}")
    print(f"FLOPs: {macs}")

    throughput = measure_throughput(model, input_tensor, args.num_runs)
    print(f"Throughput: {throughput} samples/second")

if __name__ == "__main__":
    main()
