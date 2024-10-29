import torch
import torch.nn as nn
import torch.optim as optim

from datasets import ShapeNetSkelDataset, ModelNet10Dataset, ModelNet40Dataset, ScanObjectNNDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from py_utils.train_script import train_model, save_results_to_csv
from py_utils.metrics import measure_model

from models import model_mapping
import argparse
import datetime
import os

import argparse
import yaml
import sys

def parse_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str, help='Path to the YAML file')
    parser.add_argument('--metric', type=bool, default=False, help='Performance Measurement')
    parser.add_argument('--model_path', type=str, help='Path to the model')
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

    # 从 YAML 文件中读取配置
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

args = parse_yaml_config(sys.argv[2])
args.configs = args.configs[0]

def get_datasets(args):
    data_dir = args.data_dir
    graph_type = args.graph_type
    if 'ModelNet' in data_dir:
        dataset_class = globals()[f'{data_dir.split("/")[-1]}Dataset']
        return (
            dataset_class(root=data_dir, graph_type=graph_type, train=True),
            dataset_class(root=data_dir, graph_type=graph_type, train=False),
            dataset_class(root=data_dir, graph_type=graph_type, train=False)
        )
    elif 'ScanObjectNN' in data_dir:
        use_bg = args.use_bg
        dataset_class = ScanObjectNNDataset
        return (
            dataset_class(root=data_dir, graph_type=graph_type, train=True, use_bg=use_bg),
            dataset_class(root=data_dir, graph_type=graph_type, train=False, use_bg=use_bg),
            dataset_class(root=data_dir, graph_type=graph_type, train=False, use_bg=use_bg)
        )
    elif 'ShapeNetSkel' in data_dir:
        dataset = ShapeNetSkelDataset(root=data_dir, graph_type=graph_type)
        # 切分数据集为训练集和测试集
        train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        # 再次切分训练集为训练集和验证集
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.25, random_state=42)
        return train_dataset, val_dataset, test_dataset
    else:
        raise ValueError(f"Unsupported data directory: {data_dir}")

# 使用示例
train_dataset, val_dataset, test_dataset = get_datasets(args)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

# Generate a timestamp for the file name
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

if args.metric:
    args.output = args.output + '/' + 'metrics' + '/' + f"{args.model}_" + timestamp
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_mapping[args.model](train_dataset.num_features, train_dataset.num_classes, args.configs).to(device)
    model.load_state_dict(torch.load(args.model_path))
    result = measure_model(model, train_loader, args=args, use_mp_tr=True)

else:
    # 检查输出目录是否存在，如果不存在则创建
    args.output = args.output + '/' + f"{args.model}_" + timestamp
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 重复实验
    for i in range(5):
        # 检查是否有可用的GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        model = model_mapping[args.model](train_dataset.num_features, train_dataset.num_classes, args.configs).to(device)

        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        T_max = args.epochs  
        eta_min = args.min_lr  # 最小学习率为0

        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()

        components = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'optimizer': optimizer,
            'criterion': criterion,
            'device': device,
            'scheduler': scheduler,
        }

        print(args.__dict__)
        print(f"Model: {args.model}")
        print(f"Type: {args.type}")
        print(f"Data: {args.data_dir}")
        print(f"Number of features: {train_dataset.num_features}")
        print(f"Number of classes: {train_dataset.num_classes}")
        print(f"Device: {device}")
        print(f"Model architecture:\n{model}")
        print(f"Optimizer: {optimizer}")
        print(f"Criterion: {criterion}")
        type = args.type

        # 构建文件路径
        file_path = f"{args.output}/cfg.txt"

        # 打开文件，如果文件不存在则创建，如果存在则追加
        with open(file_path, 'a') as file:
            # 写入打印内容
            file.write(str(args.__dict__) + '\n')
            file.write(f"Number of features: {train_dataset.num_features}\n")
            file.write(f"Number of classes: {train_dataset.num_classes}\n")
            file.write(f"Device: {device}\n")
            file.write(f"Model architecture:\n{model}\n")
            file.write(f"Optimizer: {optimizer}\n")
            file.write(f"Criterion: {criterion}\n")
            file.write(f"lr: {args.lr}\n")
            file.write(f"min_lr: {args.min_lr}\n")

        if type == 0:
            # 训练模型
            result = train_model(**components, args=args, num_exp=i, use_mp_tr=True, use_mp_val=True, use_mp_te=True)
        elif type == 1:
            # 训练模型
            result = train_model(**components, args=args, num_exp=i, use_mp_tr=False, use_mp_val=False, use_mp_te=True)
        elif type == 2:
            # 训练模型
            result = train_model(**components, args=args, num_exp=i, use_mp_tr=False, use_mp_val=False, use_mp_te=False)
        else:
            assert False, 'Invalid type'

        # 手动格式化打印
        print("\n")
        print(f"------------------------------result of exp{i}------------------------------------")
        print("Best Validation OA: {:.4f}".format(result['Best Validation OA']))
        print("Best Validation MACC: {:.4f}".format(result['Best Validation MACC']))
        print("Chosen Epoch: {}".format(result['Chosen Epoch']))
        print("Chosen Loss: {:.4f}".format(result['Chosen Loss']))
        print("Chosen Train Accuracy: {:.4f}".format(result['Chosen Train Accuracy']))
        print("Chosen Val OA: {:.4f}".format(result['Chosen Val OA']))
        print("Chosen Val MACC: {:.4f}".format(result['Chosen Val MACC']))
        print("Final Epoch Val OA: {:.4f}".format(result['Final Epoch Val OA']))
        print("Final Epoch Val MACC: {:.4f}".format(result['Final Epoch Val MACC']))

        print("Test OA: {:.4f}".format(result['Test OA']))
        print("Test MACC: {:.4f}".format(result['Test MACC']))
        print("Final Epoch Test OA: {:.4f}".format(result['Final Epoch Test OA']))
        print("Final Epoch Test MACC: {:.4f}".format(result['Final Epoch Test MACC']))
        print("Total Time: {:.2f}s".format(result['Total Time']))
        print(f"------------------------------result of exp{i}------------------------------------")
        print("\n")


        # Create the file name with the timestamp
        file_name = f"{args.output}/results.csv"

        # Save the results to a CSV file
        save_results_to_csv(result, file_name)