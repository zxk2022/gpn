from datasets import ShapeNetSkelDataset, ModelNet10Dataset, ModelNet40Dataset, ScanObjectNNDataset
from sklearn.model_selection import train_test_split

def get_datasets(data_dir, graph_type, use_bg=True):
    if 'ModelNet' in data_dir:
        dataset_class = globals()[f'{data_dir.split("/")[-1]}Dataset']
        return (
            dataset_class(root=data_dir, graph_type=graph_type, train=True),
            dataset_class(root=data_dir, graph_type=graph_type, train=False),
            dataset_class(root=data_dir, graph_type=graph_type, train=False)
        )
    elif data_dir == 'data/ScanObjectNN':
        dataset_class = ScanObjectNNDataset
        return (
            dataset_class(root=data_dir, graph_type=graph_type, train=True, use_bg=use_bg),
            dataset_class(root=data_dir, graph_type=graph_type, train=False, use_bg=use_bg),
            dataset_class(root=data_dir, graph_type=graph_type, train=False, use_bg=use_bg)
        )
    elif data_dir == 'data/SHAPENET_SKEL':
        dataset = ShapeNetSkelDataset(root=data_dir, graph_type=graph_type)
        # 切分数据集为训练集和测试集
        train_val_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        # 再次切分训练集为训练集和验证集
        train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.25, random_state=42)
        return train_dataset, val_dataset, test_dataset
    else:
        raise ValueError(f"Unsupported data directory: {data_dir}")


# 定义不同的图类型
graph_types = ['knn', 'gabriel', 'rng']

# 定义不同的背景使用选项
use_bg_options = [True, False]

# 调用 get_datasets 函数获取不同的数据集
for graph_type in graph_types:
    for use_bg in use_bg_options:
        # ModelNet10 数据集
        modelnet10_data_dir = 'data/ModelNet10'
        train_modelnet10, val_modelnet10, test_modelnet10 = get_datasets(modelnet10_data_dir, graph_type=graph_type, use_bg=use_bg)
        
        # ModelNet40 数据集
        modelnet40_data_dir = 'data/ModelNet40'
        train_modelnet40, val_modelnet40, test_modelnet40 = get_datasets(modelnet40_data_dir, graph_type=graph_type, use_bg=use_bg)
        
        # ScanObjectNN 数据集
        scanobjectnn_data_dir = 'data/ScanObjectNN'
        train_scanobjectnn, val_scanobjectnn, test_scanobjectnn = get_datasets(scanobjectnn_data_dir, graph_type=graph_type, use_bg=use_bg)
        
        # ShapeNetSkel 数据集
        shapenetskel_data_dir = 'data/SHAPENET_SKEL'
        train_shapenetskel, val_shapenetskel, test_shapenetskel = get_datasets(shapenetskel_data_dir, graph_type=graph_type, use_bg=use_bg)
        
        # 打印每个数据集的信息以确认是否成功加载
        print(f"Graph Type: {graph_type}, Use BG: {use_bg}")
        print("ModelNet10 Datasets:", len(train_modelnet10), len(val_modelnet10), len(test_modelnet10))
        print("ModelNet40 Datasets:", len(train_modelnet40), len(val_modelnet40), len(test_modelnet40))
        print("ScanObjectNN Datasets:", len(train_scanobjectnn), len(val_scanobjectnn), len(test_scanobjectnn))
        print("ShapeNetSkel Datasets:", len(train_shapenetskel), len(val_shapenetskel), len(test_shapenetskel))
        print("-" * 80)