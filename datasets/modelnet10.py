import os
import os.path as osp
from typing import List
from torch_geometric.data import InMemoryDataset, Data
import torch
from tqdm import tqdm
from .data_utils import parse_obj

class ModelNet10Dataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        graph_type: str,
        train: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.graph_type = graph_type
        self.train = train
        super(ModelNet10Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # 返回训练或测试文件夹下所有子文件夹的名称列表
        return [osp.join(self.raw_dir, f'modelnet10_{"train" if self.train else "test"}_1024pts_fps_pcd', folder) 
                for folder in os.listdir(osp.join(self.raw_dir, f'modelnet10_{"train" if self.train else "test"}_1024pts_fps_pcd')) 
                if osp.isdir(osp.join(self.raw_dir, f'modelnet10_{"train" if self.train else "test"}_1024pts_fps_pcd', folder))]

    @property
    def processed_file_names(self) -> str:
        return f'data_{"train" if self.train else "test"}_{self.graph_type}.pt'

    def download(self):
        pass

    def process(self):
        # 初始化一个空列表来保存所有图形数据
        data_list = []
        len_folder = len(self.raw_file_names)
        i = 1

        # 遍历每个类别文件夹
        for category_folder in self.raw_file_names:
            # 获取该类别下的所有.obj文件
            obj_files = [f for f in os.listdir(category_folder) if f.endswith(f'_{self.graph_type}.obj')]

            # 对每个文件进行处理
            for obj_file in tqdm(obj_files, desc=f'{i} / {len_folder} folder', leave=False):
                obj_path = osp.join(category_folder, obj_file)

                pos, edge_index = parse_obj(obj_path)
                label = int(osp.basename(category_folder))
                data = Data(x=pos, edge_index=edge_index, y=label)
                
                # 应用预处理
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                # 添加到数据列表中
                data_list.append(data)
            i += 1

        # 将所有数据对象组合成一个大对象，并保存到磁盘
        if len(data_list) > 0:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
