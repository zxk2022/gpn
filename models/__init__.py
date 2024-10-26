from .GIN import GINNet
from .GCN import GCNNet
from .GAT import GATNet
from .Edge import EdgeNet
from .MR import MRNet

model_mapping = {
    'GINNet': GINNet,
    'GCNNet': GCNNet,
    'GATNet': GATNet,
    'EdgeNet': EdgeNet,
    'MRNet': MRNet
}