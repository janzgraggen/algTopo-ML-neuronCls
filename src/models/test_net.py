
from torch.nn import functional as nnf
from torch_geometric.nn import ChebConv, GCNConv
import torch
from morphoclass import layers 

class TestNet(torch.nn.Module):

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv_1 = ChebConv(n_features, 128, K=5)
        self.conv_2 = GCNConv(128, 256)
        self.conv_3 = GCNConv(256, 512)
        self.pool = layers.AttentionGlobalPool(512)
        self.fc = torch.nn.Linear(512, n_classes)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv_1(x, edge_index)
        x = nnf.relu(x)
        x = self.conv_2(x, edge_index)
        x = nnf.relu(x)
        x = self.conv_3(x, edge_index)
        x = nnf.relu(x)
        x = self.pool(x, data.batch)
        x = self.fc(x)
        x = nnf.log_softmax(x, dim=1)

        return x
