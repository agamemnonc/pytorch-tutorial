import torch
from torch_geometric.data import Data

#########################
# 1. Defining the Graph #
#########################

# Edges are defined using the following notation
# In this example there are 4 edges:
# (0, 1), (1, 0), (1, 2), (2, 1)
# We are using 2 edges per connection to account for both directions of an
# edge.
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
# The input data has the node features (in this case 1D)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
# Displays the dimensions of edges and node features
print(data)

#########################
# 2. Benchmark datasets #
#########################

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid

dataset_tud = TUDataset(root='../../data/ENZYMES', name='ENZYMES')
dataset_plan = Planetoid(root='../../data/Cora', name='Cora')

###################
# 3. Mini-batches #
###################

from torch_geometric.data import DataLoader
dataset = TUDataset(root='../../data/ENZYMES', name='ENZYMES',
                        use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    batch


####################
4. Data transforms #
####################

from torch_geometric.datasets import ShapeNet
from torch_geometric import transforms as T
dataset = ShapeNet(root='../../data/ShapeNet', categories=['Airplane'],
                   pre_transform=T.KNNGraph(k=6))

################################
# 5. Leaning Methods on Graphs #
################################
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='../../data/Cora', name='Cora')

# Define network
class Net(torch.nn.Module):
    def __init__(self, num_hidden_nodes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, num_hidden_nodes)
        self.conv2 = GCNConv(num_hidden_nodes, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

# Set-up device and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_hidden_nodes=16).to(device)
data = dataset[0].to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(data)
    loss_train = criterion(out[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()
    if epoch % 10 == 0:
        loss_val = criterion(out[data.val_mask], data.y[data.val_mask])
        _, pred = out.max(dim=1)
        acc_train = pred[data.train_mask].eq(data.y[
            data.train_mask]).sum().item() / data.train_mask.sum().item()
        acc_val = pred[data.val_mask].eq(data.y[
            data.val_mask]).sum().item() / data.val_mask.sum().item()
        print("Epoch {}, train loss {:.3f}, val loss {:.3f}, " \
              "train acc {:.3f}, val_acc {:.3f}".format(
            epoch, loss_train, loss_val, acc_train, acc_val
        ))

