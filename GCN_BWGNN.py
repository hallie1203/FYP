import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.utils import subgraph, negative_sampling
from torch_geometric.nn import GCNConv, GAE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Import the BWGNN model from your file.
from bwgnn_model import BWGNN as BWGNN


# ------------------------
# GCN for Node Classification
# ------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return x


# ------------------------
# Encoder for Graph Auto-Encoder (GAE)
# ------------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


# ------------------------
# Training and Evaluation Functions for GCN
# ------------------------
def train_gcn(data, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_gcn(data, model, mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    correct = (pred[mask] == data.y[mask]).sum()
    accuracy = correct / mask.sum().item()
    return accuracy, recall, f1


# ------------------------
# Revised Training and Evaluation Functions for BWGNN (operating on DGL graphs)
# ------------------------
def train_bwgnn(g, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(g.ndata['feature'])
    loss = criterion(out[g.ndata['train_mask']], g.ndata['label'][g.ndata['train_mask']])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_bwgnn(g, model):
    model.eval()
    out = model(g.ndata['feature'])
    pred = out.argmax(dim=1)
    y_true = g.ndata['label'][g.ndata['test_mask']].cpu().numpy()
    y_pred = pred[g.ndata['test_mask']].cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    correct = (pred[g.ndata['test_mask']] == g.ndata['label'][g.ndata['test_mask']]).sum()
    accuracy = correct / g.ndata['test_mask'].sum().item()
    return accuracy, recall, f1


# ------------------------
# Helper function to convert torch-geometric data to a DGL graph
# ------------------------
def pyg_to_dgl(data):
    import dgl
    g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
    g.ndata['feature'] = data.x
    g.ndata['label'] = data.y
    g.ndata['train_mask'] = data.train_mask
    g.ndata['val_mask'] = data.val_mask
    g.ndata['test_mask'] = data.test_mask
    return g


# ------------------------
# Visualization Code for Node Embeddings
# ------------------------
def visualize_embeddings(gcn_model, bwgnn_model, data, dgl_graph):
    with torch.no_grad():
        # For GCN, use the output (logits) as embeddings.
        emb_gcn = gcn_model(data)
        # For BWGNN, pass the node features from the DGL graph.
        emb_bwgnn = bwgnn_model(dgl_graph.ndata['feature'])

    # Randomly select nodes for visualization (or use all if less than sample_size)
    num_nodes = emb_gcn.size(0)
    sample_size = 10000 if num_nodes > 10000 else num_nodes
    indices = np.random.choice(num_nodes, sample_size, replace=False)
    emb_gcn_sample = emb_gcn.cpu().numpy()[indices]
    emb_bwgnn_sample = emb_bwgnn.cpu().numpy()[indices]
    labels_sample = data.y.cpu().numpy()[indices]

    # Reduce dimensions using t-SNE.
    tsne = TSNE(n_components=2, random_state=42)
    emb_gcn_2d = tsne.fit_transform(emb_gcn_sample)
    emb_bwgnn_2d = tsne.fit_transform(emb_bwgnn_sample)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(emb_gcn_2d[:, 0], emb_gcn_2d[:, 1], c=labels_sample, cmap='coolwarm', s=10)
    plt.title("GCN Embeddings (Node Classification)")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(emb_bwgnn_2d[:, 0], emb_bwgnn_2d[:, 1], c=labels_sample, cmap='coolwarm', s=10)
    plt.title("BWGNN Embeddings")
    plt.colorbar()
    plt.show()


# ------------------------
# Main Function: Load, Train, Evaluate and Visualize
# ------------------------
def main():
    # Load dataset using PyTorch Geometric
    dataset = EllipticBitcoinDataset(root='data/EllipticBitcoin')
    data = dataset[0]
    print("Original Dataset Summary:")
    print(data)

    # Filter out nodes with label 2 (unknown)
    num_valid = data.x.size(0)
    valid_nodes = torch.arange(num_valid)
    labels_valid = data.y[:num_valid]
    mask = labels_valid != 2
    subset = valid_nodes[mask]
    print(f"Number of nodes with features before filtering: {num_valid}")
    print(f"Number of nodes after filtering (drop label 2): {subset.size(0)}")

    data.x = data.x[subset]
    data.y = labels_valid[mask]
    data.edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=num_valid)
    data.num_nodes = subset.size(0)

    # Train/Validation/Test Masks
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:int(0.4 * num_nodes)]] = True
    val_mask[indices[int(0.4 * num_nodes):int(0.7 * num_nodes)]] = True
    test_mask[indices[int(0.7 * num_nodes):]] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # GCN Model
    gcn_model = GCN(data.num_node_features, hidden_channels=16, out_channels=2)
    optimizer_gcn = torch.optim.Adam(gcn_model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("\nTraining GCN for Node Classification...")
    for epoch in range(1, 101):
        loss = train_gcn(data, gcn_model, optimizer_gcn, criterion)
        if epoch % 50 == 0:
            val_acc, val_recall, val_f1 = evaluate_gcn(data, gcn_model, data.val_mask)
            print(f"GCN Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

    # DGL Graph for BWGNN
    dgl_graph = pyg_to_dgl(data)

    # BWGNN Model
    bwgnn_model = BWGNN(data.num_node_features, 16, 2, graph=dgl_graph)
    optimizer_bwgnn = torch.optim.Adam(bwgnn_model.parameters(), lr=0.01, weight_decay=5e-4)

    print("\nTraining BWGNN for Node Classification...")
    for epoch in range(1, 501):
        loss = train_bwgnn(dgl_graph, bwgnn_model, optimizer_bwgnn, criterion)
        if epoch % 50 == 0:
            val_acc, val_recall, val_f1 = evaluate_bwgnn(dgl_graph, bwgnn_model)
            print(f"BWGNN Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

    # Visualization of embeddings
    visualize_embeddings(gcn_model, bwgnn_model, data, dgl_graph)


if __name__ == '__main__':
    main()
