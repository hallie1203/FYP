import numpy as np
import torch
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
import dgl.nn.pytorch.conv as dglnn


class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,
                                 aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        print(h_final)
        return h_final


class BaseDetector(object):
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        # Set the input feature size.
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        graph = self.data.graph.to(self.train_config['device'])
        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].bool()
        self.val_mask = graph.ndata['val_mask'].bool()
        self.test_mask = graph.ndata['test_mask'].bool()

        # Since we dropped unknown nodes before, no further filtering is needed.
        # Compute weight based on valid training nodes (assuming labels are 0 and 1).
        num_pos = self.labels[self.train_mask].sum().item()
        num_neg = (1 - self.labels[self.train_mask]).sum().item()
        if num_pos == 0 or num_neg == 0:
            self.weight = 1.0
        else:
            self.weight = num_neg / num_pos

        self.source_graph = graph
        print(train_config['inductive'])
        if train_config['inductive'] == False:
            self.train_graph = graph
            self.val_graph = graph
        else:
            self.train_graph = graph.subgraph(self.train_mask)
            self.val_graph = graph.subgraph(self.train_mask + self.val_mask)
        self.best_score = -1
        self.patience_knt = 0

    def train(self):
        pass

    def eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.array(labels)
            if torch.is_tensor(probs):
                probs_np = probs.cpu().numpy()
            else:
                probs_np = np.array(probs)

            score['AUROC'] = roc_auc_score(labels_np, probs_np)
            score['PRC'] = average_precision_score(labels_np, probs_np)

            k = int(labels_np.sum())
            score['RecK'] = sum(labels_np[probs_np.argsort()[-k:]]) / sum(labels_np)

            preds = (probs_np >= 0.5).astype(np.int32)

            from sklearn.metrics import precision_score, recall_score, f1_score
            score['Precision'] = precision_score(labels_np, preds)
            score['Recall'] = recall_score(labels_np, preds)
            score['F1'] = f1_score(labels_np, preds)

        try:
            features = self.source_graph.ndata['feature']
            features_np = features.cpu().detach().numpy() if torch.is_tensor(features) else features

            num_nodes = features_np.shape[0]
            if num_nodes > 10000:
                indices = np.random.choice(num_nodes, 10000, replace=False)
                features_np = features_np[indices]
                labels_for_plot = self.source_graph.ndata['label'].cpu().numpy()[indices]
            else:
                labels_for_plot = self.source_graph.ndata['label'].cpu().numpy()

            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features_np)

            plt.figure(figsize=(6, 5))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                                  c=labels_for_plot,
                                  cmap='coolwarm', alpha=0.7)
            plt.title("t-SNE Visualization of Graph Data")
            plt.xlabel("t-SNE Dim 1")
            plt.ylabel("t-SNE Dim 2")
            plt.colorbar(scatter)
            plt.show()
        except Exception as e:
            print("t-SNE visualization failed:", e)

        return score


class XGBGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='hist', eval_metric=eval_metric, verbose=2, **model_config)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        new_feat = gnn(self.source_graph)
        if self.train_config['inductive'] == True:
            new_feat[self.train_mask] = gnn(self.source_graph.subgraph(self.train_mask))
            val_graph = self.source_graph.subgraph(self.train_mask + self.val_mask)
            new_feat[self.val_mask] = gnn(val_graph)[val_graph.ndata['val_mask']]
        self.source_graph.ndata['feature'] = new_feat

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)])
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


def create_masks(graph, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3, seed=42):
    num_nodes = graph.number_of_nodes()
    indices = np.arange(num_nodes)
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_config = {
        "inductive": False,
        "metric": "AUROC",
        "device": device,
    }

    model_config = {
        "num_layers": 2,
        "agg": "mean",
        "init_eps": -1,
    }

    # Load EllipticBitcoinDataset from torch_geometric.
    pyg_dataset = EllipticBitcoinDataset(root='data/elliptic')
    pyg_data = pyg_dataset[0]
    print("Loaded PyG Data:")
    print(pyg_data)

    # Convert the PyG Data object to a DGL graph.
    edge_index = pyg_data.edge_index
    import dgl
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=pyg_data.num_nodes)
    g.ndata['feature'] = pyg_data.x.float()
    g.ndata['label'] = pyg_data.y.long()

    # Drop unknown nodes: keep only nodes with label >= 0.
    valid_nodes = (g.ndata['label'] >= 0).nonzero(as_tuple=True)[0]
    g = dgl.node_subgraph(g, valid_nodes)

    # Create train/val/test masks.
    create_masks(g, train_ratio=0.4, val_ratio=0.3, test_ratio=0.3, seed=42)

    # Wrap the DGL graph in a simple DataWrapper.
    class DataWrapper:
        def __init__(self, graph):
            self.graph = graph

    data = DataWrapper(g)

    if device != "cuda":
        print("GPU not available. Using CPU with tree_method 'hist' in XGBGraphDetector.")

    detector = XGBGraphDetector(train_config, model_config, data)
    try:
        test_score = detector.train()
        print("Test Score:", test_score)
    except Exception as e:
        print("An error occurred during training:", e)
        print("Tip: Ensure tree_method is set to 'hist' in XGBGraphDetector when running on CPU.")


if __name__ == '__main__':
    main()
