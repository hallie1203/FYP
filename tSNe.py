import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import EllipticBitcoinDataset

# ------------------------
# 1. Load the Elliptic Bitcoin Dataset
# ------------------------
dataset = EllipticBitcoinDataset(root='data/EllipticBitcoin')
data = dataset[0]  # The dataset contains a single graph.
print("Data summary:")
print(data)

# ------------------------
# 2. Sample a subset of nodes for t-SNE visualization
# ------------------------
X = data.x.cpu().numpy()      # Node features (num_nodes, num_features)
labels = data.y.cpu().numpy() # Node labels (0: licit, 1: illicit)

num_nodes = X.shape[0]
sample_size = 5000  # Adjust this value as needed

if num_nodes > sample_size:
    indices = np.random.choice(num_nodes, sample_size, replace=False)
    X_sample = X[indices]
    labels_sample = labels[indices]
else:
    X_sample = X
    labels_sample = labels

# ------------------------
# 3. Run t-SNE on the sampled node features
# ------------------------
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

# ------------------------
# 4. Visualize the t-SNE Embedding
# ------------------------
mask_licit = labels_sample == 0
mask_illicit = labels_sample == 1
mask_unknown = labels_sample == 2

plt.figure(figsize=(8, 8))
plt.scatter(X_tsne[mask_unknown, 0], X_tsne[mask_unknown, 1],
            color='grey', label='Unknown', alpha=0.9, edgecolors='w', s=30)
# Draw licit next (middle layer)
plt.scatter(X_tsne[mask_licit, 0], X_tsne[mask_licit, 1],
            color='blue', label='Licit', alpha=0.9, edgecolors='w', s=30)
# Draw illicit last (top layer)
plt.scatter(X_tsne[mask_illicit, 0], X_tsne[mask_illicit, 1],
            color='red', label='Illicit', alpha=0.9, edgecolors='w', s=30)

plt.legend()
plt.title("t-SNE Visualization of Elliptic Bitcoin Dataset (Subset)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()