import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def plot_tsne(model, train_dataloader, test_dataloader):
    # Use FAISS to cluster embeddings into 5 clusters
    n_clusters = 5
    d = train_embeddings.shape[1]

    # Train KMeans on the TRAIN embeddings (so clusters derive from train set)
    kmeans = faiss.Kmeans(d, n_clusters, niter=50, verbose=False, seed=1234)
    kmeans.train(train_embeddings)

    # Assign cluster indices for train and test
    # Build an index from the centroids and search nearest centroid
    index = faiss.IndexFlatL2(d)
    index.add(kmeans.centroids)
    _, train_cluster_labels = index.search(train_embeddings, 1)
    _, test_cluster_labels = index.search(test_embeddings, 1)
    train_cluster_labels = train_cluster_labels.ravel()
    test_cluster_labels = test_cluster_labels.ravel()

    # Compute 2D t-SNE projections (fit separately for train and test)
    tsne = TSNE(
        n_components=2, perplexity=30, max_iter=1000, init="pca", random_state=42
    )

    train_tsne = tsne.fit_transform(train_embeddings)
    test_tsne = tsne.fit_transform(test_embeddings)

    # Plot 4 plots: train(true), train(cluster), test(true), test(cluster)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.tight_layout(h_pad=4, w_pad=4)

    cmap = plt.get_cmap("tab10")
    scatter_kwargs = dict(s=10, alpha=0.8)

    # Top-left: train true labels
    ax = axes[0, 0]
    for cls in np.unique(y_train):
        mask = y_train == cls
        ax.scatter(
            train_tsne[mask, 0],
            train_tsne[mask, 1],
            label=str(int(cls)),
            **scatter_kwargs,
        )
    ax.set_title("Train embeddings (true labels)")
    ax.legend(title="Class", markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Top-right: train clustered labels
    ax = axes[0, 1]
    for cls in np.unique(train_cluster_labels):
        mask = train_cluster_labels == cls
        ax.scatter(
            train_tsne[mask, 0],
            train_tsne[mask, 1],
            label=f"cluster {int(cls)}",
            **scatter_kwargs,
        )
    ax.set_title(f"Train embeddings (FAISS k={n_clusters})")
    ax.legend(
        title="Cluster", markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Bottom-left: test true labels
    ax = axes[1, 0]
    for cls in np.unique(y_test):
        mask = y_test == cls
        ax.scatter(
            test_tsne[mask, 0],
            test_tsne[mask, 1],
            label=str(int(cls)),
            **scatter_kwargs,
        )
    ax.set_title("Test embeddings (true labels)")
    ax.legend(title="Class", markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Bottom-right: test clustered labels
    ax = axes[1, 1]
    for cls in np.unique(test_cluster_labels):
        mask = test_cluster_labels == cls
        ax.scatter(
            test_tsne[mask, 0],
            test_tsne[mask, 1],
            label=f"cluster {int(cls)}",
            **scatter_kwargs,
        )
    ax.set_title(f"Test embeddings (FAISS k={n_clusters})")
    ax.legend(
        title="Cluster", markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    for ax in axes.flatten():
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    plt.subplots_adjust(right=0.78)
    plt.show()
    # --- end paste ---
