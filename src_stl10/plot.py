import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.asarray(x)


def plot_tsne(
    train_embeddings,
    y_train,
    test_embeddings,
    y_test,
    ax_train=None,
    train_title=None,
    ax_test=None,
    test_title=None,
    fig_path=None,
    perplexity=30,
    max_iter=1000,
    random_state=42,
):
    """
    Compute t-SNE on the concatenation of train+test embeddings and plot
    train and test separately with true labels. If ax_train and ax_test
    are provided they will be used, otherwise a new two-column figure
    is created. No legend is drawn. If fig_path is provided the figure
    is saved to that path.
    """
    train_embeddings = _to_numpy(train_embeddings)
    test_embeddings = _to_numpy(test_embeddings)
    y_train = _to_numpy(y_train).ravel()
    y_test = _to_numpy(y_test).ravel()

    # Fit t-SNE on concatenated embeddings, then split
    X = np.vstack([train_embeddings, test_embeddings])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        init="pca",
        random_state=random_state,
    )
    X2 = tsne.fit_transform(X)
    n_train = train_embeddings.shape[0]
    train_tsne = X2[:n_train]
    test_tsne = X2[n_train:]

    # Prepare axes
    created_fig = False
    if ax_train is None or ax_test is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_train, ax_test = axes[0], axes[1]
        created_fig = True
    else:
        # if provided, try to get the figure for saving later
        fig = ax_train.figure

    scatter_kwargs = dict(s=10, alpha=0.8)

    # Use a consistent colormap across train/test based on union of classes
    classes = np.unique(np.concatenate([y_train, y_test]))
    cmap = plt.get_cmap("tab10")
    colors = {cls: cmap(i % cmap.N) for i, cls in enumerate(classes)}

    # Plot train
    ax = ax_train
    for cls in classes:
        mask = y_train == cls
        if not np.any(mask):
            continue
        ax.scatter(train_tsne[mask, 0], train_tsne[mask, 1], color=colors[cls], **scatter_kwargs)
    ax.set_title("Train embeddings (true labels)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    # Plot test
    ax = ax_test
    for cls in classes:
        mask = y_test == cls
        if not np.any(mask):
            continue
        ax.scatter(test_tsne[mask, 0], test_tsne[mask, 1], color=colors[cls], **scatter_kwargs)
    ax.set_title("Test embeddings (true labels)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    if train_title:
        ax_train.set_title(train_title)
    if test_title:
        ax_test.set_title(test_title)

    plt.tight_layout()
    if fig_path:
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        ax_train.figure.savefig(fig_path.with_name(f"{fig_path.stem}_train.png"), bbox_inches="tight", dpi=100)
        ax_test.figure.savefig(fig_path.with_name(f"{fig_path.stem}_test.png"), bbox_inches="tight", dpi=100)
        fig.savefig(fig_path.with_name(f"{fig_path.stem}_combined.png"), bbox_inches="tight", dpi=100)

    if created_fig:
        return fig, (ax_train, ax_test)
    return (ax_train, ax_test)
