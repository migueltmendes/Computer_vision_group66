from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, "rb") as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


STL10_DIR = Path(__file__).parents[1] / "data" / "stl10_binary"


class STL10_loader(Dataset):
    _path_dict = {
        "train": {
            "data": STL10_DIR / "train_X.bin",
            "labels": STL10_DIR / "train_y.bin",
        },
        "test": {
            "data": STL10_DIR / "test_X.bin",
            "labels": STL10_DIR / "test_y.bin",
        },
    }

    _original_class_mapping = {2: "bird", 5: "deer", 6: "dog", 7: "horse", 8: "monkey"}

    _original_to_new_class_idx = {
        orig: new for new, orig in enumerate(_original_class_mapping.keys())
    }

    def __init__(self, train=True, transform=None):
        """
        Initializes the STL10 dataset.

        Args:
            root (str): Root directory of the dataset.
            train (bool): If True, use the training set, otherwise use the test set.
            transform (callable, optional): A function/transform to apply to the images.
        """

        # YOUR CODE HERE
        self.train = train
        self.transform = transform

        # Pre-load images and labels
        self.images: np.ndarray = read_all_images(
            self._path_dict["train" if train else "test"]["data"]
        )
        self.labels: np.ndarray = (
            read_labels(self._path_dict["train" if train else "test"]["labels"]) - 1
        )  # Convert to 0-based indexing

        # Filter to only include specified classes
        mask = np.isin(self.labels, list(self._original_class_mapping.keys()))
        self.images = self.images[mask]
        self.labels = self.labels[mask]

        # Remap labels to new indices
        self.labels = np.array(
            [self._original_to_new_class_idx[label] for label in self.labels]
        )

        # Create tensors from data
        # self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        # YOUR CODE HERE
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image and its target label.
        """

        # Read image
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class RandomApply:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return self.transform(img)
        return img


def get_dataloaders(batch_size=64, num_workers=4):

    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = STL10_loader(train=True, transform=train_transform)
    test_dataset = STL10_loader(train=False, transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
