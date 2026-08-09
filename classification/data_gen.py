import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.color import rgb2hed, hed2rgb


def apply_he_augmentation_with_mask(image_array):
    """
    Apply H&E stain color augmentation to a histopathology image while preserving white background regions.

    Args:
        image_array: Input RGB image array with values in [0, 1].

    Returns:
        Augmented RGB image array.
    """
    # Create a mask for white/background regions (pixels where all RGB channels exceed threshold)
    white_threshold = 0.66
    white_mask = np.all(image_array > white_threshold, axis=2)

    # Preserve original white region colors for restoration after augmentation
    original_white = image_array.copy()

    hed = rgb2hed(np.clip(image_array, 0, 1.0))

    # Generate random linear transform parameters for Hematoxylin and Eosin channels
    ah = 0.98 + random.random() * 0.04
    bh = -0.02 + random.random() * 0.04
    ae = 0.98 + random.random() * 0.04
    be = -0.02 + random.random() * 0.04

    hed_augmented = hed.copy()
    hed_augmented[:, :, 0] = ah * hed[:, :, 0] + bh  # H channel
    hed_augmented[:, :, 1] = ae * hed[:, :, 1] + be  # E channel

    rgb_augmented = hed2rgb(hed_augmented)
    rgb_augmented = np.clip(rgb_augmented, 0, 1.0)

    # Restore white/background regions to their original colors
    rgb_augmented[white_mask] = original_white[white_mask]
    return rgb_augmented


class HEAugmentation:
    """Random H&E stain augmentation for histopathology images."""

    def __init__(self, apply_prob=0.5):
        self.apply_prob = apply_prob

    def __call__(self, img):
        img_array = np.array(img) / 255.0

        if random.random() < self.apply_prob:
            img_array = apply_he_augmentation_with_mask(img_array)

        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)


def find_all_images(root_dir, class_to_idx=None):
    """
    Recursively find all image files under a directory, organized by class subdirectories.

    Args:
        root_dir: Root directory containing class subdirectories.
        class_to_idx: Optional dict mapping class names to indices. Auto-generated if None.

    Returns:
        samples: List of (image_path, class_index) tuples.
        class_to_idx: Dict mapping class names to indices.
    """
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    samples = []

    if class_to_idx is None:
        classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(root_dir, target_class)

        for root, _, files in os.walk(target_dir):
            for fname in sorted(files):
                if fname.lower().endswith(extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    samples.append(item)

    return samples, class_to_idx


class PathologyImageDataset(Dataset):
    """
    PyTorch Dataset for histopathology image classification.

    Args:
        root_dir: Root directory containing class subdirectories.
        imsize: Target image size for resizing.
        is_train: Whether this is the training set (enables augmentations).
        he_augment_prob: Probability of applying H&E stain augmentation.
    """

    def __init__(self, root_dir, imsize=224, is_train=True, he_augment_prob=0.5):
        self.root_dir = root_dir
        self.is_train = is_train

        self.samples, self.class_to_idx = find_all_images(root_dir)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                HEAugmentation(apply_prob=he_augment_prob),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, class_idx


def data_loader_torch(path, batch_size, imsize=224, num_workers=4):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        path: Root data directory (should contain 'train/img' and 'test/img' subdirectories).
        batch_size: Batch size.
        imsize: Target image size.
        num_workers: Number of data loading worker processes.

    Returns:
        train_loader: Training DataLoader.
        test_loader: Testing DataLoader.
        train_samples: Number of training samples.
        test_samples: Number of testing samples.
    """
    train_dataset = PathologyImageDataset(
        root_dir=os.path.join(path, 'train/img'),
        imsize=imsize,
        is_train=True,
        he_augment_prob=0.5
    )

    test_dataset = PathologyImageDataset(
        root_dir=os.path.join(path, 'test/img'),
        imsize=imsize,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, len(train_dataset.samples), len(test_dataset.samples)


if __name__ == "__main__":
    train_loader, test_loader, train_samples, test_samples = data_loader_torch(
        path="/home/pengxiao/virtualenvs/Osteosarcoma_diagnosis/data/classification",
        batch_size=32,
        imsize=224
    )
    print(f"Training samples: {train_samples}")
    print(f"Testing samples: {test_samples}")