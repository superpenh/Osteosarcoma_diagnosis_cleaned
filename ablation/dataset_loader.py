import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import random


class DRDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=384):
        """
        Args:
            root_dir: Root directory containing OS and NOS subdirectories.
            split: 'train' or 'test'.
            image_size: Output image size (suitable for ViT input).
        """
        self.root_dir = root_dir
        self.split = split
        self.image_files = []  # (image_path, label, patient_name)
        self.image_size = image_size

        self.basic_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # Augmentation variants (training only)
        self.augment_transforms = [
            # Variant 1: mild rotation + brightness/contrast jitter
            transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomRotation(degrees=10, fill=0),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ]),
            # Variant 2: horizontal flip + mild rotation
            transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=5, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        ]

        self.augment_times = len(self.augment_transforms) if split == 'train' else 0

        for label, class_name in enumerate(['NOS', 'OS']):  # 0: NOS, 1: OS
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            split_dir = os.path.join(class_dir, split)
            if not os.path.exists(split_dir):
                print(f"Warning: directory not found {split_dir}")
                continue

            for patient_name in os.listdir(split_dir):
                patient_dir = os.path.join(split_dir, patient_name)
                if not os.path.isdir(patient_dir):
                    continue

                image_files = [f for f in os.listdir(patient_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

                for img_file in image_files:
                    img_path = os.path.join(patient_dir, img_file)
                    self.image_files.append((img_path, label, patient_name))

        # Build augmentation index mapping for training mode
        if split == 'train':
            self.augment_indices = []
            for idx in range(len(self.image_files)):
                self.augment_indices.append((idx, -1))  # -1 = original (no augmentation)
                for aug_idx in range(self.augment_times):
                    self.augment_indices.append((idx, aug_idx))
        else:
            self.augment_indices = [(idx, -1) for idx in range(len(self.image_files))]

    def square_crop(self, image):
        """
        Convert image to square by center-cropping along the longer dimension,
        using the shorter side as the square edge length.
        """
        width, height = image.size

        square_size = min(width, height)

        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size

        return image.crop((left, top, right, bottom))
    def __len__(self):
        return len(self.augment_indices)

    def __getitem__(self, idx):
        orig_idx, aug_idx = self.augment_indices[idx]
        img_path, label, patient_name = self.image_files[orig_idx]

        try:
            img = Image.open(img_path).convert('L')

            img = self.square_crop(img)

            if self.split != 'train' or aug_idx == -1:
                img_tensor = self.basic_transform(img)
            else:
                transform = self.augment_transforms[aug_idx % len(self.augment_transforms)]
                img_tensor = transform(img)

            img_filename = os.path.basename(img_path)

            return {
                'image': img_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'patient_name': patient_name,
                'filename': img_filename,
                'augmentation': aug_idx
            }

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return {
                'image': torch.zeros(1, self.image_size, self.image_size),
                'label': torch.tensor(label, dtype=torch.long),
                'patient_name': patient_name,
                'filename': os.path.basename(img_path),
                'augmentation': aug_idx
            }


if __name__ == '__main__':
    data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/internal_DR/DR'

    train_dataset = DRDataset(
        root_dir=data_root,
        split='train',
        image_size=384
    )

    test_dataset = DRDataset(
        root_dir=data_root,
        split='test',
        image_size=384
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print("\nFirst 5 training samples:")
    print("-" * 40)
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i + 1}:")
        print(f"  Filename: {sample['filename']}")
        print(f"  Patient: {sample['patient_name']}")
        print(f"  Label: {sample['label'].item()} ({'OS' if sample['label'].item() == 1 else 'NOS'})")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Augmentation: {sample['augmentation']}")
        print("-" * 40)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    dataiter = iter(train_loader)
    batch = next(dataiter)

    images = batch['image']
    labels = batch['label']
    filenames = batch['filename']

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for i, (img, label, filename) in enumerate(zip(images, labels, filenames)):
        if len(images) > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.imshow(img[0], cmap='gray')
        ax.set_title(f"{'OS' if label.item() == 1 else 'NOS'}\n{filename}")
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Training Set Samples")
    plt.show()
