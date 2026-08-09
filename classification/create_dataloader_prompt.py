from torch.utils.data import DataLoader, Sampler
import numpy as np
from classification.dataset_prompt import MedicalImageDataset, collate_fn


class PartialSampler(Sampler):
    """
    Custom sampler that only samples a specified percentage of the dataset each epoch
    """

    def __init__(self, dataset_size, percentage=0.3):
        self.dataset_size = dataset_size
        self.num_samples = int(dataset_size * percentage)

    def __iter__(self):
        # Randomly sample indices
        indices = np.random.permutation(self.dataset_size)
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples


def create_data_loaders(
        data_root,
        clinical_data_path_os,  # 新增OS类别临床数据路径
        clinical_data_path_nos,  # 新增NOS类别临床数据路径
        batch_size=16,
        num_workers=4,
        image_size=224,
        train_sample_percentage=0.3
):
    """
    Create train and test data loaders with partial data sampling for training

    Args:
        data_root (str): Path to root directory containing train/ and test/ folders
        clinical_data_path_os (str): Path to OS clinical data Excel file
        clinical_data_path_nos (str): Path to NOS clinical data Excel file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        image_size (int): Size to resize images to
        train_sample_percentage (float): Percentage of training data to use each epoch (0-1)

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = MedicalImageDataset(
        root_dir=data_root,
        clinical_data_path_os=clinical_data_path_os,  # 传递OS临床数据路径
        clinical_data_path_nos=clinical_data_path_nos,  # 传递NOS临床数据路径
        is_train=True,
        image_size=image_size
    )

    test_dataset = MedicalImageDataset(
        root_dir=data_root,
        clinical_data_path_os=clinical_data_path_os,  # 传递OS临床数据路径
        clinical_data_path_nos=clinical_data_path_nos,  # 传递NOS临床数据路径
        is_train=False,
        image_size=image_size
    )

    # Create sampler for training data
    train_sampler = PartialSampler(
        dataset_size=len(train_dataset),
        percentage=train_sample_percentage
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use custom sampler instead of shuffle
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, test_loader