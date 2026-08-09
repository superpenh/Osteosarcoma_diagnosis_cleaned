import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import re


def preprocess_input(x):
    return x * 2.0 - 1.0


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, clinical_data_path_os, clinical_data_path_nos, is_train=True, image_size=224):
        """
        Args:
            root_dir (str): Path to the base directory containing train/ or test/
            clinical_data_path_os (str): Path to the OS clinical data Excel file
            clinical_data_path_nos (str): Path to the NOS clinical data Excel file
            is_train (bool): Whether this is training set or test set
            image_size (int): Size to resize images to
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.split = 'train' if is_train else 'test'

        # Set up image transformations
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        # Load and merge clinical data from both OS and NOS sources
        self.clinical_data = self._load_and_merge_clinical_data(
            clinical_data_path_os,
            clinical_data_path_nos
        )

        print("\nMerged clinical data summary:")
        print("Total cases:", len(self.clinical_data))
        print("Available clinical features:", self.clinical_data.columns.tolist())
        print("Pathology numbers:", self.clinical_data.index.tolist()[:10], "...")

        # Cache available pathology numbers for subsequent matching
        self.available_pathology_numbers = set(self.clinical_data.index)

        # Load image paths and labels
        self.images = []
        self.labels = []

        for class_idx in ['1', '2']:
            class_path = os.path.join(root_dir, self.split, 'img', class_idx)
            if not os.path.exists(class_path):
                raise RuntimeError(f"Path does not exist: {class_path}")

            for root, _, files in os.walk(class_path):
                for img_name in files:
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, img_name)
                        self.images.append(img_path)
                        self.labels.append(int(class_idx) - 1)  # Convert to 0-based indexing

        print(f"Loaded {len(self.images)} images for {self.split} set")

    def _load_and_merge_clinical_data(self, path_os, path_nos):
        """Load and merge clinical data from OS and NOS Excel files."""
        os_data = self._load_single_clinical_data(path_os, 'OS')
        nos_data = self._load_single_clinical_data(path_nos, 'NOS')

        combined_data = pd.concat([os_data, nos_data], ignore_index=True)

        # Check for duplicate pathology numbers
        duplicates = combined_data['病理号'].duplicated()
        if duplicates.any():
            print("\nWarning: duplicate pathology numbers found:")
            print(combined_data[duplicates]['病理号'].tolist())

        # Set pathology number as index, keeping the last occurrence if duplicates exist
        combined_data.set_index('病理号', inplace=True)

        return combined_data

    def _load_single_clinical_data(self, path, data_type):
        """Load clinical data from a single Excel file."""
        print(f"\nLoading {data_type} clinical data...")

        excel_file = pd.ExcelFile(path)
        sheet_names = excel_file.sheet_names
        print(f"Found sheets: {sheet_names}")

        all_sheet_data = []
        for sheet_name in sheet_names:
            sheet_data = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                dtype=str
            )
            all_sheet_data.append(sheet_data)

        combined = pd.concat(all_sheet_data, ignore_index=True)

        required_columns = ['病理号', '性别', '年龄', '发病部位']
        missing_columns = [col for col in required_columns if col not in combined.columns]
        if missing_columns:
            raise ValueError(f"{data_type} Excel file missing required columns: {missing_columns}")

        return combined[required_columns]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            # Apply preprocessing
            image = preprocess_input(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None

        # Get pathology number from image path
        filename = os.path.basename(img_path)
        img_pathology_number = filename.split('_')[0].split('-')[0].split('(')[0].strip()
        img_pathology_number = ''.join(re.findall(r'[a-zA-Z0-9]', img_pathology_number.split('_')[0]))

        # Match pathology number against the clinical data index
        try:
            mask = self.clinical_data.index.str.contains(img_pathology_number)
            if not mask.any():
                print(f"Warning: no clinical data found for pathology number {img_pathology_number}")
                return None

            matched_row = self.clinical_data[mask].iloc[0]
            clinical_info = matched_row.to_dict()
            clinical_info = {k: str(v) for k, v in clinical_info.items()}

        except Exception as e:
            print(f"Warning: error looking up pathology number {img_pathology_number}: {str(e)}")
            return None

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'clinical_info': clinical_info,
            'pathology_number': img_pathology_number
        }


def collate_fn(batch):
    """
    Custom collate function to handle None values and create batches
    """
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    return {
        'images': torch.stack([item['image'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'clinical_info': [item['clinical_info'] for item in batch],
        'pathology_numbers': [item['pathology_number'] for item in batch]
    }