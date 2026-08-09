import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np


class OsteosarcomaDataset(Dataset):
    def __init__(self, root_dir, os_clinical_file, nos_clinical_file, image_size=224):
        self.root_dir = root_dir
        self.cases = []
        self.required_fields = ['性别', '年龄', '发病部位']
        self.image_size = image_size

        # Basic DR image preprocessing
        self.basic_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # Pathology patch preprocessing
        self.patch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load and merge clinical data
        self.clinical_data = self.load_clinical_data(os_clinical_file, nos_clinical_file)

        # Traverse and load image data
        for label, class_name in enumerate(['NOS', 'OS']):  # 0:NOS, 1:OS
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            split_dir = os.path.join(class_dir)
            if not os.path.exists(split_dir):
                print(f"Warning: directory not found {split_dir}")
                continue

            for patient_name in os.listdir(split_dir):
                patient_dir = os.path.join(split_dir, patient_name)
                if not os.path.isdir(patient_dir):
                    continue

                dr_images = []
                patch_folder = None

                for item in os.listdir(patient_dir):
                    item_path = os.path.join(patient_dir, item)
                    if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        dr_images.append(item_path)
                    elif os.path.isdir(item_path):
                        patch_files = [f for f in os.listdir(item_path)
                                       if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
                        if len(patch_files) > 0:
                            patch_folder = item_path

                if len(dr_images) > 0 and patch_folder is not None and patient_name in self.clinical_data.index:
                    self.cases.append({
                        'patient_name': patient_name,
                        'class': label,
                        'dr_images': sorted(dr_images),
                        'patch_folder': patch_folder,
                        'class_name': class_name
                    })

        print(f"Total valid cases in val split: {len(self.cases)}")

    def load_clinical_data(self, os_clinical_file, nos_clinical_file):
        """Load and merge OS and NOS clinical data."""
        all_clinical_data = []

        try:
            os_data = self.load_single_clinical_file(os_clinical_file)
            all_clinical_data.append(os_data)
            print(f"Successfully loaded OS clinical data with {len(os_data)} records")
        except Exception as e:
            print(f"Error loading OS clinical data: {str(e)}")

        try:
            nos_data = self.load_single_clinical_file(nos_clinical_file)
            all_clinical_data.append(nos_data)
            print(f"Successfully loaded NOS clinical data with {len(nos_data)} records")
        except Exception as e:
            print(f"Error loading NOS clinical data: {str(e)}")

        if not all_clinical_data:
            raise ValueError("Failed to load any clinical data")

        combined_data = pd.concat(all_clinical_data, ignore_index=True)
        combined_data.drop_duplicates(subset=['姓名'], keep='last', inplace=True)
        combined_data.set_index('姓名', inplace=True)

        print(f"Total unique patients with clinical data: {len(combined_data)}")
        return combined_data

    def load_single_clinical_file(self, file_path):
        """Load a single clinical data file."""
        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
        sheet_names = excel_file.sheet_names
        all_sheets_data = []

        for sheet_name in sheet_names:
            sheet_data = pd.read_excel(
                excel_file,
                sheet_name=sheet_name,
                engine='openpyxl',
                dtype=str
            )

            if not sheet_data.empty and '姓名' in sheet_data.columns:
                columns_to_keep = ['姓名'] + self.required_fields
                available_columns = [col for col in columns_to_keep if col in sheet_data.columns]

                if not all(field in available_columns for field in self.required_fields):
                    print(f"Warning: Sheet '{sheet_name}' in '{file_path}' missing required fields")
                    continue

                sheet_data = sheet_data[available_columns]
                all_sheets_data.append(sheet_data)
            else:
                print(f"Warning: Sheet '{sheet_name}' in '{file_path}' is empty or missing '姓名' column")

        if not all_sheets_data:
            raise ValueError(f"No valid data found in file: {file_path}")

        file_data = pd.concat(all_sheets_data, ignore_index=True)

        for col in self.required_fields:
            if col in file_data.columns:
                file_data[col] = file_data[col].apply(
                    lambda x: x[0] if isinstance(x, list) else str(x)
                )

        return file_data

    def square_pad(self, image):
        """
        Pad image to square using edge replication (consistent with training code).
        Strategy: max(width, height) + np.pad(mode='edge')
        """
        width, height = image.size

        if width == height:
            return image

        square_size = max(width, height)

        pad_width = square_size - width
        pad_height = square_size - height

        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top

        img_array = np.array(image)

        if len(img_array.shape) == 2:
            pad_params = ((top, bottom), (left, right))
        else:
            pad_params = ((top, bottom), (left, right), (0, 0))

        padded_array = np.pad(
            img_array,
            pad_params,
            mode='edge'
        )

        return Image.fromarray(padded_array.astype(np.uint8))

    def load_dr_images(self, dr_paths, max_images=None):
        """Load a variable number of DR images, at most max_images."""
        images = []
        for path in dr_paths[:max_images]:
            try:
                img = Image.open(path).convert('L')
                img = self.square_pad(img)
                img = self.basic_transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading DR image {path}: {str(e)}")
                images.append(torch.zeros(1, self.image_size, self.image_size))

        if len(images) == 0:
            images.append(torch.zeros(1, self.image_size, self.image_size))

        stacked_images = torch.stack(images)

        stacked_images = torch.stack(images)
        return stacked_images

    def load_patches(self, patch_folder, num_patches=50):
        """Load pathology patches, default 50."""
        patches = []
        patch_names = []
        try:
            all_files = sorted([f for f in os.listdir(patch_folder)
                                if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))])

            patch_files = all_files[:num_patches]

            for patch_file in patch_files:
                patch_path = os.path.join(patch_folder, patch_file)
                try:
                    patch = Image.open(patch_path).convert('RGB')
                    patch = self.patch_transform(patch)
                    patches.append(patch)
                    patch_names.append(patch_file)
                except Exception as e:
                    print(f"Error loading individual patch {patch_file}: {e}")

                if len(patches) >= num_patches:
                    break

            while len(patches) < num_patches:
                if patches:
                    patches.append(patches[-1])
                    last_name = patch_names[-1]
                    patch_names.append(f"{last_name} (padding)")
                else:
                    patches.append(torch.zeros(3, 256, 256))
                    patch_names.append("zero_padding_empty_folder")

        except Exception as e:
            print(f"Error loading patches from {patch_folder}: {str(e)}")
            patches = [torch.zeros(3, 256, 256) for _ in range(num_patches)]

        return torch.stack(patches), patch_names

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        patient_name = case['patient_name']

        try:
            clinical_info = self.clinical_data.loc[patient_name].to_dict()

            assert all(isinstance(v, str) for v in clinical_info.values()), "Values must be strings"
            patches_tensor, patch_names_list = self.load_patches(case['patch_folder'])

            return {
                'dr_images': self.load_dr_images(case['dr_images']),
                'dr_paths': case['dr_images'],
                'patch_folder': case['patch_folder'],
                'patches': patches_tensor,
                'patch_names': patch_names_list,
                'label': case['class'],
                'patient_name': case['patient_name'],
                'class_name': case['class_name'],
                'clinical_info': clinical_info
            }
        except Exception as e:
            print(f"Error processing case {patient_name}: {str(e)}")
            default_clinical_info = {field: "" for field in self.required_fields}
            return {
                'dr_images': torch.zeros(2, self.image_size, self.image_size),
                'dr_paths': [],
                'patch_folder': "",
                'patches': torch.zeros(50, 3, 256, 256),
                'patch_names': ["error"] * 50,
                'label': case['class'],
                'patient_name': case['patient_name'],
                'class_name': case['class_name'],
                'clinical_info': default_clinical_info
            }


if __name__ == '__main__':
    data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/prospective_val_set1'
    os_clinical_file = '/home/pengxiao/disk1/FAH 前瞻性研究数据/2025年 中山附一 前瞻性研究 OS & NOS 2025.XLSX'
    nos_clinical_file = '/home/pengxiao/disk1/FAH 前瞻性研究数据/2025年 中山附一 前瞻性研究 OS & NOS 2025.XLSX'

    val_dataset = OsteosarcomaDataset(
        root_dir=data_root,
        os_clinical_file=os_clinical_file,
        nos_clinical_file=nos_clinical_file,
        image_size=224
    )

    print(f"Validation set size: {len(val_dataset)}")

    if len(val_dataset) > 0:
        sample = val_dataset[90]
        print("\nSample info:")
        print("-" * 40)
        print(f"Patient: {sample['patient_name']}")
        print(f"Label: {sample['label']} ({sample['class_name']})")
        print(f"DR image shape: {sample['dr_images'].shape}")
        print(f"Patches shape: {sample['patches'].shape}")
        print(f"Clinical info: {sample['clinical_info']}")
        print("-" * 40)
        print(sample['patch_names'])
