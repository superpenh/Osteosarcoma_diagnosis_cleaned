import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import json
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from datetime import datetime
from transformers import BertModel, BertTokenizer
import timm
from dataset_loader import DRDataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image, ImageDraw

# Configure matplotlib for Chinese character display
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Fallback if SimHei is not available
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print("Warning: No Chinese font found, using English labels")


class AnnotationProcessor:
    """Process GeoJSON annotation files using edge-padding strategy."""

    def __init__(self, annotation_root):
        self.annotation_root = annotation_root

    def square_pad(self, img):
        """
        Pad image to square using edge replication.

        Args:
            img: PIL Image object

        Returns:
            padded_img: Square padded image
            padding_info: (top, left) padding offsets
        """
        width, height = img.size

        if width == height:
            return img, (0, 0)

        # Target square size (use the larger dimension)
        square_size = max(width, height)

        # Compute required padding
        pad_width = square_size - width
        pad_height = square_size - height

        # Distribute padding evenly on both sides
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top

        # Convert to numpy array and pad using edge replication
        img_array = np.array(img)
        padded_array = np.pad(
            img_array,
            ((top, bottom), (left, right)),
            mode='edge'
        )

        # Convert back to PIL Image
        padded_img = Image.fromarray(padded_array.astype(np.uint8))

        return padded_img, (top, left)

    def load_annotation_gurouliu(self, annotation_path):
        """Load annotations from a GeoJSON file."""
        if os.path.exists(annotation_path) == False:
            return None

        try:
            with open(annotation_path, 'r', encoding='utf-8') as fp:
                annotations = json.load(fp)
        except Exception as e:
            print(f"Error loading annotation {annotation_path}: {e}")
            return None

        n = 0
        region_dict = {}

        for annotation in annotations["features"]:
            c = 0
            for annotationCoordinates in annotation['geometry']['coordinates']:
                c = 0
                num = len(annotationCoordinates)
                if num == 1:
                    continue
                coords = np.zeros((num, 2))
                for coordinate in annotationCoordinates:
                    x_coord = coordinate[0]
                    y_coord = coordinate[1]
                    coords[c][0] = x_coord
                    coords[c][1] = y_coord
                    c += 1
                coords = coords[:c, :]
                region_dict[n] = coords
                n = n + 1

        return region_dict

    def load_annotation(self, patient_type, patient_name, filename):
        """Load annotations for a specific patient image."""
        # Build annotation file path
        annotation_dir = os.path.join(self.annotation_root, patient_type, patient_name)
        # Convert image filename to corresponding geojson filename
        base_name = os.path.splitext(filename)[0]
        geojson_file = os.path.join(annotation_dir, f"{base_name}.geojson")

        if not os.path.exists(geojson_file):
            return None

        region_dict = self.load_annotation_gurouliu(geojson_file)
        return region_dict

    def region_dict_to_mask_with_padding(self, region_dict, original_size, target_size):
        """
        Convert region_dict to a segmentation mask using edge padding for size transformation.

        Args:
            region_dict: Annotation data
            original_size: Original image size (width, height)
            target_size: Target size (int)
        """
        if region_dict is None or len(region_dict) == 0:
            return np.zeros((target_size, target_size), dtype=np.float32)

        width, height = original_size
        square_size = max(width, height)

        # Compute padding parameters
        pad_width = square_size - width
        pad_height = square_size - height
        left = pad_width // 2
        top = pad_height // 2

        # Create mask on the padded square canvas
        padded_mask = np.zeros((square_size, square_size), dtype=np.uint8)

        try:
            # Iterate over all regions
            for region_id, coords in region_dict.items():
                if coords.shape[0] < 3:
                    continue

                # Adjust coordinates for padding offset
                points = []
                for coord in coords:
                    x = int(np.clip(coord[0] + left, 0, square_size - 1))
                    y = int(np.clip(coord[1] + top, 0, square_size - 1))
                    points.append((x, y))

                if len(points) >= 3:
                    # Draw polygon using PIL
                    img = Image.fromarray(padded_mask)
                    draw = ImageDraw.Draw(img)
                    draw.polygon(points, fill=1)
                    padded_mask = np.array(img)

            # Resize to target size
            final_mask = cv2.resize(
                padded_mask,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST
            )

            return final_mask.astype(np.float32)

        except Exception as e:
            print(f"Error converting region_dict to mask with padding: {e}")
            return np.zeros((target_size, target_size), dtype=np.float32)

    def region_dict_to_perturbed_box_mask(self, region_dict, original_size, target_size,
                                          perturb_prob=1.0, shift_limit=0.2, scale_limit=0.2):
        """
        Convert fine-grained annotations to coarse, perturbed bounding box masks
        (simulating rough hand-drawn boxes).

        Args:
            perturb_prob: Probability of applying perturbation (0-1)
            shift_limit: Max shift ratio (relative to box width/height)
            scale_limit: Max scale ratio (relative to box width/height)
        """
        if region_dict is None or len(region_dict) == 0:
            return np.zeros((target_size, target_size), dtype=np.float32)

        width, height = original_size
        square_size = max(width, height)

        # Padding parameter calculation (maintaining coordinate system consistency)
        pad_width = square_size - width
        pad_height = square_size - height
        left_pad = pad_width // 2
        top_pad = pad_height // 2

        # 1. Collect all polygon coordinates and compute bounding rectangle
        all_points = []
        try:
            for region_id, coords in region_dict.items():
                for coord in coords:
                    x = int(np.clip(coord[0] + left_pad, 0, square_size - 1))
                    y = int(np.clip(coord[1] + top_pad, 0, square_size - 1))
                    all_points.append([x, y])

            if not all_points:
                return np.zeros((target_size, target_size), dtype=np.float32)

            all_points = np.array(all_points)
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)

            box_w = x_max - x_min
            box_h = y_max - y_min

            # 2. Apply random perturbation
            if np.random.random() < perturb_prob:
                # Random scaling (simulates drawing too large or too small)
                scale_x = 1.0 + np.random.uniform(-scale_limit, scale_limit)
                scale_y = 1.0 + np.random.uniform(-scale_limit, scale_limit)

                # Random shift (simulates hand jitter)
                shift_x = np.random.uniform(-shift_limit, shift_limit) * box_w
                shift_y = np.random.uniform(-shift_limit, shift_limit) * box_h

                # New center after shift
                center_x = x_min + box_w / 2 + shift_x
                center_y = y_min + box_h / 2 + shift_y

                # New width and height after scaling
                new_w = box_w * scale_x
                new_h = box_h * scale_y

                # Recompute coordinates
                x_min = int(center_x - new_w / 2)
                x_max = int(center_x + new_w / 2)
                y_min = int(center_y - new_h / 2)
                y_max = int(center_y + new_h / 2)

                # Clamp to valid range
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(square_size - 1, x_max)
                y_max = min(square_size - 1, y_max)

            # 3. Draw rectangle mask
            padded_mask = np.zeros((square_size, square_size), dtype=np.uint8)
            cv2.rectangle(padded_mask, (x_min, y_min), (x_max, y_max), 1, -1)

            # 4. Resize to target size
            final_mask = cv2.resize(
                padded_mask,
                (target_size, target_size),
                interpolation=cv2.INTER_NEAREST
            )

            return final_mask.astype(np.float32)

        except Exception as e:
            print(f"Error creating perturbed box mask: {e}")
            return np.zeros((target_size, target_size), dtype=np.float32)

    def generate_random_box_mask(self, target_size, min_size=50, max_size=200):
        """
        Generate a fully random rectangle mask (for controlled variable experiments).
        Completely ignores real annotations; used to verify whether the model
        truly exploits positional information.
        """
        mask = np.zeros((target_size, target_size), dtype=np.uint8)

        # 1. Random width and height (within reasonable bounds)
        w = np.random.randint(min_size, max_size)
        h = np.random.randint(min_size, max_size)

        # 2. Random top-left corner, ensuring box stays within image bounds
        max_x = target_size - w
        max_y = target_size - h

        if max_x <= 0 or max_y <= 0:
            x, y = 0, 0
        else:
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)

        # 3. Draw filled rectangle
        cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)

        return mask.astype(np.float32)



class ClinicalInfoProcessor(nn.Module):
    """Clinical information processor: converts clinical info to vector representations (dynamic concat)."""

    def __init__(self, bert_model="/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/",
                 output_dim=256):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )

        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Only projection layer is trainable
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, clinical_info):
        # Dynamically build structured text from available fields
        texts = []
        for info in clinical_info:
            parts = []
            if '性别' in info: parts.append(f"性别{info['性别']}")
            if '年龄' in info: parts.append(f"年龄{info['年龄']}岁")
            if '发病部位' in info: parts.append(f"发病部位在{info['发病部位']}")
            texts.append(" ".join(parts))

        # Tokenize
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(next(self.bert.parameters()).device)

            # Get BERT embeddings
            outputs = self.bert(**encoded)

        # Use [CLS] token output as text representation
        text_features = outputs.last_hidden_state[:, 0]

        # Project to target dimension
        clinical_embedding = self.projection(text_features)
        return clinical_embedding


class SpatialAwareModule(nn.Module):
    """
    Spatial Awareness Module (SAM) - Interactive Attention version.
    Segmentation results directly weight feature maps, forcing the classifier
    to focus on segmented regions.
    """

    def __init__(self, backbone_channels_list, hidden_dim=256, clinical_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        shallow_channels = backbone_channels_list[0]
        deep_channels = backbone_channels_list[-1]

        # 1. Deep feature processing
        self.deep_conv = nn.Sequential(
            nn.Conv2d(deep_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Visual feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(shallow_channels + hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 3. Clinical gating
        self.gate_generator = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 4. Segmentation branch (outputs 1-channel attention map)
        self.seg_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

        # 5. Feature enhancement adapter (for better post-attention fusion)
        self.attend_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 6. Classification branch
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, image_features, clinical_embedding=None):
        f1 = image_features[0]
        f4 = image_features[-1]

        # --- 1. Feature fusion ---
        deep_processed = self.deep_conv(f4)
        upsampled_deep = F.interpolate(deep_processed, size=f1.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([f1, upsampled_deep], dim=1)
        fusion_features = self.fusion_conv(fused)

        # --- 2. Clinical information injection ---
        shared_features = fusion_features
        if clinical_embedding is not None:
            gate = self.gate_generator(clinical_embedding)
            gate = gate.unsqueeze(-1).unsqueeze(-1)
            shared_features = fusion_features + (fusion_features * gate)

        # --- 3. Generate segmentation mask (attention map) ---
        seg_logits_small = self.seg_head(shared_features)

        # Generate 0~1 attention map
        attention_map = torch.sigmoid(seg_logits_small)  # [B, 1, H_feat, W_feat]

        # --- 4. Use mask to enhance features ---
        # Features = original + (original * mask).
        # Masked regions get amplified; uncovered regions preserved via residual connection.
        # More stable than direct multiplication — avoids total feature loss on poor mask init.
        annotated_features = shared_features * (1 + attention_map)

        # Optional smoothing convolution
        refined_features = self.attend_conv(annotated_features)

        # --- 5. Classification and output ---
        cls_pred = self.classifier(refined_features)

        # Upsample small logits to original image size for loss computation
        seg_pred_upsampled = F.interpolate(seg_logits_small, size=(384, 384), mode='bilinear', align_corners=False)
        seg_pred_final = torch.sigmoid(seg_pred_upsampled)

        return {
            'segmentation': seg_pred_final,
            'classification': cls_pred,
            'features': refined_features
        }



class DDKGModel(nn.Module):
    """DDKG main model."""

    def __init__(self, num_classes=2, image_size=384, clinical_dim=256,
                 bert_model_path="/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/",
                 swin_pretrained_path=None):
        super().__init__()

        # Swin Transformer backbone
        print("Loading Swin Transformer backbone...")
        self.backbone = timm.create_model(
            'swin_large_patch4_window12_384.ms_in22k_ft_in1k',
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )

        original_patch_embed = self.backbone.patch_embed.proj
        self.backbone.patch_embed.proj = nn.Conv2d(
            1, original_patch_embed.out_channels,
            kernel_size=original_patch_embed.kernel_size,
            stride=original_patch_embed.stride,
            padding=original_patch_embed.padding
        )

        with torch.no_grad():
            self.backbone.patch_embed.proj.weight = nn.Parameter(
                original_patch_embed.weight.mean(dim=1, keepdim=True)
            )
            if original_patch_embed.bias is not None:
                self.backbone.patch_embed.proj.bias = nn.Parameter(
                    original_patch_embed.bias.clone()
                )
        if swin_pretrained_path and os.path.exists(swin_pretrained_path):
            print(f"Loading pretrained Swin weights from: {swin_pretrained_path}")
            self.load_swin_pretrained_weights(swin_pretrained_path)

        # Clinical information processor
        self.clinical_processor = ClinicalInfoProcessor(
            bert_model=bert_model_path,
            output_dim=clinical_dim
        )

        swin_channels_list = self.backbone.feature_info.channels()
        print(f"All Swin channels: {swin_channels_list}")

        # Spatial Awareness Module
        self.sam = SpatialAwareModule(
            swin_channels_list,
            clinical_dim=clinical_dim
        )
        print(f"SAM hidden_dim: {self.sam.hidden_dim}")


        print("Freezing early stages of Swin Transformer...")
        # Freeze patch_embed and stages 0, 1
        for name, param in self.backbone.named_parameters():
            if name.startswith('patch_embed') or name.startswith('layers.0') or name.startswith('layers.1'):
                param.requires_grad = False

        # Verify freezing
        trainable_params_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable parameters in backbone after freezing: {trainable_params_backbone:,}")

    def load_swin_pretrained_weights(self, pretrained_path):
        """Load pretrained Swin weights."""
        try:
            # Load pretrained weights
            if pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
                checkpoint = torch.load(pretrained_path, map_location='cpu')

                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    pretrained_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    pretrained_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                else:
                    pretrained_dict = checkpoint
            else:
                pretrained_dict = torch.load(pretrained_path, map_location='cpu')

            # Get current model state_dict
            model_dict = self.backbone.state_dict()

            # Filter out mismatched keys, especially patch_embed related ones
            filtered_dict = {}
            patch_embed_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias']

            for k, v in pretrained_dict.items():
                # Remove possible prefixes
                key = k
                if key.startswith('backbone.'):
                    key = key[9:]
                if key.startswith('model.'):
                    key = key[6:]

                # Check if key exists in current model
                if key in model_dict:
                    # Skip patch_embed layers (input channels were modified)
                    if not any(patch_key in key for patch_key in patch_embed_keys):
                        # Check shape compatibility
                        if v.shape == model_dict[key].shape:
                            filtered_dict[key] = v
                        else:
                            print(f"Shape mismatch for {key}: {v.shape} vs {model_dict[key].shape}")
                    else:
                        print(f"Skipping patch_embed layer: {key}")

            # Load filtered weights
            missing_keys, unexpected_keys = self.backbone.load_state_dict(filtered_dict, strict=False)

            print(f"Successfully loaded pretrained Swin weights!")
            print(f"Loaded {len(filtered_dict)} parameters")
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)} (expected for patch_embed modifications)")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            print(f"Error loading pretrained Swin weights: {e}")
            print("Continuing with ImageNet pretrained weights...")

    def forward(self, images, clinical_info=None, return_features=True):
        # 1. Extract visual features
        image_features = self.backbone(images)
        converted_features = []
        for feat in image_features:
            if len(feat.shape) == 4 and feat.shape[-1] > feat.shape[1]:
                feat = feat.permute(0, 3, 1, 2)
            converted_features.append(feat)

        # 2. Extract clinical text features
        clinical_embedding = None
        if clinical_info is not None:
            clinical_embedding = self.clinical_processor(clinical_info)

        # 3. Process visual and clinical features through SAM
        sam_outputs = self.sam(converted_features, clinical_embedding)

        results = {
            'segmentation': sam_outputs['segmentation'],
            'classification': sam_outputs['classification']
        }

        if return_features:
            results['features'] = sam_outputs['features']

        return results



class DDKGDataset(DRDataset):
    """Extended dataset class with annotation support and dynamic clinical fields."""

    def __init__(self, root_dir, clinical_data_dir, annotation_root, split='train', image_size=384, random_box_mode=False, active_fields=None):
        super().__init__(root_dir, split, image_size)

        # Dynamically set required clinical fields
        self.required_fields = active_fields if active_fields is not None else ['性别', '年龄', '发病部位']
        self.clinical_data = {"OS": {}, "NOS": {}}


        # Load clinical data
        os_data_path = os.path.join(clinical_data_dir, "OS.XLSX")
        nos_data_path = os.path.join(clinical_data_dir, "NOS.XLSX")

        try:
            self._load_clinical_data(os_data_path, "OS")
        except Exception as e:
            print(f"Error loading OS clinical data: {str(e)}")

        try:
            self._load_clinical_data(nos_data_path, "NOS")
        except Exception as e:
            print(f"Error loading NOS clinical data: {str(e)}")

        # Annotation processor
        self.annotation_processor = AnnotationProcessor(annotation_root)

        self.random_box_mode = random_box_mode
        if self.random_box_mode:
            print(f"Warning: Dataset {split} is in RANDOM BOX MODE. Annotations are FAKE.")

    def _load_clinical_data(self, file_path, label_type):
        """Load clinical data for a specific type (OS or NOS)."""
        if not os.path.exists(file_path):
            print(f"Warning: file not found {file_path}")
            return

        try:
            excel_file = pd.ExcelFile(file_path, engine='openpyxl')
            sheet_names = excel_file.sheet_names

            for sheet_name in sheet_names:
                sheet_data = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')

                if not sheet_data.empty and '姓名' in sheet_data.columns:
                    columns_to_keep = ['姓名'] + self.required_fields
                    available_columns = [col for col in columns_to_keep if col in sheet_data.columns]

                    if not all(field in available_columns for field in self.required_fields):
                        print(f"Warning: sheet '{sheet_name}' is missing required fields")
                        continue

                    sheet_data = sheet_data[available_columns]

                    for _, row in sheet_data.iterrows():
                        patient_name = row['姓名']
                        patient_data = {field: row[field] for field in self.required_fields}
                        self.clinical_data[label_type][patient_name] = patient_data

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            raise

    def __getitem__(self, idx):
        orig_idx, aug_idx = self.augment_indices[idx]
        img_path, label, patient_name = self.image_files[orig_idx]

        try:
            # Load and convert to grayscale
            img = Image.open(img_path).convert('L')
            original_size = img.size

            # Use edge padding instead of square_crop
            img, padding_info = self.annotation_processor.square_pad(img)

            # Apply augmentation or basic transform
            if self.split != 'train' or aug_idx == -1:
                img_tensor = self.basic_transform(img)
            else:
                transform = self.augment_transforms[aug_idx % len(self.augment_transforms)]
                img_tensor = transform(img)

            # Retrieve clinical information
            patient_type = "OS" if label == 1 else "NOS"
            patient_base_name = patient_name.split('_aug')[0] if '_aug' in patient_name else patient_name

            clinical_info = None
            if patient_base_name in self.clinical_data[patient_type]:
                clinical_info = self.clinical_data[patient_type][patient_base_name]
            else:
                other_type = "NOS" if patient_type == "OS" else "OS"
                if patient_base_name in self.clinical_data[other_type]:
                    clinical_info = self.clinical_data[other_type][patient_base_name]

            if clinical_info is None or not all(field in clinical_info for field in self.required_fields):
                return None

            # Load annotation
            img_filename = os.path.basename(img_path)
            region_dict = self.annotation_processor.load_annotation(patient_type, patient_base_name, img_filename)

            if not self.random_box_mode:
                if region_dict is None or len(region_dict) == 0:
                    return None

            # Generate padded segmentation mask
            if self.random_box_mode:
                mask = self.annotation_processor.generate_random_box_mask(
                    self.image_size, min_size=30, max_size=150
                )
                has_annotation = True

            elif region_dict is not None:
                mask = self.annotation_processor.region_dict_to_mask_with_padding(
                    region_dict,
                    original_size,
                    self.image_size
                )
                has_annotation = True
            else:
                # Fallback
                mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
                has_annotation = False

            mask_tensor = torch.from_numpy(mask).float()  # [H, W]

            return {
                'image': img_tensor,
                'label': torch.tensor(label, dtype=torch.long),
                'clinical_info': clinical_info,
                'patient_name': patient_name,
                'filename': img_filename,
                'mask': mask_tensor,
                'has_annotation': has_annotation,
                'augmentation': aug_idx
            }

        except Exception as e:
            print(f"Error loading data for index {idx}: {str(e)}")
            return None


def collate_fn(batch):
    """Collate function for DataLoader."""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    return {
        'image': torch.stack([item['image'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'clinical_info': [item['clinical_info'] for item in batch],
        'patient_name': [item['patient_name'] for item in batch],
        'mask': torch.stack([item['mask'] for item in batch]),
        'has_annotation': torch.tensor([item['has_annotation'] for item in batch], dtype=torch.bool),
        'filename': [item['filename'] for item in batch]
    }


def create_data_loaders(opt):
    """Create data loaders."""

    train_loader = None

    if not opt.eval:
        train_dataset = DDKGDataset(
            root_dir=opt.data_path,
            clinical_data_dir=opt.clinical_data_dir,
            annotation_root=opt.annotation_root,
            split='train',
            image_size=opt.image_size,
            random_box_mode=opt.random_box_experiment,
            active_fields=opt.active_fields
        )

        if len(train_dataset) > 0:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=collate_fn
            )
        else:
            print("Warning: Training dataset is empty!")

    test_dataset = DDKGDataset(
        root_dir=opt.data_path,
        clinical_data_dir=opt.clinical_data_dir,
        annotation_root=opt.annotation_root,
        split='test',
        image_size=opt.image_size,
        active_fields=opt.active_fields
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


class Options:
    def __init__(self):
        self.data_path = '/data/pengxiao/Osteosarcoma_diagnosis/data/internal_DR/DR/'
        self.clinical_data_dir = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息'
        self.annotation_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/internal_DR/DR_annotation'
        self.batch_size = 4
        self.epoch = 100
        self.num_class = 2
        self.base_checkpoint_path = './checkpoints/gating_with2loss_clinicalinfo_bianli/'
        self.eval = False
        self.image_size = 384

        # Swin Transformer typically requires smaller learning rates
        self.learning_rate = 1e-5
        self.weight_decay = 1e-4

        # Loss weights
        self.seg_loss_weight = 1.0
        self.cls_loss_weight = 1.0

        self.use_wandb = False
        self.experiment_name = f'ddkg_swin_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.eval_frequency = 1

        self.swin_pretrained_path = None

        self.random_box_experiment = False

        # Currently active clinical fields
        self.active_fields = ['性别', '年龄', '发病部位']
        self.run_name = "full"

def compute_segmentation_loss(pred_mask, target_mask, has_annotation):
    """
    Compute segmentation loss only for samples with annotations.

    Args:
        pred_mask: Predicted mask [B, H, W]
        target_mask: Target mask [B, H, W]
        has_annotation: Boolean flag per sample [B]
    """
    annotated_indices = torch.where(has_annotation)[0]

    if len(annotated_indices) == 0:
        return torch.tensor(0.0, device=pred_mask.device, requires_grad=True)

    pred_annotated = pred_mask[annotated_indices]
    target_annotated = target_mask[annotated_indices]

    # Binary Cross Entropy Loss
    bce_loss = F.binary_cross_entropy(pred_annotated, target_annotated, reduction='mean')

    # Dice Loss
    smooth = 1e-6
    pred_flat = pred_annotated.view(-1)
    target_flat = target_annotated.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice_score = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_score

    return bce_loss + dice_loss


def train_and_evaluate(opt):
    """Training and evaluation function."""
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if opt.use_wandb:
        wandb.init(project="DR_DDKG_Swin", name=opt.experiment_name, config=opt)

    # Create data loaders
    train_loader, test_loader = create_data_loaders(opt)

    if not train_loader or not test_loader:
        print("Error: Failed to create data loaders. Exiting...")
        return

    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # Create model
    model = DDKGModel(
        num_classes=opt.num_class,
        image_size=opt.image_size,
        clinical_dim=256,
        swin_pretrained_path=opt.swin_pretrained_path
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer and loss function
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': opt.learning_rate * 0.1},
        {'params': other_params, 'lr': opt.learning_rate}
    ], weight_decay=opt.weight_decay)
    count_nos = 861
    count_os = 2112
    total_samples = count_nos + count_os

    # Class-balanced weights: Total / (Num_Classes * Count)
    weight_nos = total_samples / (2 * count_nos)
    weight_os = total_samples / (2 * count_os)

    class_weights = torch.tensor([weight_nos, weight_os], dtype=torch.float).to(device)

    print(f"Applying class-balanced weights -> NOS(0): {weight_nos:.4f}, OS(1): {weight_os:.4f}")

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

    best_auc = 0.0
    global_step = 0

    def evaluate_model():
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                if data is None:
                    continue

                images = data['image'].to(device)
                labels = data['label'].to(device)
                masks = data['mask'].to(device)
                clinical_info = data['clinical_info']

                # Forward pass
                outputs = model(images, clinical_info)

                cls_loss = cls_criterion(outputs['classification'], labels)
                seg_loss = compute_segmentation_loss(
                    outputs['segmentation'].squeeze(1),
                    masks,
                    data['has_annotation'].to(device)
                )

                total_loss_batch = (opt.cls_loss_weight * cls_loss +
                                    opt.seg_loss_weight * seg_loss)

                # Statistics
                _, predicted = torch.max(outputs['classification'], 1)
                probs = F.softmax(outputs['classification'], dim=1)[:, 1]

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += total_loss_batch.item()
                total_seg_loss += seg_loss.item()
                total_cls_loss += cls_loss.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

                if batch_idx % 2 == 0:
                    print(f'TEST [batch {batch_idx + 1}/{len(test_loader)}]: '
                          f'loss={total_loss_batch.item():.4f}, acc={(correct / total * 100):.3f}%')

        # Compute metrics
        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0

        if len(y_true) > 0 and len(y_pred) > 0:
            cm = confusion_matrix(y_true, y_pred)
            print("\nConfusion Matrix:")
            print("Predicted:  NOS  OS")
            print(f"True NOS: {cm[0][0]:4d} {cm[0][1]:4d}")
            print(f"     OS: {cm[1][0]:4d} {cm[1][1]:4d}")
            print(f"\nAccuracy: {accuracy:.4f}, AUC: {auc:.4f}, Avg Val Loss: {avg_loss:.4f}")

            if opt.use_wandb:
                wandb.log({
                    "Test/Accuracy": accuracy,
                    "Test/AUC": auc,
                    "Test/Average_Loss": avg_loss,
                    "Test/Seg_Loss": total_seg_loss / len(test_loader),
                    "Test/Cls_Loss": total_cls_loss / len(test_loader),
                })

        return accuracy, avg_loss, auc

    if opt.eval:
        print("Evaluation mode: running evaluation only")
        evaluate_model()
        return

    # Training loop
    print(f"\nStarting training: {opt.epoch} epochs, {len(train_loader)} batches per epoch")
    print(f"Backbone: Swin Transformer Large")

    for epoch in range(opt.epoch):
        model.train()
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start_time = time.time()

        for batch_idx, data in enumerate(train_loader):
            if data is None:
                continue

            start_time = time.time()

            images = data['image'].to(device)
            labels = data['label'].to(device)
            masks = data['mask'].to(device)
            clinical_info = data['clinical_info']

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, clinical_info)

            cls_loss = cls_criterion(outputs['classification'], labels)
            seg_loss = compute_segmentation_loss(
                outputs['segmentation'].squeeze(1),
                masks,
                data['has_annotation'].to(device)
            )

            total_loss = (opt.cls_loss_weight * cls_loss +
                          opt.seg_loss_weight * seg_loss)

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            _, predicted = torch.max(outputs['classification'], 1)
            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_accuracy = batch_correct / batch_total if batch_total > 0 else 0

            epoch_loss += total_loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_correct += batch_correct
            epoch_total += batch_total

            batch_time = time.time() - start_time
            global_step += 1

            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'TRAIN [epoch {epoch + 1}/{opt.epoch}, batch {batch_idx + 1}/{len(train_loader)}]: '
                      f'loss={total_loss.item():.4f}, acc={batch_accuracy:.4f}, '
                      f'lr={current_lr:.2e}, time={batch_time:.2f}s')

            if opt.use_wandb:
                wandb.log({
                    "Train/Loss": total_loss.item(),
                    "Train/Seg_Loss": seg_loss.item(),
                    "Train/Cls_Loss": cls_loss.item(),
                    "Train/Batch_Accuracy": batch_accuracy,
                    "Train/Learning_Rate": optimizer.param_groups[0]['lr'],
                    "Train/Global_Step": global_step
                })

        # Update learning rate
        scheduler.step()

        # Compute and print per-epoch statistics
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0

        print(f'\nEpoch {epoch + 1} Summary - '
              f'Time: {epoch_time:.2f}s, '
              f'Avg Loss: {epoch_avg_loss:.4f}, '
              f'Avg Acc: {epoch_avg_acc:.4f}, '
              f'Seg Loss: {epoch_seg_loss / len(train_loader):.4f}, '
              f'Cls Loss: {epoch_cls_loss / len(train_loader):.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

        if opt.use_wandb:
            wandb.log({
                "Train/Epoch": epoch + 1,
                "Train/Epoch_Loss": epoch_avg_loss,
                "Train/Epoch_Accuracy": epoch_avg_acc,
                "Train/Epoch_Time": epoch_time
            })

        # Evaluate at specified frequency
        if (epoch + 1) % opt.eval_frequency == 0:
            print(f"\nEvaluating - Epoch {epoch + 1}/{opt.epoch}")
            accuracy, avg_loss, auc = evaluate_model()

            if opt.use_wandb:
                wandb.log({
                    "Eval/Epoch": epoch + 1,
                    "Eval/Accuracy": accuracy,
                    "Eval/AUC": auc,
                    "Eval/Loss": avg_loss
                })

            # Save best model
            if auc > best_auc:
                best_auc = auc
                save_path = os.path.join(opt.checkpoint_path, "ddkg_swin_model_gating",
                                         f'best_swin_model_epoch{epoch + 1}_acc{accuracy:.3f}_auc{auc:.3f}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': accuracy,
                    'auc': auc,
                    'global_step': global_step,
                    'config': opt.__dict__,
                }, save_path)
                print(f'=> Saved best Swin model to {save_path}')


    # Final evaluation after training
    print("\nTraining complete. Running final evaluation...")
    final_accuracy, final_avg_loss, final_auc = evaluate_model()
    print(f"Final Swin model - Accuracy: {final_accuracy:.4f}, AUC: {final_auc:.4f}, Avg Loss: {final_avg_loss:.4f}")

    print(f"\nGenerating final Swin training report and visualization...")
    final_vis_path = os.path.join(opt.checkpoint_path, "final_swin_training_report")
    os.makedirs(final_vis_path, exist_ok=True)

    # Save training summary
    summary_path = os.path.join(final_vis_path, "swin_training_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("DDKG Swin Transformer Training Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Backbone: Swin Transformer Large\n")
        f.write(f"Training epochs: {opt.epoch}\n")
        f.write(f"Batch size: {opt.batch_size}\n")
        f.write(f"Learning rate: {opt.learning_rate}\n")
        f.write(f"Image size: {opt.image_size}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")
        f.write("Final Performance Metrics:\n")
        f.write(f"  Accuracy: {final_accuracy:.4f}\n")
        f.write(f"  AUC: {final_auc:.4f}\n")
        f.write(f"  Average Loss: {final_avg_loss:.4f}\n")
        f.write(f"Training samples: {len(train_loader.dataset)}\n")
        f.write(f"Test samples: {len(test_loader.dataset)}\n")
        if opt.swin_pretrained_path:
            f.write(f"Pretrained weights: {opt.swin_pretrained_path}\n")

    print(f"Final Swin report saved at: {final_vis_path}")
    print(f"Training summary: {summary_path}")

    if opt.use_wandb:
        wandb.log({
            "Final/Accuracy": final_accuracy,
            "Final/AUC": final_auc,
            "Final/Loss": final_avg_loss,
        })
        wandb.finish()


import copy
import gc


def main():
    """Main entry point - auto-runs clinical info ablation experiments."""
    base_opt = Options()

    # Check required paths
    required_paths = [base_opt.data_path, base_opt.clinical_data_dir, base_opt.annotation_root]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Error: path does not exist - {path}")
            return

    if base_opt.eval:
        print("Currently only training mode is supported for auto ablation experiments.")
        return

    ablation_experiments = {
        "7_all_features": ['性别', '年龄', '发病部位'],
        "1_only_gender": ['性别'],
        "2_only_age": ['年龄'],
        "3_only_location": ['发病部位'],
        "4_gender_age": ['性别', '年龄'],
        "5_gender_location": ['性别', '发病部位'],
        "6_age_location": ['年龄', '发病部位'],

    }

    print("=" * 60)
    print(f"Starting DDKG Swin clinical info ablation experiments")
    print(f"Planned {len(ablation_experiments)} groups...")
    print("=" * 60)

    # Loop over experimental groups
    for run_name, fields in ablation_experiments.items():
        print("\n\n" + "#" * 60)
        print(f"Starting experiment group: {run_name}")
        print(f"Active clinical features: {fields}")
        print("#" * 60)

        # Deep copy config to prevent cross-contamination
        opt = copy.deepcopy(base_opt)

        # Modify experiment-specific config
        opt.active_fields = fields
        opt.run_name = run_name
        opt.checkpoint_path = os.path.join(opt.base_checkpoint_path, run_name)
        opt.experiment_name = f'ablation_{run_name}_{datetime.now().strftime("%Y%m%d_%H%M")}'

        os.makedirs(opt.checkpoint_path, exist_ok=True)

        # Print dataset stats for this config
        train_loader, test_loader = create_data_loaders(opt)
        print(f"Training set valid samples: {len(train_loader.dataset)}")
        print(f"Test set valid samples: {len(test_loader.dataset)}")

        # Execute training
        try:
            train_and_evaluate(opt)
        except Exception as e:
            print(f"Experiment group {run_name} error: {str(e)}")
            continue

        print(f"Experiment group {run_name} completed!")

        # Clean up VRAM and memory
        del train_loader
        del test_loader
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll ablation experiments completed!")
    print(f"Model weights and results saved at: {base_opt.base_checkpoint_path}")


if __name__ == '__main__':
    main()
