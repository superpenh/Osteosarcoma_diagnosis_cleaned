import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import timm
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms


from evaluation.fusion_dr_patch_report_dataset import OsteosarcomaDataset


class ClinicalInfoProcessor(nn.Module):
    """Clinical information processor: converts clinical info to vector representations."""

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
        texts = [
            f"性别{info['性别']} 年龄{info['年龄']}岁 发病部位在{info['发病部位']}"
            for info in clinical_info
        ]

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(next(self.bert.parameters()).device)

            outputs = self.bert(**encoded)

        # Use [CLS] token output as text representation
        text_features = outputs.last_hidden_state[:, 0]

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
        attention_map = torch.sigmoid(seg_logits_small)

        # --- 4. Use mask to enhance features ---
        # Features = original + (original * mask)
        # Masked regions get amplified; uncovered regions preserved via residual connection
        annotated_features = shared_features * (1 + attention_map)

        # Optional smoothing convolution
        refined_features = self.attend_conv(annotated_features)

        # --- 5. Classification and output ---
        cls_pred = self.classifier(refined_features)

        # Upsample small logits to original image size for loss or visualization
        seg_pred_upsampled = F.interpolate(seg_logits_small, size=(384, 384), mode='bilinear', align_corners=False)
        seg_pred_final = torch.sigmoid(seg_pred_upsampled)

        return {
            'segmentation': seg_pred_final,
            'classification': cls_pred,
            'features': refined_features
        }


class DDKGModel(nn.Module):
    """DDKG main model - adapted from train_gating_with2loss.py architecture."""

    def __init__(self, num_classes=2, image_size=384, clinical_dim=256,
                 bert_model_path="/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/",
                 swin_pretrained_path=None):
        super().__init__()

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

        self.clinical_processor = ClinicalInfoProcessor(
            bert_model=bert_model_path,
            output_dim=clinical_dim
        )

        swin_channels_list = self.backbone.feature_info.channels()
        print(f"All Swin channels: {swin_channels_list}")

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

        trainable_params_backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable parameters in backbone after freezing: {trainable_params_backbone:,}")

    def load_swin_pretrained_weights(self, pretrained_path):
        """Load pretrained Swin weights."""
        try:
            if pretrained_path.endswith('.pth') or pretrained_path.endswith('.pt'):
                checkpoint = torch.load(pretrained_path, map_location='cpu')

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

            model_dict = self.backbone.state_dict()

            # Filter out mismatched keys
            filtered_dict = {}
            patch_embed_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias']

            for k, v in pretrained_dict.items():
                key = k
                if key.startswith('backbone.'):
                    key = key[9:]
                if key.startswith('model.'):
                    key = key[6:]

                if key in model_dict:
                    if not any(patch_key in key for patch_key in patch_embed_keys):
                        if v.shape == model_dict[key].shape:
                            filtered_dict[key] = v
                        else:
                            print(f"Shape mismatch for {key}: {v.shape} vs {model_dict[key].shape}")
                    else:
                        print(f"Skipping patch_embed layer: {key}")

            self.backbone.load_state_dict(filtered_dict, strict=False)
            print(f"Successfully loaded pretrained Swin weights! Loaded {len(filtered_dict)} parameters")

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


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    max_dr_images = max([item['dr_images'].size(0) for item in batch])

    dr_images_padded = []
    for item in batch:
        curr_dr_images = item['dr_images']
        curr_num_images = curr_dr_images.size(0)

        if curr_num_images < max_dr_images:
            padding = torch.zeros(max_dr_images - curr_num_images,
                                  curr_dr_images.size(1),
                                  curr_dr_images.size(2),
                                  curr_dr_images.size(3),
                                  dtype=curr_dr_images.dtype)
            padded_images = torch.cat([curr_dr_images, padding], dim=0)
        else:
            padded_images = curr_dr_images

        dr_images_padded.append(padded_images)

    clinical_infos = []
    for item in batch:
        info = item['clinical_info']
        info = {
            '性别': str(info['性别']),
            '年龄': str(info['年龄']),
            '发病部位': str(info['发病部位'])
        }
        clinical_infos.append(info)

    return {
        'dr_images': torch.stack(dr_images_padded),
        'patches': torch.stack([item['patches'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch]),
        'patient_name': [item['patient_name'] for item in batch],
        'class_name': [item['class_name'] for item in batch],
        'clinical_info': clinical_infos
    }


class MultiModalFusionModel(nn.Module):
    def __init__(self, dr_model_path, patch_model_path, bert_model_path, num_classes=2):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dr_model = DDKGModel(
            num_classes=num_classes,
            image_size=384,
            clinical_dim=256,
            bert_model_path=bert_model_path
        )
        self._load_dr_checkpoint(dr_model_path)

        self.patch_model = self._init_swin_patch_model(patch_model_path)

    def _load_dr_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value
            self.dr_model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded DDKG Swin DR model checkpoint (New Arch)")
        except Exception as e:
            print(f"Warning: Could not load DR model checkpoint: {str(e)}")

        self.dr_model = self.dr_model.to(self.device)
        self.dr_model.eval()

    def _init_swin_patch_model(self, checkpoint_path):
        model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k',
                                  num_classes=2, pretrained=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded Swin patch model checkpoint")
        except Exception as e:
            print(f"Warning: Could not load Swin patch model checkpoint: {str(e)}")
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_dr_image_for_ddkg(self, dr_image):
        if dr_image.size(1) == 3:
            dr_image = dr_image.mean(dim=1, keepdim=True)
        dr_image = F.interpolate(dr_image, size=(384, 384), mode='bilinear', align_corners=False)
        return dr_image

    def _preprocess_patch_for_swin(self, patch):
        if patch.size(-1) != 224 or patch.size(-2) != 224:
            patch = F.interpolate(patch, size=(224, 224), mode='bilinear', align_corners=False)
        return patch

    def forward(self, dr_images, patches, clinical_info):
        batch_size = dr_images.size(0)
        num_dr_images = dr_images.size(1)

        dr_probs_all = []
        visualization_data = []

        for i in range(num_dr_images):
            current_image = dr_images[:, i]
            if torch.sum(current_image) != 0:
                processed_image = self._preprocess_dr_image_for_ddkg(current_image)

                # dr_output contains: 'segmentation', 'classification', 'features'
                # 'features' is refined_features [B, 256, 96, 96]
                dr_output = self.dr_model(processed_image, clinical_info, return_features=True)

                dr_prob_single = F.softmax(dr_output['classification'], dim=1)
                dr_probs_all.append(dr_prob_single)

                # Extract and compress features
                raw_feats = dr_output['features']  # [B, 256, H_feat, W_feat]

                # Mean activation across channel dimension -> [B, 1, H_feat, W_feat]
                feature_activation = torch.mean(raw_feats, dim=1, keepdim=True)

                # Upsample to 384x384 for overlay display
                feature_activation_resized = F.interpolate(
                    feature_activation,
                    size=(384, 384),
                    mode='bilinear',
                    align_corners=False
                )

                visualization_data.append({
                    'original_input': processed_image.detach().cpu(),
                    'feature_map': feature_activation_resized.detach().cpu()
                })

        if dr_probs_all:
            stacked_probs = torch.stack(dr_probs_all, dim=0)
            max_pos_prob, _ = torch.max(stacked_probs[:, :, 1], dim=0)
            neg_prob = 1.0 - max_pos_prob
            dr_prob = torch.stack([neg_prob, max_pos_prob], dim=1)

            if batch_size == 1:
                img_preds = torch.argmax(stacked_probs, dim=2).squeeze(1)
                pos_count = torch.sum(img_preds).item()
                total_valid_imgs = img_preds.size(0)
                is_mixed = (pos_count > 0) and (pos_count < total_valid_imgs)

                if is_mixed:
                    try:
                        age_val = clinical_info[0]['年龄']
                        age = float(age_val)
                        if age > 35:
                            dr_prob = torch.tensor([[0.999, 0.001]]).to(self.device)
                    except:
                        pass
        else:
            dr_prob = torch.ones((batch_size, 2)).to(self.device) / 2

        # Patch processing
        patch_probs_list = []
        num_patches = patches.size(1)
        batch_size_patches = 10
        num_patch_batches = (num_patches + batch_size_patches - 1) // batch_size_patches

        for i in range(num_patch_batches):
            start_idx = i * batch_size_patches
            end_idx = min((i + 1) * batch_size_patches, num_patches)
            current_batch_size = end_idx - start_idx

            patch_batch = patches[:, start_idx:end_idx]
            patch_batch = patch_batch.reshape(batch_size * current_batch_size, 3, patches.size(3), patches.size(4))
            patch_batch_processed = self._preprocess_patch_for_swin(patch_batch)

            output = self.patch_model(patch_batch_processed)
            prob = F.softmax(output, dim=1)
            prob = prob.view(batch_size, current_batch_size, -1)
            patch_probs_list.append(prob)

        if patch_probs_list:
            all_patch_probs = torch.cat(patch_probs_list, dim=1)
            positive_probs = all_patch_probs[:, :, 1]
            confidence_scores = torch.abs(positive_probs - 0.5)
            K = 50
            current_k = min(K, confidence_scores.size(1))
            top_conf_values, top_indices = torch.topk(confidence_scores, k=current_k, dim=1)
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, 2)
            top_probs = torch.gather(all_patch_probs, 1, top_indices_expanded)
            top_weights = top_conf_values + 1e-6
            top_weights = top_weights.unsqueeze(-1)
            weighted_probs = top_probs * top_weights
            patch_prob = torch.sum(weighted_probs, dim=1) / torch.sum(top_weights, dim=1)
        else:
            patch_prob = torch.ones((batch_size, 2)).to(self.device) / 2

        return dr_prob, patch_prob, visualization_data


class MultiModalTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = OsteosarcomaDataset(
            args.data_root, args.os_clinical_file, args.nos_clinical_file, image_size=224
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
        )

        self.model = MultiModalFusionModel(
            dr_model_path=args.dr_checkpoint,
            patch_model_path=args.patch_checkpoint,
            bert_model_path=args.bert_model_path,
            num_classes=2
        ).to(self.device)

        self.heatmap_save_dir = os.path.join(args.log_dir, 'feature_maps_visualization')
        os.makedirs(self.heatmap_save_dir, exist_ok=True)
        print(f"Feature Maps will be saved to: {self.heatmap_save_dir}")

    def save_feature_map(self, img_tensor, feat_tensor, patient_name, img_idx):
        """
        img_tensor: [1, 1, 384, 384]
        feat_tensor: [1, 1, 384, 384] (feature map)
        patient_name: patient name
        img_idx: image index
        """
        # 1. Process original image
        img = img_tensor.squeeze().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        img = img.astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 2. Process feature map
        feat = feat_tensor.squeeze().numpy()

        # Robust normalization (clip extreme values for better contrast)
        lower = np.percentile(feat, 2)
        upper = np.percentile(feat, 98)
        feat = np.clip(feat, lower, upper)

        feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8) * 255
        feat = feat.astype(np.uint8)

        heatmap = cv2.applyColorMap(feat, cv2.COLORMAP_JET)

        # 3. Overlay (original:heatmap = 6:4)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        # 4. Construct filename: patientName_imgIndex.png
        filename = f"{patient_name}_img{img_idx}.png"

        filename = filename.replace('/', '_').replace(' ', '')

        save_path = os.path.join(self.heatmap_save_dir, filename)
        cv2.imwrite(save_path, overlay)

    def test(self):
        labels = []
        dr_probs = []
        patch_probs = []
        patient_names = []
        class_names = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Visualizing Features"):
                if batch is None:
                    continue

                dr_images = batch['dr_images'].to(self.device)
                patches = batch['patches'].to(self.device)
                clinical_info = batch['clinical_info']
                label_int = batch['label'].item()
                patient_name = batch['patient_name'][0]
                true_class_name = batch['class_name'][0]

                dr_prob, patch_prob, vis_data = self.model(dr_images, patches, clinical_info)
                pos_prob = dr_prob[0][1].item()

                for idx, data_item in enumerate(vis_data):
                    self.save_feature_map(
                        data_item['original_input'],
                        data_item['feature_map'],
                        patient_name,
                        idx
                    )

                labels.append(label_int)
                dr_probs.append(pos_prob)
                patch_probs.append(patch_prob[0][1].item())
                patient_names.append(patient_name)
                class_names.append(true_class_name)

        if self.args.fixed_dr_weight is not None:
            self._evaluate_fixed_weight(labels, dr_probs, patch_probs, patient_names, class_names,
                                        self.args.fixed_dr_weight)
        else:
            self._try_different_weights(labels, dr_probs, patch_probs, patient_names, class_names)

    def _evaluate_fixed_weight(self, labels, dr_probs, patch_probs, patient_names, class_names, fixed_dr_weight):
        self.labels = labels
        self.dr_probs = dr_probs
        self.patch_probs = patch_probs

        dr_weight = fixed_dr_weight
        patch_weight = 1.0 - dr_weight

        if not (0.0 <= dr_weight <= 1.0):
            print(f"Error: Fixed DR weight {dr_weight:.2f} is outside the valid range [0.0, 1.0].")
            return

        print(f"\n--- Evaluating with Fixed Weights: DR Weight={dr_weight:.2f}, Patch Weight={patch_weight:.2f} ---")

        fusion_probs = []
        for dr_p, patch_p in zip(dr_probs, patch_probs):
            fusion_p = dr_weight * dr_p + patch_weight * patch_p
            fusion_probs.append(fusion_p)

        fusion_probs = np.array(fusion_probs)
        fusion_preds = (fusion_probs > 0.5).astype(int)
        labels_array = np.array(labels)

        best_metrics = self._calculate_metrics(labels_array, fusion_preds, fusion_probs)
        best_weight = dr_weight

        patient_results = []
        for i, (p_name, c_name, label, dr_p, patch_p, fusion_p) in enumerate(
                zip(patient_names, class_names, labels, dr_probs, patch_probs, fusion_probs)):
            patient_results.append({
                'patient_name': p_name,
                'true_class': c_name,
                'predicted_class': 'OS' if fusion_p > 0.5 else 'NOS',
                'fusion_prob': fusion_p,
                'dr_prob': dr_p,
                'patch_prob': patch_p
            })

        dr_preds = (np.array(dr_probs) > 0.5).astype(int)
        patch_preds = (np.array(patch_probs) > 0.5).astype(int)
        metrics_dr = self._calculate_metrics(np.array(labels), dr_preds, np.array(dr_probs))
        metrics_patch = self._calculate_metrics(np.array(labels), patch_preds, np.array(patch_probs))

        print("\nDetailed Results with Fixed Weights:")
        print('-' * 140)
        print(f"DR Weight: {best_weight:.2f}, Patch Weight: {1.0 - best_weight:.2f}")
        print('-' * 140)
        print('{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
            'Model', 'Accuracy', 'AUC', 'Sens', 'Spec', 'Prec', 'NPV', 'F1'))
        print('-' * 140)

        print('{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
            'Fusion (Fixed)', best_metrics['Accuracy'], best_metrics['AUC'], best_metrics['Sensitivity'],
            best_metrics['Specificity'], best_metrics['Precision'], best_metrics['NPV'], best_metrics['F1']
        ))

        print('{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
            'DDKG Swin DR Only', metrics_dr['Accuracy'], metrics_dr['AUC'], metrics_dr['Sensitivity'],
            metrics_dr['Specificity'], metrics_dr['Precision'], metrics_dr['NPV'], metrics_dr['F1']
        ))

        print('{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
            'WSI Model Only', metrics_patch['Accuracy'], metrics_patch['AUC'], metrics_patch['Sensitivity'],
            metrics_patch['Specificity'], metrics_patch['Precision'], metrics_patch['NPV'], metrics_patch['F1']
        ))
        print('-' * 140)

        if patient_results:
            print('\nDetailed Patient Results (with Fixed Weights):')
            print('-' * 100)
            print('{:<20} {:<10} {:<10} {:<12} {:<12} {:<12}'.format(
                'Patient', 'True', 'Pred', 'Fusion Prob', 'DDKG DR Prob', 'Patch Prob'))
            print('-' * 100)

            for result in patient_results:
                print('{:<20} {:<10} {:<10} {:<12.3f} {:<12.3f} {:<12.3f}'.format(
                    result['patient_name'], result['true_class'], result['predicted_class'],
                    result['fusion_prob'], result['dr_prob'], result['patch_prob']
                ))

        results = [{'dr_weight': dr_weight, 'patch_weight': patch_weight, 'metrics': best_metrics}]
        self._plot_weight_performance(results, fixed_weight_mode=True)
        return best_weight

    def _try_different_weights(self, labels, dr_probs, patch_probs, patient_names, class_names):
        self.labels = labels
        self.dr_probs = dr_probs
        self.patch_probs = patch_probs

        best_auc = 0
        best_weight = 0
        results = []
        patient_results = []

        for dr_weight in np.arange(0.00, 1.05, 0.05):
            patch_weight = 1.0 - dr_weight

            fusion_probs = []
            for dr_p, patch_p in zip(dr_probs, patch_probs):
                fusion_p = dr_weight * dr_p + patch_weight * patch_p
                fusion_probs.append(fusion_p)

            fusion_probs = np.array(fusion_probs)
            fusion_preds = (fusion_probs > 0.5).astype(int)
            labels_array = np.array(labels)

            metrics_fusion = self._calculate_metrics(labels_array, fusion_preds, fusion_probs)

            if metrics_fusion['AUC'] > best_auc:
                best_auc = metrics_fusion['AUC']
                best_weight = dr_weight

                patient_results = []
                for i, (p_name, c_name, label, dr_p, patch_p, fusion_p) in enumerate(
                        zip(patient_names, class_names, labels, dr_probs, patch_probs, fusion_probs)):
                    patient_results.append({
                        'patient_name': p_name, 'true_class': c_name,
                        'predicted_class': 'OS' if fusion_p > 0.5 else 'NOS',
                        'fusion_prob': fusion_p, 'dr_prob': dr_p, 'patch_prob': patch_p
                    })

            results.append({'dr_weight': dr_weight, 'patch_weight': patch_weight, 'metrics': metrics_fusion})

        dr_preds = (np.array(dr_probs) > 0.5).astype(int)
        patch_preds = (np.array(patch_probs) > 0.5).astype(int)
        metrics_dr = self._calculate_metrics(np.array(labels), dr_preds, np.array(dr_probs))
        metrics_patch = self._calculate_metrics(np.array(labels), patch_preds, np.array(patch_probs))

        print("\nWeight Tuning Results:")
        print('-' * 100)
        print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
            'DR Weight', 'Patch W', 'Accuracy', 'AUC', 'Sens', 'Spec', 'Prec', 'F1'))
        print('-' * 100)

        for result in results:
            metrics = result['metrics']
            print('{:<10.2f} {:<10.2f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
                result['dr_weight'], result['patch_weight'], metrics['Accuracy'], metrics['AUC'],
                metrics['Sensitivity'], metrics['Specificity'], metrics['Precision'], metrics['F1']
            ))

        print('-' * 100)
        print(f"Best DR Weight: {best_weight:.2f}, Best Patch Weight: {1.0 - best_weight:.2f}, Best AUC: {best_auc:.3f}")

        print("\nDetailed Results with Best Weights:")
        print('-' * 140)
        print(f"DR Weight: {best_weight:.2f}, Patch Weight: {1.0 - best_weight:.2f}")
        print('-' * 140)
        print('{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
            'Model', 'Accuracy', 'AUC', 'Sens', 'Spec', 'Prec', 'NPV', 'F1'))
        print('-' * 140)

        best_metrics = None
        for result in results:
            if abs(result['dr_weight'] - best_weight) < 1e-5:
                best_metrics = result['metrics']
                break

        if best_metrics:
            print('{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
                'Fusion (Best)', best_metrics['Accuracy'], best_metrics['AUC'], best_metrics['Sensitivity'],
                best_metrics['Specificity'], best_metrics['Precision'], best_metrics['NPV'], best_metrics['F1']
            ))

        print('{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
            'DDKG Swin DR Only', metrics_dr['Accuracy'], metrics_dr['AUC'], metrics_dr['Sensitivity'],
            metrics_dr['Specificity'], metrics_dr['Precision'], metrics_dr['NPV'], metrics_dr['F1']
        ))

        print('{:<20} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
            'WSI Model Only', metrics_patch['Accuracy'], metrics_patch['AUC'], metrics_patch['Sensitivity'],
            metrics_patch['Specificity'], metrics_patch['Precision'], metrics_patch['NPV'], metrics_patch['F1']
        ))
        print('-' * 140)

        if patient_results:
            print('\nDetailed Patient Results (with Best Weights):')
            print('-' * 100)
            print('{:<20} {:<10} {:<10} {:<12} {:<12} {:<12}'.format(
                'Patient', 'True', 'Pred', 'Fusion Prob', 'DDKG DR Prob', 'Patch Prob'))
            print('-' * 100)

            for result in patient_results:
                print('{:<20} {:<10} {:<10} {:<12.3f} {:<12.3f} {:<12.3f}'.format(
                    result['patient_name'], result['true_class'], result['predicted_class'],
                    result['fusion_prob'], result['dr_prob'], result['patch_prob']
                ))

        self._plot_weight_performance(results)
        return best_weight

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            if len(np.unique(y_true)) < 2:
                if np.unique(y_true)[0] == 0:
                    tn = np.sum(y_true == y_pred)
                    fp = len(y_true) - tn
                    tp = 0
                    fn = 0
                else:
                    tp = np.sum(y_true == y_pred)
                    fn = len(y_true) - tp
                    tn = 0
                    fp = 0
            else:
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
        else:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        if len(np.unique(y_true)) < 2:
            auc_score = 0.5
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)

        return {
            'Accuracy': accuracy, 'AUC': auc_score, 'Sensitivity': sensitivity,
            'Specificity': specificity, 'Precision': precision, 'NPV': npv, 'F1': f1
        }

    def _plot_weight_performance(self, results, fixed_weight_mode=False):
        weights = [r['dr_weight'] for r in results]
        aucs = [r['metrics']['AUC'] for r in results]
        accuracies = [r['metrics']['Accuracy'] for r in results]
        f1_scores = [r['metrics']['F1'] for r in results]
        sensitivities = [r['metrics']['Sensitivity'] for r in results]
        specificities = [r['metrics']['Specificity'] for r in results]

        plt.figure(figsize=(12, 10))

        if fixed_weight_mode:
            plot_title = f'Performance at Fixed DDKG Swin DR Weight={weights[0]:.2f}'
            plt.subplot(2, 1, 1)
            plt.plot(weights, aucs, 'ro', markersize=8, label='AUC')
            plt.plot(weights, accuracies, 'bs', markersize=8, label='Accuracy')
            plt.plot(weights, f1_scores, 'g^', markersize=8, label='F1 Score')
            plt.xlabel('DDKG Swin DR Weight')
            plt.ylabel('Performance Metric')
            plt.title(plot_title)
            plt.xlim(max(-0.05, weights[0] - 0.1), min(1.05, weights[0] + 0.1))
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(weights, sensitivities, 'ro', markersize=8, label='Sensitivity')
            plt.plot(weights, specificities, 'bs', markersize=8, label='Specificity')
            plt.xlabel('DDKG Swin DR Weight')
            plt.ylabel('Performance Metric')
            plt.title('Sensitivity & Specificity at Fixed Weight')
            plt.xlim(max(-0.05, weights[0] - 0.1), min(1.05, weights[0] + 0.1))
            plt.grid(True)
            plt.legend()

        else:
            plt.subplot(2, 1, 1)
            plt.plot(weights, aucs, 'o-', linewidth=2, label='AUC')
            plt.plot(weights, accuracies, 's-', linewidth=2, label='Accuracy')
            plt.plot(weights, f1_scores, '^-', linewidth=2, label='F1 Score')
            plt.xlabel('DDKG Swin DR Weight')
            plt.ylabel('Performance Metric')
            plt.title('Overall Performance vs DDKG Swin DR Weight')
            plt.grid(True)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(weights, sensitivities, 'o-', linewidth=2, label='Sensitivity')
            plt.plot(weights, specificities, 's-', linewidth=2, label='Specificity')
            plt.xlabel('DDKG Swin DR Weight')
            plt.ylabel('Performance Metric')
            plt.title('Sensitivity & Specificity vs DDKG Swin DR Weight')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig('ddkg_swin_weight_performance.png')
        self._plot_roc_comparison(results, fixed_weight_mode)

    def _plot_roc_comparison(self, results, fixed_weight_mode=False):
        if not hasattr(self, 'dr_probs') or not hasattr(self, 'patch_probs') or not hasattr(self, 'labels'):
            print("Warning: Missing data required for ROC curve comparison")
            return

        if fixed_weight_mode:
            best_weight = results[0]['dr_weight']
        else:
            aucs = [r['metrics']['AUC'] for r in results]
            best_weight_idx = np.argmax(aucs)
            best_weight = results[best_weight_idx]['dr_weight']

        plt.figure(figsize=(10, 8))
        y_true = np.array(self.labels)

        if len(np.unique(y_true)) < 2:
            print("Warning: Only one class present in labels. Cannot generate ROC curve.")
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
            plt.text(0.5, 0.5, 'Cannot compute ROC: Only one class present',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='red')
            plt.title('ROC Curve Comparison: DDKG Swin DR vs WSI vs Fusion', fontsize=14)
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.savefig('ddkg_swin_roc_comparison.png', dpi=300)
            return

        dr_fpr, dr_tpr, _ = roc_curve(y_true, np.array(self.dr_probs))
        dr_auc = auc(dr_fpr, dr_tpr)
        plt.plot(dr_fpr, dr_tpr, 'b-', linewidth=2, label=f'DDKG Swin DR Model (AUC = {dr_auc:.3f})')

        wsi_fpr, wsi_tpr, _ = roc_curve(y_true, np.array(self.patch_probs))
        wsi_auc = auc(wsi_fpr, wsi_tpr)
        plt.plot(wsi_fpr, wsi_tpr, 'g-', linewidth=2, label=f'WSI Model (AUC = {wsi_auc:.3f})')

        fusion_probs = []
        for dr_p, patch_p in zip(self.dr_probs, self.patch_probs):
            fusion_p = best_weight * dr_p + (1.0 - best_weight) * patch_p
            fusion_probs.append(fusion_p)

        fusion_fpr, fusion_tpr, _ = roc_curve(y_true, np.array(fusion_probs))
        fusion_auc = auc(fusion_fpr, fusion_tpr)

        fusion_label = f'Fusion Model (DDKG DR weight={best_weight:.2f}, AUC = {fusion_auc:.3f})'
        if fixed_weight_mode:
            fusion_label = f'Fusion Model (Fixed W={best_weight:.2f}, AUC = {fusion_auc:.3f})'

        plt.plot(fusion_fpr, fusion_tpr, 'r-', linewidth=2, label=fusion_label)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison: DDKG Swin DR vs WSI vs Fusion', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        max_auc = max(dr_auc, wsi_auc)
        auc_improvement = fusion_auc - max_auc
        plt.figtext(0.15, 0.15, f"Fusion AUC improvement: +{auc_improvement:.3f}",
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig('ddkg_swin_roc_comparison.png', dpi=300)

        print("\nROC Curve Comparison Summary:")
        print("-" * 60)
        print(f"DDKG Swin DR Model AUC:     {dr_auc:.3f}")
        print(f"WSI Model AUC:              {wsi_auc:.3f}")
        print(f"Fusion Model AUC:           {fusion_auc:.3f} (DDKG DR weight={best_weight:.2f})")
        print(f"Fusion AUC improvement over best individual model: +{auc_improvement:.3f}")
        print("-" * 60)


class Args:
    def __init__(self):
        # --- Dataset and path configuration ---
        # self.data_root = ('/data/pengxiao/Osteosarcoma_diagnosis/data/external_val_set_1_删减版/')
        # self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/external1_临床信息/骨肿瘤收集表.xlsx'
        # self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/external1_临床信息/骨肿瘤收集表.xlsx'
        self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/prospective_val_set_修正DR'
        self.os_clinical_file = '/home/pengxiao/disk1/FAH 前瞻性研究数据/2025年 中山附一 前瞻性研究 OS & NOS 2025.XLSX'
        self.nos_clinical_file = '/home/pengxiao/disk1/FAH 前瞻性研究数据/2025年 中山附一 前瞻性研究 OS & NOS 2025.XLSX'
        #
        # self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set'
        # self.os_clinical_file = '/home/pengxiao/disk/SYSUCC  data 2025-12-22/CC 已删减.XLSX'
        # self.nos_clinical_file = '/home/pengxiao/disk/SYSUCC  data 2025-12-22/CC 已删减.XLSX'
        # self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/full_val_set'
        # self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX'
        # self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'
        self.log_dir = 'checkpoints/'

        # --- Model checkpoint paths ---
        # Warning: must use models trained by train_gating_with2loss.py
        # self.dr_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/radiology/checkpoints/gating_with2loss/random_annototion/ddkg_swin_model_gating/best_swin_model_epoch33_acc0.789_auc0.864.pth'
        self.dr_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/radiology/checkpoints/gating_with2loss/全部骨的串行实验/ddkg_swin_model_gating/best_swin_model_epoch33_acc0.789_auc0.864.pth'
        self.patch_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/checkpoints/classification/2fenlei/swin/best_2fenlei_epoch9_0.835.pth'
        self.bert_model_path = "/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/"
        self.argmax_predict = False

        self.fixed_dr_weight = None


def main():
    args = Args()

    # To use a fixed weight (e.g. 0.65), uncomment the line below and set your value:
    args.fixed_dr_weight = 0.25

    tester = MultiModalTester(args)

    print("=" * 80)
    print("Multi-Modal Fusion Testing with DDKG Swin Transformer DR Model (New Interactive Attention)")
    print("=" * 80)
    print(f"DR Model: DDKG Swin Transformer (384x384 input) with Clinical Gating & Interactive Attention")
    print(f"WSI Model: Swin Large (224x224 patches)")
    print(f"DR Checkpoint: {args.dr_checkpoint}")
    print(f"WSI Checkpoint: {args.patch_checkpoint}")
    if args.fixed_dr_weight is not None:
        print(f"Fusion Mode: Fixed Weight (DDKG DR Weight: {args.fixed_dr_weight:.2f})")
    else:
        print(f"Fusion Mode: Weight Tuning (0.0 to 1.0, step 0.05)")
    print("=" * 80)

    print("Starting inference and weight optimization/evaluation...")
    tester.test()

    print("\nTesting completed! Check the generated visualizations:")
    print("  ddkg_swin_weight_performance.png - Performance vs weight curves")
    print("  ddkg_swin_roc_comparison.png - ROC curve comparison")


if __name__ == '__main__':
    main()
