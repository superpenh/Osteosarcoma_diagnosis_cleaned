import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import random
import timm
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms


try:
    from evaluation.fusion_dr_patch_report_dataset import OsteosarcomaDataset
except ImportError:
    print("Warning: Could not import OsteosarcomaDataset. Make sure the file exists.")


def setup_seed(seed=42):
    """Global random seed lock to ensure reproducible results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Global random seed set to {seed}. Deterministic mode enabled.")


class ClinicalInfoProcessor(nn.Module):
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
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, clinical_info):
        texts = [f"性别{info['性别']} 年龄{info['年龄']}岁 发病部位在{info['发病部位']}" for info in clinical_info]
        with torch.no_grad():
            encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
                next(self.bert.parameters()).device)
            outputs = self.bert(**encoded)
        text_features = outputs.last_hidden_state[:, 0]
        clinical_embedding = self.projection(text_features)
        return clinical_embedding


class SpatialAwareModule(nn.Module):
    def __init__(self, backbone_channels_list, hidden_dim=256, clinical_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        shallow_channels = backbone_channels_list[0]
        deep_channels = backbone_channels_list[-1]
        self.deep_conv = nn.Sequential(nn.Conv2d(deep_channels, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim),
                                       nn.ReLU(inplace=True))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(shallow_channels + hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)
        )
        self.gate_generator = nn.Sequential(nn.Linear(clinical_dim, hidden_dim), nn.Sigmoid())
        self.seg_head = nn.Sequential(nn.Conv2d(hidden_dim, 64, 3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))
        self.attend_conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim),
                                         nn.ReLU(inplace=True))

        # Note: contains Dropout — model.eval() must be called
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, image_features, clinical_embedding=None):
        f1 = image_features[0]
        f4 = image_features[-1]
        deep_processed = self.deep_conv(f4)
        upsampled_deep = F.interpolate(deep_processed, size=f1.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([f1, upsampled_deep], dim=1)
        fusion_features = self.fusion_conv(fused)
        shared_features = fusion_features
        if clinical_embedding is not None:
            gate = self.gate_generator(clinical_embedding)
            gate = gate.unsqueeze(-1).unsqueeze(-1)
            shared_features = fusion_features + (fusion_features * gate)
        seg_logits_small = self.seg_head(shared_features)
        attention_map = torch.sigmoid(seg_logits_small)
        annotated_features = shared_features * (1 + attention_map)
        refined_features = self.attend_conv(annotated_features)
        cls_pred = self.classifier(refined_features)
        seg_pred_upsampled = F.interpolate(seg_logits_small, size=(384, 384), mode='bilinear', align_corners=False)
        seg_pred_final = torch.sigmoid(seg_pred_upsampled)
        return {'segmentation': seg_pred_final, 'classification': cls_pred, 'features': refined_features}


class DDKGModel(nn.Module):
    def __init__(self, num_classes=2, image_size=384, clinical_dim=256,
                 bert_model_path="/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/",
                 swin_pretrained_path=None):
        super().__init__()
        print("Loading Swin Transformer backbone...")
        self.backbone = timm.create_model('swin_large_patch4_window12_384.ms_in22k_ft_in1k', pretrained=True,
                                          features_only=True, out_indices=[0, 1, 2, 3])
        original_patch_embed = self.backbone.patch_embed.proj
        self.backbone.patch_embed.proj = nn.Conv2d(1, original_patch_embed.out_channels,
                                                   kernel_size=original_patch_embed.kernel_size,
                                                   stride=original_patch_embed.stride,
                                                   padding=original_patch_embed.padding)
        with torch.no_grad():
            self.backbone.patch_embed.proj.weight = nn.Parameter(original_patch_embed.weight.mean(dim=1, keepdim=True))
            if original_patch_embed.bias is not None:
                self.backbone.patch_embed.proj.bias = nn.Parameter(original_patch_embed.bias.clone())
        self.clinical_processor = ClinicalInfoProcessor(bert_model=bert_model_path, output_dim=clinical_dim)
        swin_channels_list = self.backbone.feature_info.channels()
        self.sam = SpatialAwareModule(swin_channels_list, clinical_dim=clinical_dim)
        for name, param in self.backbone.named_parameters():
            if name.startswith('patch_embed') or name.startswith('layers.0') or name.startswith('layers.1'):
                param.requires_grad = False

    def forward(self, images, clinical_info=None, return_features=True):
        image_features = self.backbone(images)
        converted_features = []
        for feat in image_features:
            if len(feat.shape) == 4 and feat.shape[-1] > feat.shape[1]:
                feat = feat.permute(0, 3, 1, 2)
            converted_features.append(feat)
        clinical_embedding = None
        if clinical_info is not None:
            clinical_embedding = self.clinical_processor(clinical_info)
        sam_outputs = self.sam(converted_features, clinical_embedding)
        results = {'segmentation': sam_outputs['segmentation'], 'classification': sam_outputs['classification']}
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
            padding = torch.zeros(max_dr_images - curr_num_images, curr_dr_images.size(1), curr_dr_images.size(2),
                                  curr_dr_images.size(3), dtype=curr_dr_images.dtype)
            padded_images = torch.cat([curr_dr_images, padding], dim=0)
        else:
            padded_images = curr_dr_images
        dr_images_padded.append(padded_images)
    clinical_infos = []
    for item in batch:
        info = item['clinical_info']
        info = {'性别': str(info['性别']), '年龄': str(info['年龄']), '发病部位': str(info['发病部位'])}
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
    def __init__(self, dr_model_path, patch_model_path, bert_model_path, num_classes=2, age_threshold=30):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.age_threshold = age_threshold

        self.dr_model = DDKGModel(num_classes=num_classes, image_size=384, clinical_dim=256,
                                  bert_model_path=bert_model_path)
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
        model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', num_classes=2, pretrained=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value
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

    def compute_dr_prob(self, stacked_probs, clinical_info, age_threshold):
        if stacked_probs is None:
            return torch.tensor([[0.5, 0.5]]).to(self.device)

        max_pos_prob, _ = torch.max(stacked_probs[:, :, 1], dim=0)
        neg_prob = 1.0 - max_pos_prob
        dr_prob = torch.stack([neg_prob, max_pos_prob], dim=1)

        if stacked_probs.size(1) == 1:
            img_preds = torch.argmax(stacked_probs, dim=2).squeeze(1)
            pos_count = torch.sum(img_preds).item()
            total_valid_imgs = img_preds.size(0)
            is_mixed = (pos_count > 0) and (pos_count < total_valid_imgs)

            if is_mixed:
                try:
                    age_val = clinical_info[0]['年龄']
                    age = float(age_val)
                    if age > age_threshold:
                        dr_prob = torch.tensor([[0.999, 0.001]]).to(self.device)
                except (ValueError, KeyError, IndexError) as e:
                    pass
        return dr_prob

    def forward(self, dr_images, patches, clinical_info):
        batch_size = dr_images.size(0)
        num_dr_images = dr_images.size(1)

        dr_probs_all = []
        for i in range(num_dr_images):
            current_image = dr_images[:, i]
            if torch.sum(current_image) != 0:
                processed_image = self._preprocess_dr_image_for_ddkg(current_image)
                dr_output = self.dr_model(processed_image, clinical_info)
                dr_prob_single = F.softmax(dr_output['classification'], dim=1)
                dr_probs_all.append(dr_prob_single)

        stacked_probs = None
        if dr_probs_all:
            stacked_probs = torch.stack(dr_probs_all, dim=0)

        dr_prob = self.compute_dr_prob(stacked_probs, clinical_info, self.age_threshold)

        patch_probs_list = []
        num_patches = patches.size(1)
        patch_height = patches.size(3)
        patch_width = patches.size(4)
        batch_size_patches = 10
        num_patch_batches = (num_patches + batch_size_patches - 1) // batch_size_patches

        for i in range(num_patch_batches):
            start_idx = i * batch_size_patches
            end_idx = min((i + 1) * batch_size_patches, num_patches)
            current_batch_size = end_idx - start_idx
            patch_batch = patches[:, start_idx:end_idx]
            patch_batch = patch_batch.reshape(batch_size * current_batch_size, 3, patch_height, patch_width)
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
            sum_weighted_probs = torch.sum(weighted_probs, dim=1)
            sum_of_weights = torch.sum(top_weights, dim=1)
            patch_prob = sum_weighted_probs / sum_of_weights
        else:
            patch_prob = torch.ones((batch_size, 2)).to(self.device) / 2

        return dr_prob, patch_prob, stacked_probs


class MultiModalTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = OsteosarcomaDataset(args.data_root, args.os_clinical_file, args.nos_clinical_file,
                                           image_size=224)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
        self.model = MultiModalFusionModel(
            dr_model_path=args.dr_checkpoint,
            patch_model_path=args.patch_checkpoint,
            bert_model_path=args.bert_model_path,
            num_classes=2,
            age_threshold=30
        ).to(self.device)

    def optimize_age_threshold(self):
        # Explicitly call eval() to ensure Dropout and BatchNorm are frozen
        print("Ensuring model is in strict evaluation mode (disabling Dropout)...")
        self.model.eval()
        self.model.dr_model.eval()
        self.model.patch_model.eval()

        print("Step 1: Running Model Inference to collect raw probabilities...")
        raw_results = []
        labels = []
        patient_names = []
        class_names = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Inference Phase"):
                if batch is None: continue

                dr_images = batch['dr_images'].to(self.device)
                patches = batch['patches'].to(self.device)
                clinical_info = batch['clinical_info']
                label = batch['label'].item()

                _, patch_prob, stacked_probs = self.model(dr_images, patches, clinical_info)

                raw_results.append({
                    'stacked_probs': stacked_probs.cpu() if stacked_probs is not None else None,
                    'patch_prob': patch_prob.cpu(),
                    'clinical_info': clinical_info
                })
                labels.append(label)
                patient_names.append(batch['patient_name'][0])
                class_names.append(batch['class_name'][0])

        print(f"\nStep 2: Optimizing Age Threshold (10 to 90, step 5)...")
        print('-' * 120)
        print('{:<12} {:<12} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
            'Age Thresh', 'Fusion W', 'AUC', 'Accuracy', 'Sens', 'Spec', 'F1'))
        print('-' * 120)

        best_global_auc = 0
        best_global_config = None

        TARGET_AGE = 30
        TARGET_WEIGHT = 0.25

        for age_thresh in range(10, 95, 5):
            current_dr_probs = []
            current_patch_probs = []

            for item in raw_results:
                stacked_probs_gpu = item['stacked_probs'].to(self.device) if item['stacked_probs'] is not None else None
                patch_prob_gpu = item['patch_prob'].to(self.device)

                dr_prob_new = self.model.compute_dr_prob(stacked_probs_gpu, item['clinical_info'], age_thresh)

                current_dr_probs.append(dr_prob_new[0][1].item())
                current_patch_probs.append(patch_prob_gpu[0][1].item())

            metrics = self._evaluate_fixed_fusion_weight(labels, current_dr_probs, current_patch_probs,
                                                         dr_weight=TARGET_WEIGHT)

            print('{:<12} {:<12.2f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}'.format(
                f"Age {age_thresh}", TARGET_WEIGHT, metrics['AUC'], metrics['Accuracy'],
                metrics['Sensitivity'], metrics['Specificity'], metrics['F1']
            ))

            if metrics['AUC'] > best_global_auc:
                best_global_auc = metrics['AUC']
                best_global_config = {
                    'age_threshold': age_thresh,
                    'fusion_weight': TARGET_WEIGHT,
                    'metrics': metrics
                }

        print('-' * 120)
        print(f"Optimization Complete!")
        if best_global_config:
            print(f"Best Age Threshold: {best_global_config['age_threshold']}")
            print(f"Fixed Fusion Weight (DR): {best_global_config['fusion_weight']:.2f}")
            print(f"Best AUC: {best_global_config['metrics']['AUC']:.4f}")
        print('-' * 120)

    def _evaluate_fixed_fusion_weight(self, labels, dr_probs, patch_probs, dr_weight=0.25):
        patch_weight = 1.0 - dr_weight
        fusion_probs = np.array(dr_probs) * dr_weight + np.array(patch_probs) * patch_weight
        fusion_preds = (fusion_probs > 0.5).astype(int)
        metrics = self._calculate_metrics(np.array(labels), fusion_preds, fusion_probs)
        return metrics

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        try:
            if len(np.unique(y_true)) < 2:
                auc_score = 0.5
            else:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = auc(fpr, tpr)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
            return {'Accuracy': accuracy, 'AUC': auc_score, 'Sensitivity': sensitivity, 'Specificity': specificity,
                    'Precision': precision, 'F1': f1}
        except Exception:
            return {'Accuracy': 0, 'AUC': 0, 'Sensitivity': 0, 'Specificity': 0, 'Precision': 0, 'F1': 0}


class Args:
    def __init__(self):
        self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/full_val_set'
        self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX'
        self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'
        self.dr_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/radiology/checkpoints/gating_with2loss/全部骨的串行实验/ddkg_swin_model_gating/best_swin_model_epoch33_acc0.789_auc0.864.pth'
        self.patch_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/checkpoints/classification/2fenlei/swin/best_2fenlei_epoch9_0.835.pth'
        self.bert_model_path = "/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/"
        self.fixed_dr_weight = None


def main():
    setup_seed(42)

    args = Args()
    tester = MultiModalTester(args)
    print("=" * 80)
    print("Optimization Mode: Finding Best Age Threshold (10-90) [Deterministic Mode]")
    print("=" * 80)
    tester.optimize_age_threshold()


if __name__ == '__main__':
    main()
