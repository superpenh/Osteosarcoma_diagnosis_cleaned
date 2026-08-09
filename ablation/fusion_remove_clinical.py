import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
import pandas as pd

from evaluation.fusion_dr_patch_report_dataset import OsteosarcomaDataset
from radiology.train_gating_with2loss import DDKGModel


class MultiModalFusionModel(nn.Module):
    def __init__(self, dr_model_path, patch_model_path, bert_model_path, num_classes=2):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dr_model = DDKGModel(num_classes=num_classes, image_size=384, clinical_dim=256,
                                  bert_model_path=bert_model_path)
        self._load_dr_checkpoint(dr_model_path)
        self.patch_model = self._init_swin_patch_model(patch_model_path)

    def _load_dr_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint,
                                                                      dict) and 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            self.dr_model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded DDKG Swin DR model checkpoint")
        except:
            pass
        self.dr_model.to(self.device).eval()

    def _init_swin_patch_model(self, checkpoint_path):
        model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', num_classes=2, pretrained=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint,
                                                                      dict) and 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)
        except:
            pass
        return model.to(self.device).eval()

    def _preprocess_dr_image_for_ddkg(self, dr_image):
        if dr_image.size(1) == 3: dr_image = dr_image.mean(dim=1, keepdim=True)
        return F.interpolate(dr_image, size=(384, 384), mode='bilinear', align_corners=False)

    def forward(self, dr_images, patches, clinical_info):
        batch_size = dr_images.size(0)
        num_dr_images = dr_images.size(1)
        dr_probs_all = []
        for i in range(num_dr_images):
            current_image = dr_images[:, i]
            if torch.sum(current_image) != 0:
                processed_image = self._preprocess_dr_image_for_ddkg(current_image)
                dr_output = self.dr_model(processed_image, clinical_info)
                dr_probs_all.append(F.softmax(dr_output['classification'], dim=1))

        if dr_probs_all:
            stacked_probs = torch.stack(dr_probs_all, dim=0)
            max_pos_prob, _ = torch.max(stacked_probs[:, :, 1], dim=0)
            dr_prob = torch.stack([1.0 - max_pos_prob, max_pos_prob], dim=1)
            # Age correction logic removed for this ablation
        else:
            dr_prob = torch.ones((batch_size, 2)).to(self.device) / 2

        patch_probs_list = []
        num_patches = patches.size(1)
        for i in range((num_patches + 9) // 10):
            p_batch = patches[:, i * 10: min((i + 1) * 10, num_patches)].reshape(-1, 3, 224, 224)
            p_batch = F.interpolate(p_batch, size=(224, 224), mode='bilinear', align_corners=False)
            prob = F.softmax(self.patch_model(p_batch), dim=1).view(batch_size, -1, 2)
            patch_probs_list.append(prob)

        if patch_probs_list:
            all_p = torch.cat(patch_probs_list, dim=1)
            conf = torch.abs(all_p[:, :, 1] - 0.5)
            val, idx = torch.topk(conf, k=min(50, conf.size(1)), dim=1)
            top_p = torch.gather(all_p, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
            w = (val + 1e-6).unsqueeze(-1)
            patch_prob = torch.sum(top_p * w, dim=1) / torch.sum(w, dim=1)
        else:
            patch_prob = torch.ones((batch_size, 2)).to(self.device) / 2

        return dr_prob, patch_prob


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None

    max_dr = max([item['dr_images'].size(0) for item in batch])

    dr_padded = []
    for item in batch:
        curr = item['dr_images']
        num_to_pad = max_dr - curr.size(0)
        if num_to_pad > 0:
            pad = torch.zeros(num_to_pad, curr.size(1), curr.size(2), curr.size(3), dtype=curr.dtype)
            padded_images = torch.cat([curr, pad], dim=0)
        else:
            padded_images = curr
        dr_padded.append(padded_images)

    clinical_infos = []
    for item in batch:
        info = item['clinical_info']
        clinical_infos.append({
            '性别': str(info['性别']),
            '年龄': str(info['年龄']),
            '发病部位': str(info['发病部位'])
        })

    return {
        'dr_images': torch.stack(dr_padded),
        'patches': torch.stack([item['patches'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch]),
        'patient_name': [item['patient_name'] for item in batch],
        'class_name': [item['class_name'] for item in batch],
        'clinical_info': clinical_infos
    }


class MultiModalTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = OsteosarcomaDataset(args.data_root, args.os_clinical_file, args.nos_clinical_file, image_size=224)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
        self.model = MultiModalFusionModel(args.dr_checkpoint, args.patch_checkpoint, args.bert_model_path).to(self.device)

    def _calculate_metrics(self, y_true, y_pred, y_prob):
        y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
        if len(np.unique(y_true)) < 2:
            tn, fp, fn, tp = (np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1)),
                              np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1)))
            auc_score = 0.5
        else:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)
        acc = (tp + tn) / len(y_true)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        return {'Accuracy': acc, 'AUC': auc_score, 'Sensitivity': sens, 'Specificity': spec, 'Precision': prec,
                'NPV': npv, 'F1': f1}

    def _calculate_metrics_with_ci(self, y_true, y_prob, n_bootstraps=1000):
        """Compute 95% confidence intervals via Bootstrap."""
        y_true, y_prob = np.array(y_true), np.array(y_prob)
        point_metrics = self._calculate_metrics(y_true, (y_prob > 0.5).astype(int), y_prob)
        boot_metrics = {k: [] for k in point_metrics.keys()}
        rng = np.random.RandomState(42)
        indices = np.arange(len(y_true))
        for _ in range(n_bootstraps):
            bs_idx = rng.choice(indices, size=len(indices), replace=True)
            bs_true, bs_prob = y_true[bs_idx], y_prob[bs_idx]
            res = self._calculate_metrics(bs_true, (bs_prob > 0.5).astype(int), bs_prob)
            for k in boot_metrics: boot_metrics[k].append(res[k])
        final_results = {}
        for k in point_metrics:
            scores = np.sort(np.array(boot_metrics[k]))
            lower, upper = np.percentile(scores, 2.5), np.percentile(scores, 97.5)
            final_results[k] = f"{point_metrics[k]:.3f}({lower:.3f}-{upper:.3f})"
        return final_results

    def test(self):
        labels, dr_ps, patch_ps, names, c_names = [], [], [], [], []
        total_dr_images_count = 0
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Testing"):
                if batch is None: continue
                num_dr_images_in_batch = batch['dr_images'].size(1)
                for i in range(num_dr_images_in_batch):
                    if torch.sum(batch['dr_images'][:, i]) != 0:
                        total_dr_images_count += 1
                dr_p, patch_p = self.model(batch['dr_images'].to(self.device), batch['patches'].to(self.device),
                                           batch['clinical_info'])
                labels.append(batch['label'].item())
                dr_ps.append(dr_p[0][1].item())
                patch_ps.append(patch_p[0][1].item())
                names.append(batch['patient_name'][0])
                c_names.append(batch['class_name'][0])

        print(f"\n" + "=" * 60)
        print(f"Dataset statistics: {total_dr_images_count} valid DR images processed")
        print("=" * 60 + "\n")

        self.labels, self.dr_probs, self.patch_probs = labels, dr_ps, patch_ps
        if self.args.fixed_dr_weight is not None:
            self._evaluate_fixed_weight(labels, dr_ps, patch_ps, names, c_names, self.args.fixed_dr_weight)
        else:
            self._try_different_weights(labels, dr_ps, patch_ps, names, c_names)

    def _evaluate_fixed_weight(self, labels, dr_probs, patch_probs, names, c_names, dr_weight):
        print(f"\n--- Evaluating with Fixed Weight: DR={dr_weight:.2f} ---")
        fusion_probs = np.array(dr_probs) * dr_weight + np.array(patch_probs) * (1 - dr_weight)
        print("Calculating Bootstrap Confidence Intervals (1000 iterations)...")
        ci_fusion = self._calculate_metrics_with_ci(labels, fusion_probs)
        ci_dr = self._calculate_metrics_with_ci(labels, dr_probs)
        ci_patch = self._calculate_metrics_with_ci(labels, patch_probs)
        header = '-' * 240
        print(header)
        fmt = '{:<25} {:<30} {:<30} {:<30} {:<30} {:<30} {:<30} {:<30}'
        print(fmt.format('Model', 'Accuracy (95% CI)', 'AUC (95% CI)', 'Sens (95% CI)', 'Spec (95% CI)', 'Prec (95% CI)',
                         'NPV (95% CI)', 'F1 (95% CI)'))
        print(header)
        for name, ci in [('Fusion (Fixed)', ci_fusion), ('DDKG DR Only', ci_dr), ('WSI Model Only', ci_patch)]:
            print(fmt.format(name, ci['Accuracy'], ci['AUC'], ci['Sensitivity'], ci['Specificity'],
                             ci['Precision'], ci['NPV'], ci['F1']))
        print(header)

    def _try_different_weights(self, labels, dr_probs, patch_probs, names, c_names):
        best_auc, best_w = -1, 0
        for w in np.arange(0, 1.05, 0.05):
            f_p = np.array(dr_probs) * w + np.array(patch_probs) * (1 - w)
            cur_auc = self._calculate_metrics(labels, (f_p > 0.5).astype(int), f_p)['AUC']
            if cur_auc > best_auc: best_auc, best_w = cur_auc, w
        self._evaluate_fixed_weight(labels, dr_probs, patch_probs, names, c_names, best_w)


class Args:
    def __init__(self):
        # self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set'
        # self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set/CC 已删减.XLSX'
        # self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set/CC 已删减.XLSX'
        # self.data_root = ('/data/pengxiao/Osteosarcoma_diagnosis/data/external_val_set_1_删减版/')
        # self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/external1_临床信息/骨肿瘤收集表.xlsx'
        # self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/external1_临床信息/骨肿瘤收集表.xlsx'
        # self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/prospective_val_set_修正DR'
        # self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/prospective_val_set_修正DR/2025年 中山附一 前瞻性研究 OS & NOS 2025.XLSX'
        # self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/prospective_val_set_修正DR//2025年 中山附一 前瞻性研究 OS & NOS 2025.XLSX'
        #
        self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/full_val_set'
        self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX'
        self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'
        self.dr_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/ablation/checkpoints/gating_with2loss_ablation_remove_clinical/ddkg_swin_model_gating/best_swin_model_epoch8_acc0.764_auc0.842.pth'
        self.patch_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/checkpoints/classification/2fenlei/swin/best_2fenlei_epoch9_0.835.pth'
        self.bert_model_path = "/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/"
        self.fixed_dr_weight = 0.25


if __name__ == '__main__':
    args = Args()
    tester = MultiModalTester(args)
    tester.test()
