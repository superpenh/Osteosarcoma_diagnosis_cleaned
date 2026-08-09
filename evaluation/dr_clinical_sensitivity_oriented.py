"""
DR+Clinical Only Sensitivity-Oriented Threshold Analysis
=========================================================
Purpose: Simulate an initial-visit scenario using only first-visit information (clinical + DR).

Pipeline:
1. On the internal validation set (prospective study data), obtain probabilities from the DR+Clinical model.
2. Set multiple sensitivity-oriented thresholds (95%, 90%, 85%).
3. For each threshold, find the corresponding probability threshold.
4. Apply these probability thresholds to the CC dataset and evaluate OS high-risk recall.

Note: Only uses the DDKG DR model; WSI/SAM branches are not used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
from torch.utils.data import DataLoader

from evaluation.fusion_dr_patch_report_dataset import OsteosarcomaDataset
from radiology.train_gating_with2loss import DDKGModel


def collate_fn(batch):
    """Collate function for batching variable-length DR images"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

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


class DRClinicalOnlyModel(nn.Module):
    """DDKG model using DR + Clinical only."""

    def __init__(self, dr_model_path, bert_model_path, num_classes=2):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dr_model = DDKGModel(
            num_classes=num_classes,
            image_size=384,
            clinical_dim=256,
            bert_model_path=bert_model_path
        )
        self._load_dr_checkpoint(dr_model_path)

    def _load_dr_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            self.dr_model.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded DDKG DR model checkpoint")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
        self.dr_model.to(self.device).eval()

    def _preprocess_dr_image(self, dr_image):
        """
        Preprocess DR image: ensure [1, 384, 384] input for DDKG.
        dr_image shape: [C, H, W] or [1, H, W]
        """
        if dr_image.dim() == 3 and dr_image.size(0) == 1:
            dr_image = dr_image.unsqueeze(1)

        if dr_image.size(0) == 3:
            dr_image = dr_image.mean(dim=0, keepdim=True).unsqueeze(0)
        elif dr_image.size(0) != 1:
            dr_image = dr_image[:1].unsqueeze(0)

        return F.interpolate(dr_image, size=(384, 384), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def forward(self, dr_images, clinical_info):
        """
        Args:
            dr_images: [B, N, 1, H, W] - N DR images
            clinical_info: list of dicts with keys '性别', '年龄', '发病部位'
        Returns:
            dr_prob: [B, 2] - probability distribution
        """
        batch_size = dr_images.size(0)
        num_dr_images = dr_images.size(1)

        dr_probs_all = []
        for i in range(num_dr_images):
            current_image = dr_images[:, i]
            if torch.sum(current_image) != 0:
                processed_image = self._preprocess_dr_image(current_image)
                output = self.dr_model(processed_image, clinical_info)
                dr_probs_all.append(F.softmax(output['classification'], dim=1))

        if dr_probs_all:
            stacked_probs = torch.stack(dr_probs_all, dim=0)
            max_pos_prob, _ = torch.max(stacked_probs[:, :, 1], dim=0)
            dr_prob = torch.stack([1.0 - max_pos_prob, max_pos_prob], dim=1)

            # Age correction: when predictions are mixed and age > 35, force high OS probability
            if batch_size == 1:
                img_preds = torch.argmax(stacked_probs, dim=2).squeeze(1)
                is_mixed = (torch.sum(img_preds).item() > 0) and (torch.sum(img_preds).item() < img_preds.size(0))
                if is_mixed:
                    try:
                        age = float(clinical_info[0]['年龄'])
                        if age > 35:
                            dr_prob = torch.tensor([[0.999, 0.001]]).to(self.device)
                    except:
                        pass
        else:
            dr_prob = torch.ones((batch_size, 2)).to(self.device) / 2

        return dr_prob


class SensitivityOrientedEvaluator:
    """Sensitivity-oriented threshold analyzer."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load internal validation set (prospective study data)
        print("Loading internal validation dataset (prospective)...")
        self.internal_dataset = OsteosarcomaDataset(
            args.internal_data_root,
            args.internal_os_clinical,
            args.internal_nos_clinical,
            image_size=224
        )
        self.internal_loader = DataLoader(
            self.internal_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
        )

        # Load CC dataset
        print("Loading CC dataset...")
        self.cc_dataset = OsteosarcomaDataset(
            args.cc_data_root,
            args.cc_os_clinical,
            args.cc_nos_clinical,
            image_size=224
        )
        self.cc_loader = DataLoader(
            self.cc_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
        )

        # Load model
        print("Loading DR+Clinical model...")
        self.model = DRClinicalOnlyModel(
            args.dr_checkpoint,
            args.bert_model_path
        ).to(self.device)

    def _get_predictions(self, dataloader, desc="Inference"):
        """Get predicted probabilities and labels for a dataset."""
        labels = []
        probs = []
        names = []
        class_names = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                if batch is None:
                    continue

                dr_prob = self.model(
                    batch['dr_images'].to(self.device),
                    batch['clinical_info']
                )

                labels.append(batch['label'].item())
                probs.append(dr_prob[0][1].item())
                names.append(batch['patient_name'][0])
                class_names.append(batch['class_name'][0])

        return np.array(labels), np.array(probs), names, class_names

    def find_prob_threshold_for_sensitivity(self, y_true, y_prob, target_sensitivity):
        """
        Find the probability threshold required to achieve a target sensitivity.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target_sensitivity: Target sensitivity (e.g., 0.95)

        Returns:
            prob_threshold: Corresponding probability threshold
            actual_sensitivity: Actual sensitivity achieved at that threshold
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)

        idx = np.argmin(np.abs(tpr - target_sensitivity))
        prob_threshold = thresholds[idx]
        actual_sensitivity = tpr[idx]

        return prob_threshold, actual_sensitivity

    def evaluate_on_cc_with_thresholds(self, cc_labels, cc_probs, prob_thresholds_dict):
        """
        Evaluate OS high-risk recall on the CC dataset using different probability thresholds.

        Args:
            cc_labels: True labels for CC dataset
            cc_probs: Predicted probabilities for CC dataset
            prob_thresholds_dict: {sensitivity_name: prob_threshold}
        """
        print("\n" + "=" * 80)
        print("CC Dataset Evaluation Results (Sensitivity-Oriented Thresholds)")
        print("=" * 80)

        results = []

        for sens_name, prob_threshold in prob_thresholds_dict.items():
            cc_preds = (cc_probs >= prob_threshold).astype(int)

            tp = np.sum((cc_labels == 1) & (cc_preds == 1))
            fn = np.sum((cc_labels == 1) & (cc_preds == 0))
            os_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            tn = np.sum((cc_labels == 0) & (cc_preds == 0))
            fp = np.sum((cc_labels == 0) & (cc_preds == 1))
            accuracy = (tp + tn) / len(cc_labels) if len(cc_labels) > 0 else 0

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

            results.append({
                'Target Sensitivity': sens_name,
                'Prob Threshold': f"{prob_threshold:.4f}",
                'OS Recall (CC)': f"{os_recall:.3f}",
                'Accuracy (CC)': f"{accuracy:.3f}",
                'Specificity (CC)': f"{specificity:.3f}",
                'PPV (CC)': f"{ppv:.3f}"
            })

            print(f"\n[{sens_name}] Prob Threshold = {prob_threshold:.4f}")
            print(f"  -> CC Dataset OS high-risk recall: {os_recall:.3f} ({tp}/{tp+fn})")
            print(f"  -> CC Dataset accuracy: {accuracy:.3f}")
            print(f"  -> CC Dataset specificity: {specificity:.3f}")
            print(f"  -> CC Dataset PPV: {ppv:.3f}")

        print("\n" + "-" * 80)
        print("Summary Table:")
        print("-" * 80)
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        print("-" * 80)

        return results

    def run(self):
        """Main execution pipeline."""
        # Step 1: Get predictions on internal validation set
        print("\n" + "=" * 60)
        print("Step 1: Internal Validation Set (Prospective)")
        print("=" * 60)
        internal_labels, internal_probs, _, _ = self._get_predictions(
            self.internal_loader, "Internal Val Inference"
        )

        # Step 2: Get predictions on CC dataset
        print("\n" + "=" * 60)
        print("Step 2: CC Dataset")
        print("=" * 60)
        cc_labels, cc_probs, _, _ = self._get_predictions(
            self.cc_loader, "CC Inference"
        )

        # Step 3: Compute probability thresholds for target sensitivities on internal set
        print("\n" + "=" * 60)
        print("Step 3: Finding Probability Thresholds for Target Sensitivities")
        print("=" * 60)

        target_sensitivities = [0.95, 0.90, 0.85, 0.80]
        prob_thresholds_dict = {}

        for target_sens in target_sensitivities:
            prob_thresh, actual_sens = self.find_prob_threshold_for_sensitivity(
                internal_labels, internal_probs, target_sens
            )
            prob_thresholds_dict[f"{int(target_sens*100)}% Sensitivity"] = prob_thresh
            print(f"Target {target_sens*100:.0f}% Sensitivity -> Prob Threshold: {prob_thresh:.4f}, Actual: {actual_sens:.4f}")

        # Baseline: sensitivity at 0.5 threshold
        print("\n--- Baseline: 0.5 Threshold on Internal Validation Set ---")
        internal_preds_05 = (internal_probs >= 0.5).astype(int)
        tp = np.sum((internal_labels == 1) & (internal_preds_05 == 1))
        fn = np.sum((internal_labels == 1) & (internal_preds_05 == 0))
        sens_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Threshold=0.5 -> Sensitivity on Internal Val: {sens_05:.3f}")

        # Internal validation set statistics
        print("\n" + "=" * 60)
        print("Internal Validation Set Statistics")
        print("=" * 60)
        fpr, tpr, thresholds = roc_curve(internal_labels, internal_probs)
        internal_auc = auc(fpr, tpr)
        print(f"Total samples: {len(internal_labels)}")
        print(f"OS (positive) samples: {np.sum(internal_labels == 1)}")
        print(f"NOS (negative) samples: {np.sum(internal_labels == 0)}")
        print(f"AUC: {internal_auc:.4f}")

        # Step 4: Apply found thresholds to CC dataset
        print("\n" + "=" * 60)
        print("Step 4: Apply Thresholds to CC Dataset")
        print("=" * 60)

        self.evaluate_on_cc_with_thresholds(cc_labels, cc_probs, prob_thresholds_dict)

        # Baseline: 0.5 threshold on CC
        print("\n" + "=" * 60)
        print("Baseline: 0.5 Threshold on CC Dataset")
        print("=" * 60)
        cc_preds_05 = (cc_probs >= 0.5).astype(int)
        tp = np.sum((cc_labels == 1) & (cc_preds_05 == 1))
        fn = np.sum((cc_labels == 1) & (cc_preds_05 == 0))
        tn = np.sum((cc_labels == 0) & (cc_preds_05 == 0))
        fp = np.sum((cc_labels == 0) & (cc_preds_05 == 1))
        os_recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc_05 = (tp + tn) / len(cc_labels)
        spec_05 = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv_05 = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"OS Recall: {os_recall_05:.3f}")
        print(f"Accuracy: {acc_05:.3f}")
        print(f"Specificity: {spec_05:.3f}")
        print(f"PPV: {ppv_05:.3f}")

        return prob_thresholds_dict


class Args:
    def __init__(self):
        # Internal validation set (prospective study data)
        # self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/full_val_set'
        # self.os_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX'
        # self.nos_clinical_file = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'
        self.internal_data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/full_val_set'
        self.internal_os_clinical = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX'
        self.internal_nos_clinical = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'

        # CC dataset (for final evaluation)
        self.cc_data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set'
        self.cc_os_clinical = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set/CC 已删减.XLSX'
        self.cc_nos_clinical = '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set/CC 已删减.XLSX'

        # Model paths
        self.dr_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/radiology/checkpoints/gating_with2loss/全部骨的串行实验/ddkg_swin_model_gating/best_swin_model_epoch33_acc0.789_auc0.864.pth'
        self.bert_model_path = "/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/"


if __name__ == '__main__':
    args = Args()
    evaluator = SensitivityOrientedEvaluator(args)
    prob_thresholds = evaluator.run()

    print("\n" + "=" * 60)
    print("Final Probability Thresholds Found:")
    print("=" * 60)
    for name, thresh in prob_thresholds.items():
        print(f"  {name}: {thresh:.4f}")
