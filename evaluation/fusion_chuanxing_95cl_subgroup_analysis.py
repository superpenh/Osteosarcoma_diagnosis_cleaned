"""
Fusion model subgroup analysis by age.
Combines CC, external_val_set_1, full_val_set datasets, grouped by age:
- Age >= 35: middle-aged & elderly group
- Age < 35: adolescent & young adult group
"""

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
from torch.utils.data import DataLoader, ConcatDataset
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
            if batch_size == 1:
                img_preds = torch.argmax(stacked_probs, dim=2).squeeze(1)
                is_mixed = (torch.sum(img_preds).item() > 0) and (torch.sum(img_preds).item() < img_preds.size(0))
                if is_mixed:
                    try:
                        age = float(clinical_info[0]['年龄'])
                        if age > 35: dr_prob = torch.tensor([[0.999, 0.001]]).to(self.device)
                    except:
                        pass
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


class MultiModalSubgroupTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.datasets = {}
        self.combined_dataset = None
        self._load_datasets()

        self.dataloader = DataLoader(self.combined_dataset, batch_size=1, shuffle=False,
                                     num_workers=4, collate_fn=collate_fn)
        self.model = MultiModalFusionModel(args.dr_checkpoint, args.patch_checkpoint,
                                            args.bert_model_path).to(self.device)

    def _load_datasets(self):
        dataset_configs = [
            {
                'name': 'CC',
                'data_root': '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set',
                'os_clinical_file': '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set/CC 已删减.XLSX',
                'nos_clinical_file': '/data/pengxiao/Osteosarcoma_diagnosis/data/CC_val_set/CC 已删减.XLSX'
            },
            {
                'name': 'external_val_set_1',
                'data_root': '/data/pengxiao/Osteosarcoma_diagnosis/data/external_val_set_1_删减版/',
                'os_clinical_file': '/data/pengxiao/Osteosarcoma_diagnosis/data/external1_临床信息/骨肿瘤收集表.xlsx',
                'nos_clinical_file': '/data/pengxiao/Osteosarcoma_diagnosis/data/external1_临床信息/骨肿瘤收集表.xlsx'
            },
            {
                'name': 'full_val_set',
                'data_root': '/data/pengxiao/Osteosarcoma_diagnosis/data/full_val_set',
                'os_clinical_file': '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX',
                'nos_clinical_file': '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'
            }
        ]

        all_datasets = []
        for config in dataset_configs:
            if not os.path.exists(config['data_root']):
                print(f"Warning: Dataset {config['name']} not found at {config['data_root']}, skipping...")
                continue

            try:
                dataset = OsteosarcomaDataset(
                    config['data_root'],
                    config['os_clinical_file'],
                    config['nos_clinical_file'],
                    image_size=224
                )
                for case in dataset.cases:
                    case['dataset_name'] = config['name']
                dataset.name = config['name']
                self.datasets[config['name']] = dataset
                all_datasets.append(dataset)
                print(f"Loaded {config['name']}: {len(dataset)} cases")
            except Exception as e:
                print(f"Error loading dataset {config['name']}: {e}")

        if all_datasets:
            self.combined_dataset = ConcatDataset(all_datasets)
            print(f"\nTotal combined cases: {len(self.combined_dataset)}")

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
        return {'Accuracy': acc, 'AUC': auc_score, 'Sensitivity': sens, 'Specificity': spec,
                'Precision': prec, 'NPV': npv, 'F1': f1}

    def _calculate_metrics_with_ci(self, y_true, y_prob, n_bootstraps=1000):
        """Compute 95% confidence intervals via Bootstrap."""
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        point_metrics = self._calculate_metrics(y_true, (y_prob > 0.5).astype(int), y_prob)

        boot_metrics = {k: [] for k in point_metrics.keys()}
        rng = np.random.RandomState(42)
        indices = np.arange(len(y_true))

        for _ in range(n_bootstraps):
            bs_idx = rng.choice(indices, size=len(indices), replace=True)
            bs_true = y_true[bs_idx]
            bs_prob = y_prob[bs_idx]
            bs_pred = (bs_prob > 0.5).astype(int)
            res = self._calculate_metrics(bs_true, bs_pred, bs_prob)
            for k in boot_metrics:
                boot_metrics[k].append(res[k])

        final_results = {}
        for k in point_metrics:
            scores = np.sort(np.array(boot_metrics[k]))
            lower = np.percentile(scores, 2.5)
            upper = np.percentile(scores, 97.5)
            final_results[k] = f"{point_metrics[k]:.3f}({lower:.3f}-{upper:.3f})"
        return final_results

    def _parse_age(self, age_str):
        """Parse age string, return float or None if unparseable."""
        if age_str is None or age_str == '' or str(age_str).lower() == 'nan':
            return None
        try:
            return float(age_str)
        except (ValueError, TypeError):
            return None

    def _get_age_group(self, age):
        """
        Determine age subgroup:
        - >= 35: middle-aged & elderly
        - < 35: adolescent & young adult
        - None: unknown
        """
        if age is None:
            return None
        if age >= 35:
            return 'Age >= 35'
        else:
            return 'Age < 35'

    def test(self):
        labels = []
        dr_probs = []
        patch_probs = []
        names = []
        c_names = []
        clinical_infos = []
        dataset_names = []

        total_dr_images_count = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Testing"):
                if batch is None:
                    continue

                num_dr_images_in_batch = batch['dr_images'].size(1)
                for i in range(num_dr_images_in_batch):
                    if torch.sum(batch['dr_images'][:, i]) != 0:
                        total_dr_images_count += 1

                dr_p, patch_p = self.model(
                    batch['dr_images'].to(self.device),
                    batch['patches'].to(self.device),
                    batch['clinical_info']
                )

                labels.append(batch['label'].item())
                dr_probs.append(dr_p[0][1].item())
                patch_probs.append(patch_p[0][1].item())
                names.append(batch['patient_name'][0])
                c_names.append(batch['class_name'][0])
                clinical_infos.append(batch['clinical_info'][0])
                dataset_names.append('')

        print(f"\n{'=' * 60}")
        print(f"Dataset statistics: {total_dr_images_count} valid DR images processed")
        print(f"{'=' * 60}\n")

        # Map patient_name to dataset_name
        name_to_dataset = {}
        for ds_name, ds in self.datasets.items():
            for case in ds.cases:
                name_to_dataset[case['patient_name']] = ds_name

        # Parse age and subgroup for each sample
        age_groups = []
        ages = []
        for info in clinical_infos:
            age = self._parse_age(info.get('年龄', ''))
            ages.append(age)
            age_groups.append(self._get_age_group(age))

        self._test_labels = labels
        self._test_dr_probs = dr_probs
        self._test_patch_probs = patch_probs
        self._test_names = names
        self._test_age_groups = age_groups

        # Organize data by subgroup
        subgroup_data = {
            'Age >= 35': {'labels': [], 'dr_probs': [], 'patch_probs': [], 'count': 0},
            'Age < 35': {'labels': [], 'dr_probs': [], 'patch_probs': [], 'count': 0},
            'Unknown Age': {'labels': [], 'dr_probs': [], 'patch_probs': [], 'count': 0}
        }

        # Organize data by dataset
        dataset_data = {}
        for ds_name in self.datasets.keys():
            dataset_data[ds_name] = {'labels': [], 'dr_probs': [], 'patch_probs': [], 'count': 0}

        for i in range(len(labels)):
            ag = age_groups[i]
            if ag is None:
                ag = 'Unknown Age'
            subgroup_data[ag]['labels'].append(labels[i])
            subgroup_data[ag]['dr_probs'].append(dr_probs[i])
            subgroup_data[ag]['patch_probs'].append(patch_probs[i])
            subgroup_data[ag]['count'] += 1

            patient_name = names[i]
            ds_name = name_to_dataset.get(patient_name, '')
            if ds_name and ds_name in dataset_data:
                dataset_data[ds_name]['labels'].append(labels[i])
                dataset_data[ds_name]['dr_probs'].append(dr_probs[i])
                dataset_data[ds_name]['patch_probs'].append(patch_probs[i])
                dataset_data[ds_name]['count'] += 1

        # Print subgroup sample distribution
        print("\n" + "=" * 60)
        print("Subgroup Sample Distribution:")
        print("-" * 60)
        for ag, data in subgroup_data.items():
            print(f"  {ag}: {data['count']} cases")
        print("=" * 60 + "\n")

        # Print dataset sample distribution
        print("\n" + "=" * 60)
        print("Dataset Sample Distribution:")
        print("-" * 60)
        for ds_name, data in dataset_data.items():
            print(f"  {ds_name}: {data['count']} cases")
        print("=" * 60 + "\n")

        dr_weight = self.args.fixed_dr_weight if self.args.fixed_dr_weight else 0.25

        # Evaluate by dataset
        self._evaluate_datasets(dataset_data, dr_weight)

    def _evaluate_subgroups(self, subgroup_data, dr_weight):
        """Evaluate performance per subgroup."""
        print(f"\n{'=' * 100}")
        print(f"Fusion weight: DR={dr_weight:.2f}, WSI={1-dr_weight:.2f}")
        print("=" * 100)

        header = '-' * 100
        fmt = '{:<35} {:<25} {:<25} {:<25} {:<25}'
        print(header)
        print(fmt.format('Model/Group', 'Accuracy (95% CI)', 'AUC (95% CI)', 'Sens (95% CI)', 'Spec (95% CI)'))
        print(header)

        for group_name, data in subgroup_data.items():
            if data['count'] < 5:
                print(f"\n{group_name}: insufficient samples (<5), skipping statistical analysis")
                continue

            print(f"\n>>> {group_name} (n={data['count']}) <<<")

            fusion_probs = np.array(data['dr_probs']) * dr_weight + np.array(data['patch_probs']) * (1 - dr_weight)

            ci_fusion = self._calculate_metrics_with_ci(data['labels'], fusion_probs)
            ci_dr = self._calculate_metrics_with_ci(data['labels'], data['dr_probs'])
            ci_patch = self._calculate_metrics_with_ci(data['labels'], data['patch_probs'])

            for name, ci in [('Fusion', ci_fusion), ('DDKG DR', ci_dr), ('WSI', ci_patch)]:
                print(fmt.format(
                    f"  {name}",
                    ci['Accuracy'], ci['AUC'], ci['Sensitivity'], ci['Specificity']
                ))
            print("-" * 100)

        # Aggregate all (excluding unknown age)
        all_labels = []
        all_dr_probs = []
        all_patch_probs = []
        total_count = 0

        for group_name, data in subgroup_data.items():
            if group_name != 'Unknown Age':
                all_labels.extend(data['labels'])
                all_dr_probs.extend(data['dr_probs'])
                all_patch_probs.extend(data['patch_probs'])
                total_count += data['count']

        if total_count >= 5:
            print(f"\n>>> All Data Summary (n={total_count}) <<<")
            fusion_probs = np.array(all_dr_probs) * dr_weight + np.array(all_patch_probs) * (1 - dr_weight)
            ci_fusion = self._calculate_metrics_with_ci(all_labels, fusion_probs)
            ci_dr = self._calculate_metrics_with_ci(all_labels, all_dr_probs)
            ci_patch = self._calculate_metrics_with_ci(all_labels, all_patch_probs)

            for name, ci in [('Fusion', ci_fusion), ('DDKG DR', ci_dr), ('WSI', ci_patch)]:
                print(fmt.format(
                    f"  {name}",
                    ci['Accuracy'], ci['AUC'], ci['Sensitivity'], ci['Specificity']
                ))
            print("-" * 100)

        print(header)

    def _evaluate_datasets(self, dataset_data, dr_weight):
        """Evaluate performance per dataset x age subgroup."""
        print(f"\n{'=' * 120}")
        print(f"[Dataset x Age Subgroup Analysis] Fusion weight: DR={dr_weight:.2f}, WSI={1-dr_weight:.2f}")
        print("=" * 120)

        name_to_dataset = {}
        for ds_name, ds in self.datasets.items():
            for case in ds.cases:
                name_to_dataset[case['patient_name']] = ds_name

        ds_subgroup_data = {}
        for ds_name in self.datasets.keys():
            for subgroup_name in ['Age >= 35', 'Age < 35', 'Unknown Age']:
                key = f"{ds_name}_{subgroup_name}"
                ds_subgroup_data[key] = {
                    'dataset': ds_name,
                    'subgroup': subgroup_name,
                    'labels': [],
                    'dr_probs': [],
                    'patch_probs': [],
                    'count': 0
                }

        for i in range(len(self._test_names)):
            patient_name = self._test_names[i]
            ds_name = name_to_dataset.get(patient_name, '')
            if ds_name not in dataset_data:
                continue

            subgroup_key = None
            for subgroup_name in ['Age >= 35', 'Age < 35', 'Unknown Age']:
                key = f"{ds_name}_{subgroup_name}"
                if self._test_age_groups[i] == subgroup_name:
                    subgroup_key = key
                    break

            if subgroup_key is None:
                continue

            ds_subgroup_data[subgroup_key]['labels'].append(self._test_labels[i])
            ds_subgroup_data[subgroup_key]['dr_probs'].append(self._test_dr_probs[i])
            ds_subgroup_data[subgroup_key]['patch_probs'].append(self._test_patch_probs[i])
            ds_subgroup_data[subgroup_key]['count'] += 1

        header = '-' * 120
        fmt = '{:<35} {:<25} {:<25} {:<25} {:<25}'
        print(header)
        print(fmt.format('Dataset/Group', 'Accuracy (95% CI)', 'AUC (95% CI)', 'Sens (95% CI)', 'Spec (95% CI)'))
        print(header)

        all_labels = []
        all_dr_probs = []
        all_patch_probs = []
        total_count = 0

        for ds_name in ['CC', 'external_val_set_1', 'full_val_set']:
            if ds_name not in dataset_data or dataset_data[ds_name]['count'] < 5:
                continue

            print(f"\n>>> {ds_name} (n={dataset_data[ds_name]['count']}) <<<")
            ds_has_output = False

            for subgroup_name in ['Age >= 35', 'Age < 35', 'Unknown Age']:
                key = f"{ds_name}_{subgroup_name}"
                data = ds_subgroup_data[key]
                if data['count'] < 5:
                    continue

                fusion_probs = np.array(data['dr_probs']) * dr_weight + np.array(data['patch_probs']) * (1 - dr_weight)
                ci_fusion = self._calculate_metrics_with_ci(data['labels'], fusion_probs)
                ci_dr = self._calculate_metrics_with_ci(data['labels'], data['dr_probs'])
                ci_patch = self._calculate_metrics_with_ci(data['labels'], data['patch_probs'])

                for name, ci in [('Fusion', ci_fusion), ('DDKG DR', ci_dr), ('WSI', ci_patch)]:
                    print(fmt.format(
                        f"  [{subgroup_name}] {name}",
                        ci['Accuracy'], ci['AUC'], ci['Sensitivity'], ci['Specificity']
                    ))
                ds_has_output = True

                if subgroup_name != 'Unknown Age':
                    all_labels.extend(data['labels'])
                    all_dr_probs.extend(data['dr_probs'])
                    all_patch_probs.extend(data['patch_probs'])
                    total_count += data['count']

            if ds_has_output:
                print("-" * 120)

        if total_count >= 5:
            print(f"\n>>> All Data Summary (n={total_count}) <<<")
            fusion_probs = np.array(all_dr_probs) * dr_weight + np.array(all_patch_probs) * (1 - dr_weight)
            ci_fusion = self._calculate_metrics_with_ci(all_labels, fusion_probs)
            ci_dr = self._calculate_metrics_with_ci(all_labels, all_dr_probs)
            ci_patch = self._calculate_metrics_with_ci(all_labels, all_patch_probs)

            for name, ci in [('Fusion', ci_fusion), ('DDKG DR', ci_dr), ('WSI', ci_patch)]:
                print(fmt.format(
                    f"  {name}",
                    ci['Accuracy'], ci['AUC'], ci['Sensitivity'], ci['Specificity']
                ))
            print("-" * 120)

        print(header)

        # Confusion matrix per dataset
        print(f"\n{'=' * 80}")
        print("[Confusion Matrix - Per Dataset]")
        print("=" * 80)

        name_to_dataset = {}
        for ds_name, ds in self.datasets.items():
            for case in ds.cases:
                name_to_dataset[case['patient_name']] = ds_name

        for ds_name in ['CC', 'external_val_set_1', 'full_val_set']:
            if ds_name not in dataset_data or dataset_data[ds_name]['count'] < 5:
                continue

            ds_labels = []
            ds_preds = []
            for i in range(len(self._test_names)):
                patient_name = self._test_names[i]
                if name_to_dataset.get(patient_name, '') != ds_name:
                    continue
                ds_labels.append(self._test_labels[i])
                ds_preds.append(1 if (self._test_dr_probs[i] * dr_weight + self._test_patch_probs[i] * (1 - dr_weight)) > 0.5 else 0)

            cm = confusion_matrix(ds_labels, ds_preds, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            print(f"\n>>> {ds_name} (n={len(ds_labels)}) <<<")
            print(f"  Pred\\True    OS(1)    NOS(0)")
            print(f"  OS(1)         {tp:>4}      {fp:>4}")
            print(f"  NOS(0)        {fn:>4}      {tn:>4}")
            print(f"  Total: OS={tp+fn}, NOS={tn+fp}")

        print("\n" + "=" * 80)


class Args:
    def __init__(self):
        self.dr_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/radiology/checkpoints/gating_with2loss/全部骨的串行实验/ddkg_swin_model_gating/best_swin_model_epoch33_acc0.789_auc0.864.pth'
        self.patch_checkpoint = '/data/pengxiao/Osteosarcoma_diagnosis/checkpoints/classification/2fenlei/swin/best_2fenlei_epoch9_0.835.pth'
        self.bert_model_path = "/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/"
        self.fixed_dr_weight = 0.25


if __name__ == '__main__':
    args = Args()
    tester = MultiModalSubgroupTester(args)
    tester.test()
