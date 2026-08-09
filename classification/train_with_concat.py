import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

from classification.create_dataloader_prompt import create_data_loaders


# --- 1. Configuration ---
class Options:
    def __init__(self):
        # --- Data paths (must match create_data_loaders) ---
        self.data_root = '/data/pengxiao/Osteosarcoma_diagnosis/data/classification'
        self.clinical_data_path_os = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/OS.XLSX'
        self.clinical_data_path_nos = '/data/pengxiao/Osteosarcoma_diagnosis/data/临床信息/NOS.XLSX'
        self.bert_model_path = '/data/pengxiao/Osteosarcoma_diagnosis/radiology/pretrained_weights/'

        # --- Data loader parameters ---
        self.image_size = 224
        self.batch_size = 32
        self.num_workers = 4
        self.train_sample_percentage = 1.0

        # --- Model and training parameters ---
        self.epoch = 100
        self.num_class = 2
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # --- Learning rates ---
        self.backbone_lr = 1e-5
        self.other_lr = 1e-4
        self.weight_decay = 1e-4

        # --- Loss weights ---
        self.cls_loss_weight = 1.0
        # KL loss weight removed — using direct feature concatenation instead

        # --- Checkpoint and evaluation ---
        self.checkpoint_path = './checkpoints/swin_with_fusion_ablation'
        self.eval = False


# --- 2. Model definition (clinical text processor, feature concatenation fusion) ---
class ClinicalInfoProcessor(nn.Module):
    def __init__(self, bert_model_path, output_dim=256):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        for param in self.bert.parameters(): param.requires_grad = False
        for param in self.projection.parameters(): param.requires_grad = True

    def forward(self, clinical_info_list):
        texts = [f"性别{info['性别']} 年龄{info['年龄']}岁 发病部位在{info['发病部位']}" for info in clinical_info_list]
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(
            next(self.bert.parameters()).device)
        with torch.no_grad(): outputs = self.bert(**encoded)
        return self.projection(outputs.last_hidden_state[:, 0])


class SwinWithFusion(nn.Module):
    """
    Ablation model with direct feature concatenation (same strategy as DR_train_remove_sam.py).
    """

    def __init__(self, num_classes=2, clinical_dim=256, bert_model_path=None):
        super().__init__()
        self.backbone = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True,
                                          num_classes=0)
        self.visual_dim = self.backbone.num_features
        self.clinical_processor = ClinicalInfoProcessor(bert_model_path=bert_model_path, output_dim=clinical_dim)

        # Clinical + visual fusion classifier head
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.visual_dim + clinical_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

        # Visual-only classifier head (fallback when no clinical info is available)
        self.visual_only_classifier = nn.Sequential(
            nn.Linear(self.visual_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, clinical_info=None):
        visual_features = self.backbone(images)

        if clinical_info is not None:
            clinical_embedding = self.clinical_processor(clinical_info)
            fusion_feat = torch.cat([visual_features, clinical_embedding], dim=1)
            cls_logits = self.fusion_classifier(fusion_feat)
        else:
            cls_logits = self.visual_only_classifier(visual_features)

        return {'classification': cls_logits}


# --- 3. Training and evaluation loop ---
def main():
    opt = Options()
    os.makedirs(opt.checkpoint_path, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {opt.device}")

    train_loader, test_loader = create_data_loaders(
        opt.data_root,
        opt.clinical_data_path_os,
        opt.clinical_data_path_nos,
        opt.batch_size,
        opt.num_workers,
        opt.image_size,
        opt.train_sample_percentage
    )
    print(f"Data loaders created. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = SwinWithFusion(num_classes=opt.num_class, bert_model_path=opt.bert_model_path).to(opt.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

    param_groups = [
        {'params': model.backbone.parameters(), 'lr': opt.backbone_lr},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': opt.other_lr}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=opt.weight_decay)
    cls_criterion = nn.CrossEntropyLoss()

    best_auc = 0.0

    print("--- Starting training (direct feature concatenation ablation) ---")
    for epoch in range(opt.epoch):
        model.train()
        train_correct, train_total = 0, 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue

            images = batch['images'].to(opt.device)
            labels = batch['labels'].to(opt.device)
            clinical_info = batch['clinical_info']

            optimizer.zero_grad()
            outputs = model(images, clinical_info)

            # Classification loss only; KL distillation loss removed
            cls_loss = cls_criterion(outputs['classification'], labels)
            total_loss = cls_loss

            total_loss.backward()
            optimizer.step()

            _, predicted_epoch = torch.max(outputs['classification'], 1)
            train_total += labels.size(0)
            train_correct += (predicted_epoch == labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                _, predicted_batch = torch.max(outputs['classification'].data, 1)
                correct_in_batch = (predicted_batch == labels).sum().item()
                total_in_batch = labels.size(0)
                batch_acc = correct_in_batch / total_in_batch if total_in_batch > 0 else 0.0

                print(
                    f'Epoch [{epoch + 1}/{opt.epoch}], Batch [{batch_idx + 1}/{len(train_loader)}], '
                    f'Loss: {total_loss.item():.4f}, Batch Acc: {batch_acc:.4f}'
                )

        train_acc = train_correct / train_total if train_total > 0 else 0
        print(f'--- Epoch {epoch + 1} training complete ---')
        print(f'Overall Training Accuracy: {train_acc:.4f}')

        # --- Evaluate after each epoch ---
        model.eval()
        test_correct, test_total = 0, 0
        y_true, y_pred, y_scores = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None: continue
                images = batch['images'].to(opt.device)
                labels = batch['labels'].to(opt.device)
                clinical_info = batch['clinical_info']

                # Must pass clinical_info during inference; otherwise the visual_only branch is used
                outputs = model(images, clinical_info)

                logits = outputs['classification']
                probabilities = F.softmax(logits, dim=1)

                _, predicted = torch.max(logits, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_scores.extend(probabilities[:, 1].cpu().numpy())

        test_acc = test_correct / test_total if test_total > 0 else 0
        if len(np.unique(y_true)) > 1:
            test_auc = roc_auc_score(y_true, y_scores)
        else:
            test_auc = 0.0

        print(f'\n--- Epoch {epoch + 1} evaluation results ---')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Test AUC: {test_auc:.4f}')
        print(classification_report(y_true, y_pred, target_names=["NOS", "OS"], zero_division=0))

        if test_auc > best_auc:
            best_auc = test_auc
            save_path = os.path.join(opt.checkpoint_path, f'best_model_auc_{best_auc:.4f}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'*** New best AUC: {best_auc:.4f}! Model saved to {save_path} ***\n')


if __name__ == '__main__':
    main()