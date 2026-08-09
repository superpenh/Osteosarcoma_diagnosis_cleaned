import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import torch.nn.functional as F

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

from opts import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from data_gen import data_loader_torch

img_shape = (opt.imSize, opt.imSize)
train_loader, test_loader, train_samples, test_samples = data_loader_torch(
    path=opt.data_path,
    batch_size=opt.batch_size,
    imsize=opt.imSize
)

iter_epoch = int(train_samples / opt.batch_size)
test_iter = int(test_samples / opt.batch_size)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


# Swin Transformer as backbone
model = timm.create_model('swin_large_patch4_window7_224.ms_in22k_ft_in1k', num_classes=2, pretrained=True)

count_parameters(model)

# Separate learning rates for feature extractor and classification head
feature_params = []
classifier_params = []

for name, param in model.named_parameters():
    if "head" in name:  # Swin's classification head
        classifier_params.append(param)
    else:
        feature_params.append(param)

param_groups = [
    {'params': feature_params, 'lr': 1e-5},     # lower LR for pretrained backbone
    {'params': classifier_params, 'lr': 1e-4}    # higher LR for classification head
]

optimizer = optim.AdamW(param_groups, weight_decay=1e-5)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

os.makedirs(os.path.join(opt.checkpoint_path, "swin"), exist_ok=True)

tot_iter = int(train_samples / opt.batch_size * opt.iter_epoch_ratio) * opt.epoch
test_iter = len(test_loader)
global_step = 0

best_acc = 0.0
best_epoch = 0


def preprocess_input(x):
    x = x * 2.0 - 1.0  # scale from [0,1] to [-1,1]
    return x


for epoch in range(opt.epoch):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        start_time = time.time()

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if x_batch.shape[-1] != 224:
            x_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
        train_loss += loss.item()

        it = epoch * iter_epoch + batch_idx

        if batch_idx % (len(train_loader) // 10) == 0:
            accuracy = train_correct / train_total
            end_time = time.time() - start_time
            print(f'TRAIN [iter {batch_idx}/{len(train_loader)} epoch {epoch + 1}/{opt.epoch} time {end_time:.3f}]: '
                  f'lr={optimizer.param_groups[0]["lr"]:.6f} loss={loss.item():.4f}, acc={accuracy:.3f}')
            sys.stdout.flush()

        global_step += 1

        if (it % iter_epoch == 0 and it != 0) or opt.eval:
            model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            y_true = []
            y_pred = []

            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    if x_test.shape[-1] != 224:
                        x_test = F.interpolate(x_test, size=(224, 224), mode='bilinear', align_corners=False)
                    outputs = model(x_test)
                    loss = criterion(outputs, y_test)

                    _, predicted = torch.max(outputs, 1)
                    test_total += y_test.size(0)
                    test_correct += (predicted == y_test).sum().item()
                    test_loss += loss.item()

                    y_true.extend(y_test.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

                overall_acc = test_correct / test_total
                avg_loss = test_loss / len(test_loader)

                cm = confusion_matrix(y_true, y_pred)

                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=range(opt.num_class),
                            yticklabels=range(opt.num_class))
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
                plt.show()

                print(
                    f'TEST [Epoch {epoch + 1}/{opt.epoch}]: loss={avg_loss:.4f}, acc={(overall_acc * 100):.3f}%, total={test_total}')
                sys.stdout.flush()

                # Save best model based on accuracy
                if not opt.eval and overall_acc > best_acc:
                    best_acc = overall_acc
                    best_epoch = epoch + 1

                    for filename in os.listdir(os.path.join(opt.checkpoint_path, "swin")):
                        if filename.startswith("best_"):
                            os.remove(os.path.join(opt.checkpoint_path, "swin", filename))

                    best_model_path = os.path.join(opt.checkpoint_path, "swin",
                                                   f'best_2fenlei_epoch{best_epoch}_{best_acc:.3f}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f'=> NEW BEST! Saved checkpoint at {best_model_path}')

                # Save per-epoch checkpoint
                if not opt.eval:
                    regular_model_path = os.path.join(opt.checkpoint_path, "swin",
                                                      f'2fenlei.{epoch + 1}_{overall_acc:.3f}.pth')
                    torch.save(model.state_dict(), regular_model_path)
                    print(f'=> Saved regular checkpoint at {regular_model_path}')

            model.train()

    if opt.eval:
        break

print(f"Training complete! Best accuracy: {best_acc:.3f} at epoch {best_epoch}")
