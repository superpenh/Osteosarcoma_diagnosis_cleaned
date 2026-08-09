import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from unet import UNet
from utils.dice_score import dice_loss
from data_gen import data_loader
from opts import *
import torch.nn.functional as F

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_generator, test_generator, train_samples, test_samples = data_loader(
    opt.data_path, opt.batch_size, imSize=opt.imSize, mean=dataset_mean, std=dataset_std)
iter_epoch = int(train_samples / opt.batch_size * opt.iter_epoch_ratio)
test_iter = int(test_samples / opt.batch_size)

model = UNet(n_channels=3, n_classes=opt.num_class).to(device)

criterion = nn.CrossEntropyLoss()
if opt.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
elif opt.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, nesterov=True)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)


def masked_cross_entropy_loss(outputs, targets, weights, class_weights=None):
    """Weighted cross-entropy loss with spatial masking and class balancing."""
    if class_weights is None:
        class_weights = torch.tensor([5.8, 1.0]).to(device)

    loss = F.cross_entropy(outputs, targets, weight=class_weights, reduction='none')
    loss = loss * weights
    return loss.sum() / weights.sum()


def masked_dice_loss(outputs, targets, weights, class_weights=None, smooth=1e-6):
    """Dice loss with per-class weighting."""
    if class_weights is None:
        class_weights = torch.tensor([5.8, 1.0]).to(device)

    class_weights = torch.abs(class_weights)

    soft_outputs = F.softmax(outputs, dim=1)
    num_classes = soft_outputs.shape[1]
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    weights = weights.unsqueeze(1)  # [B, 1, H, W]

    dice_loss_per_class = torch.zeros(num_classes, device=outputs.device)

    for cls in range(num_classes):
        pred_cls = soft_outputs[:, cls, :, :]  # [B, H, W]
        target_cls = one_hot_targets[:, cls, :, :]  # [B, H, W]

        intersection = torch.sum(pred_cls * target_cls * weights.squeeze(1))
        pred_sum = torch.sum(pred_cls * weights.squeeze(1))
        target_sum = torch.sum(target_cls * weights.squeeze(1))

        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice_loss_per_class[cls] = 1. - dice

    weighted_dice_loss = torch.sum(dice_loss_per_class * class_weights) / torch.sum(class_weights)

    return weighted_dice_loss


def masked_accuracy(pred_map, target, weights):
    assert pred_map.shape == target.shape

    valid_mask = weights > 0
    pred_map = torch.tensor(pred_map).to(weights.device)
    target = torch.tensor(target).to(weights.device)

    correct = (pred_map == target) & valid_mask
    total_valid_pixels = valid_mask.sum().item()

    if total_valid_pixels == 0:
        return 0.0

    accuracy = correct.sum().item() / total_valid_pixels
    return accuracy


# model.load_state_dict(torch.load('../checkpoints/segmentation/segmentor/epoch_3_0.996.pth', map_location=device))

tot_iter = iter_epoch * opt.epoch
for it in range(tot_iter):
    model.train()
    running_loss = 0.0
    x_batch, y_batch, weight_batch, _ = next(train_generator)

    x_batch = torch.from_numpy(x_batch).float().permute(0, 3, 1, 2).to(device)
    y_batch = torch.from_numpy(y_batch).long().to(device)
    print("Unique values in y_batch:", torch.unique(y_batch))
    weight_batch = torch.from_numpy(weight_batch).float().to(device)

    weight_batch[weight_batch == 0] = 0.5  # Include white background regions

    optimizer.zero_grad()

    outputs = model(x_batch)
    pred_map = torch.argmax(outputs, dim=1)
    pred_map = pred_map.cpu().numpy()

    class_weights = torch.tensor([1.0, 5.8]).to(device)
    loss = masked_cross_entropy_loss(outputs, y_batch, weight_batch, class_weights)
    loss += masked_dice_loss(outputs, y_batch, weight_batch, class_weights)

    y_batch = y_batch.cpu().numpy()
    accuracy = masked_accuracy(pred_map, y_batch, weight_batch.cpu())

    if torch.isnan(loss):
        print("Warning: Encountered NaN loss, skipping this batch.")
        continue

    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if it % 50 == 0:
        # Compute per-batch Tumor Dice for monitoring
        smooth = 1e-6

        intersection = ((pred_map == 1) & (y_batch == 1)).sum()
        pred_sum = (pred_map == 1).sum()
        target_sum = (y_batch == 1).sum()

        train_tumor_dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

        epoch_float = it / iter_epoch
        print(f"Epoch [{epoch_float:.4f}], Loss: {loss.item():.4f}, "
              f"Acc: {accuracy:.4f}, Train_Dice(Tumor): {train_tumor_dice:.4f}")

    if it % iter_epoch == 0 and it != 0 or opt.eval:
        scheduler.step()
        print('Learning rate:', scheduler.get_last_lr()[0])
        model.eval()

        # Accumulate metrics across the full validation set
        total_val_loss = 0.0
        total_intersection = 0.0
        total_union = 0.0
        num_batches = 0

        with torch.no_grad():
            for ti in range(test_iter):
                x_batch, y_batch, weight_batch, _ = next(test_generator)
                x_batch = torch.from_numpy(x_batch).float().permute(0, 3, 1, 2).to(device)
                y_batch = torch.from_numpy(y_batch).long().to(device)
                weight_batch = torch.from_numpy(weight_batch).float().to(device)

                outputs = model(x_batch)

                loss = masked_cross_entropy_loss(outputs, y_batch, weight_batch)
                loss += masked_dice_loss(outputs, y_batch, weight_batch)
                total_val_loss += loss.item()

                # Compute global Dice score (for tumor class, class 1)
                pred_map = torch.argmax(outputs, dim=1)

                valid_mask = weight_batch > 0
                pred_tumor = (pred_map == 1) & valid_mask
                true_tumor = (y_batch == 1) & valid_mask

                inter = (pred_tumor & true_tumor).sum().item()
                union = pred_tumor.sum().item() + true_tumor.sum().item()

                total_intersection += inter
                total_union += union
                num_batches += 1

                if ti % (test_iter / 10) == 0:
                    print(f"TEST [iter {ti}/{test_iter}]: loss={loss.item():.4f}")

        avg_val_loss = total_val_loss / num_batches
        avg_tumor_dice = (2.0 * total_intersection + 1e-6) / (total_union + 1e-6)

        print(f"Validation Finished. Avg Loss: {avg_val_loss:.4f}, Tumor Dice: {avg_tumor_dice:.4f}")

        epoch = it // iter_epoch
        checkpoint_path = os.path.join(opt.checkpoint_path, f'epoch_{epoch + 1}_dice_{avg_tumor_dice:.3f}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"=> Save checkpoint at {checkpoint_path}")
