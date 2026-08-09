import os, sys
import argparse, shutil, glob, openslide
import numpy as np
import imageio
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from wsi_util import patch_sampling, SlideDataset, collate_fn,visualize_sampling_points, visualize_heatmap
from unet import UNet
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from PIL import Image
from tabulate import tabulate
import time
import pandas as pd
Image.MAX_IMAGE_PIXELS = None  # 解除像素限制

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return total_params

def is_tissue(patch, saturation_threshold=10, grad_threshold=20):
    """
    Filter tissue patches — exclude blue pen marks, retain pink/purple tissue, and detect texture variation.

    Args:
        patch: RGB image patch
        saturation_threshold: Minimum saturation required
        grad_threshold: Texture variation threshold
    """
    if patch.dtype != np.uint8:
        patch = patch.astype(np.uint8)

    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    s = np.mean(hsv[:, :, 1])

    saturation_ratio = np.sum(s > saturation_threshold) / s.size

    # Background if high-saturation pixels are below 15%
    if saturation_ratio < 0.4:
        return False

    r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
    blue_mask = (b > 1.2 * r) & (b > 1.2 * g) & (b > 100) & (b > (r + g))
    blue_ratio = np.sum(blue_mask) / blue_mask.size

    # Pink/purple tissue regions typically have high R and B, low G
    pink_purple_mask = (r > 80) & (r > g) & (s > saturation_threshold)
    pink_purple_ratio = np.sum(pink_purple_mask) / pink_purple_mask.size
    all_pink_mask = (r + b) > 400
    all_pink_ratio = np.sum(all_pink_mask) / all_pink_mask.size

    # Filter out if too much blue pen or not enough pink/purple tissue
    if blue_ratio > 0.2 or pink_purple_ratio < 0.5 or all_pink_ratio > 0.8:
        return False

    # Texture analysis via Sobel gradient magnitude
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mean_gradient = np.mean(gradient_magnitude)

    # Reject textureless regions
    if mean_gradient < grad_threshold:
        return False

    return True



def seg_wsi(args):
    def normalize(data):
        return data / data.max()

    sampled_img_size = 256
    border = 16  # single-side border

    model = UNet(n_channels=3, n_classes=opt.num_class)

    checkpoint_path = '../checkpoints/segmentation/segmentor/epoch_3_dice_0.905.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    count_model_params(model)
    print("[*] Success to load model.")

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    print("=> Starting WSI patch generation...")

    wsi_filelist = []
    wsi_filelist.extend(sorted(glob.glob(os.path.join(args.wsi_dir, '*.svs'))))
    wsi_filelist.extend(sorted(glob.glob(os.path.join(args.wsi_dir, '*.tif'))))

    segmented_files = next(os.walk(args.res_dir))[1]

    print("=> Found {} whole slide images in total.".format(len(wsi_filelist)))
    print("=> {} has been processed.".format(len(segmented_files)))
    wsi_filelist = [a for a in wsi_filelist if os.path.splitext(os.path.basename(a))[0] not in segmented_files]
    print("=> {} is being processed.".format(len(wsi_filelist)))

    # end = min(args.end, len(wsi_filelist))
    end = len(wsi_filelist)

    timing_records = []

    for index in range(args.start, end):
        wsi_filepath = wsi_filelist[index]
        wsi_img_name = os.path.splitext(os.path.basename(wsi_filepath))[0]

        if os.path.isdir(os.path.join(args.res_dir, wsi_img_name)) or wsi_img_name in segmented_files:
            continue

        print("=> Start {}/{} segment {}".format(index + 1, end, wsi_img_name))

        try:
            t0 = time.time()
            dataset = SlideDataset(
                slide_path=wsi_filepath,
                level=args.slide_level,
                imsize=args.imSize,
                to_real_scale=4,#   TODO 正常数据集这里是4
                overlap=0
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,#   TODO 正常数据集这里是8
                collate_fn=collate_fn
            )
            # Initialize result arrays
            output_size = dataset.get_output_size()
            wsi_seg_results = np.zeros(output_size, dtype=np.float16)
            wsi_img_results = np.zeros(output_size + [3], dtype=np.uint8)
            wsi_mask = np.zeros(output_size, dtype=np.float16)

            print(f"=> Start {index + 1}/{end} segment {dataset.get_slide_name()}")

            candidates = []

            for step, (batch_imgs, locs) in enumerate(tqdm(dataloader)):
                batch_imgs_np = batch_imgs.numpy().transpose(0, 2, 3, 1)

                for im, loc in zip(batch_imgs_np, locs):
                    y, x = loc[0], loc[1]
                    h, w = im.shape[:2]

                    y_end = min(y + h, wsi_img_results.shape[0])
                    x_end = min(x + w, wsi_img_results.shape[1])

                    wsi_img_results[y:y_end, x:x_end] = im[:y_end - y, :x_end - x].astype(np.uint8)

                # Filter patches that pass the tissue check
                valid_indices = [i for i, img in enumerate(batch_imgs_np) if is_tissue(img)]

                if not valid_indices:
                    continue

                filtered_imgs = batch_imgs[valid_indices]
                filtered_locs = [locs[i] for i in valid_indices]

                filtered_imgs_tensor = filtered_imgs.to(device)
                with torch.no_grad():
                    batch_pred = F.softmax(model(filtered_imgs_tensor), dim=1)

                filtered_imgs_np = filtered_imgs.cpu().numpy().transpose(0, 2, 3, 1)
                batch_logits = batch_pred[:, 1, :, :].cpu().numpy()

                for seg, im, loc in zip(batch_logits, filtered_imgs_np, filtered_locs):
                    y, x = loc[0], loc[1]
                    seg_h, seg_w = seg.shape

                    y_end = min(y + seg_h, wsi_seg_results.shape[0])
                    x_end = min(x + seg_w, wsi_seg_results.shape[1])

                    wsi_mask[y:y_end, x:x_end] += 1
                    wsi_seg_results[y:y_end, x:x_end] = np.maximum(wsi_seg_results[y:y_end, x:x_end],
                                                                   seg[:y_end - y, :x_end - x].astype(np.float16))
                    wsi_img_results[y:y_end, x:x_end] = im[:y_end - y, :x_end - x].astype(np.uint8)
                    candidates.append([(y, x), seg.copy(), im.copy()])

            print(f"=> Total candidates after filtering: {len(candidates)}")

            # 保存结果
            cur_dir = os.path.join(args.res_dir, os.path.splitext(wsi_img_name)[0])
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

            # if len(candidates) > 0:
            #     try:
            #         # 引用我们在 wsi_util 中新加的函数
            #         from wsi_util import visualize_all_candidates_with_score
            #
            #         all_viz = visualize_all_candidates_with_score(wsi_img_results, candidates)
            #
            #         # 缩放图片，防止图片过大导致查看困难 (0.5倍)
            #         all_viz_resized = cv2.resize(all_viz.astype(np.uint8), (0, 0), fx=0.5, fy=0.5)
            #
            #         # 保存图片，命名为 _all_candidates_scores.jpg
            #         save_path = os.path.join(cur_dir, wsi_img_name + '_all_candidates_scores.jpg')
            #         imageio.imwrite(save_path, all_viz_resized)
            #         print(f"=> Saved all candidates with scores to {save_path}")
            #
            #     except Exception as e:
            #         print(f"Error visualizing candidates: {e}")


            sample_patches_all = patch_sampling(candidates, tot_samples=float('inf'), stride_ratio=0.1, sample_size=[256, 256], threshold=0.9)
            if len(sample_patches_all) != 0:
                # 按分数从高到低排序
                sample_patches_sorted = sorted(sample_patches_all, key=lambda x: float(x[2].mean()), reverse=True)
                # Top 50% patches for overview visualization
                top_50_percent_idx = max(1, int(len(sample_patches_sorted) * 0.5))
                sample_patches_top50percent = sample_patches_sorted[:top_50_percent_idx]
                # Save top 50 patches individually
                sample_patches_50 = sample_patches_sorted[:min(50, len(sample_patches_sorted))]
                for idx in range(len(sample_patches_50)):
                    file_name_surfix = f'_{idx:03}_{sample_patches_50[idx][0][0]:06}_{sample_patches_50[idx][0][1]:06}'
                    cur_patch_path = os.path.join(cur_dir, wsi_img_name + file_name_surfix)
                    sample_img = sample_patches_50[idx][1].astype(np.uint8)
                    sample_seg = (normalize(sample_patches_50[idx][2]) * 255.0).astype(np.uint8)
                    imageio.imwrite(cur_patch_path + '.png', sample_img)
                # # 可视化前50%满足条件的点
                # locs = [a[0] for a in sample_patches_top50percent]
                # wsi_img_point_results = visualize_sampling_points(wsi_img_results, locs)
                # # 缩小图片（0.25倍）便于查看
                # wsi_img_point_resized = cv2.resize(wsi_img_point_results.astype(np.uint8), (0, 0), fx=0.25, fy=0.25)
                # imageio.imwrite(os.path.join(cur_dir, wsi_img_name + '_samplepoint.jpg'), wsi_img_point_resized)

            elapsed = time.time() - t0
            slide_name = dataset.get_slide_name()
            timing_records.append({'patient_name': slide_name, 'time_cost (s)': elapsed})
            print(f"=> [{slide_name}] Segmentation completed in {elapsed:.2f}s")

        except Exception as e:
            print(e)
            pass

    # Output timing summary table after all slides are processed
    if timing_records:
        csv_path = os.path.join(args.wsi_dir, 'segmentation_timing.csv')
        df = pd.DataFrame(timing_records)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        headers = ["Patient Name", "Time Cost (s)"]
        table = [[r['patient_name'], f"{r['time_cost (s)']:.3f}"] for r in timing_records]
        print("\n" + "=" * 60)
        print("[Segmentation Timing Summary]")
        print(tabulate(table, headers=headers, tablefmt='grid'))
        print(f"\nTotal: {len(timing_records)} slides processed")
        print(f"CSV saved to: {csv_path}")
        print("=" * 60)


if __name__ == "__main__":
    from opts import *
    if not os.path.isdir(opt.res_dir):
        os.mkdir(opt.res_dir)
    seg_wsi(opt)
