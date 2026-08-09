import os, sys
import argparse, shutil, glob, openslide
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import DataLoader
from wsi_util import SlideDataset, collate_fn, visualize_sampling_points
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import random
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# Ablation study: random sampling baseline
def seg_wsi_random_baseline(args):
    def normalize(data):
        return data / (data.max() + 1e-6)

    print("=> Starting Random Patch Sampling (Ablation Study)...")

    wsi_filelist = []
    wsi_filelist.extend(sorted(glob.glob(os.path.join(args.wsi_dir, '*.svs'))))
    wsi_filelist.extend(sorted(glob.glob(os.path.join(args.wsi_dir, '*.tif'))))

    segmented_files = next(os.walk(args.res_dir))[1]
    wsi_filelist = [a for a in wsi_filelist if os.path.splitext(os.path.basename(a))[0] not in segmented_files]

    print(f"=> Found {len(wsi_filelist)} slides to process randomly.")

    for index in range(args.start, len(wsi_filelist)):
        wsi_filepath = wsi_filelist[index]
        wsi_img_name = os.path.splitext(os.path.basename(wsi_filepath))[0]

        print(f"=> Random Sampling {index + 1}/{len(wsi_filelist)}: {wsi_img_name}")

        try:
            dataset = SlideDataset(
                slide_path=wsi_filepath,
                level=args.slide_level,
                imsize=args.imSize,
                to_real_scale=4,
                overlap=0
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=collate_fn
            )

            output_size = dataset.get_output_size()
            wsi_img_results = np.zeros(output_size + [3], dtype=np.uint8)

            # Collect all tissue patches
            all_tissue_candidates = []

            for step, (batch_imgs, locs) in enumerate(tqdm(dataloader, desc="Scanning Tissue")):
                batch_imgs_np = batch_imgs.numpy().transpose(0, 2, 3, 1)

                for i, (im, loc) in enumerate(zip(batch_imgs_np, locs)):
                    # 1. Fill the overview canvas
                    y, x = loc[0], loc[1]
                    h, w = im.shape[:2]
                    y_end = min(y + h, wsi_img_results.shape[0])
                    x_end = min(x + w, wsi_img_results.shape[1])
                    wsi_img_results[y:y_end, x:x_end] = im[:y_end - y, :x_end - x].astype(np.uint8)

                    # 2. Tissue filtering
                    from seg_wsi_2torch import is_tissue
                    if is_tissue(im):
                        # Build candidate structure matching the original code: [(y, x), dummy_seg, patch_img]
                        # In this ablation, seg_map is all zeros since we sample randomly, not by score
                        dummy_seg = np.zeros((h, w), dtype=np.float16)
                        all_tissue_candidates.append([(y, x), dummy_seg, im.copy()])

            print(f"=> Found {len(all_tissue_candidates)} tissue patches.")

            # 3. Randomly sample 50 patches
            tot_samples = 50
            if len(all_tissue_candidates) > tot_samples:
                sample_patches = random.sample(all_tissue_candidates, tot_samples)
            else:
                sample_patches = all_tissue_candidates
                print("Warning: Tissue patches less than 50, taking all.")

            # 4. Save results
            cur_dir = os.path.join(args.res_dir, wsi_img_name)
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)

            if len(sample_patches) != 0:
                for idx in range(len(sample_patches)):
                    loc, _, patch_img = sample_patches[idx]
                    file_name_surfix = f'_{idx:03}_{loc[0]:06}_{loc[1]:06}'
                    cur_patch_path = os.path.join(cur_dir, wsi_img_name + file_name_surfix)

                    imageio.imwrite(cur_patch_path + '.png', patch_img.astype(np.uint8))


        except Exception as e:
            print(f"Error processing {wsi_img_name}: {e}")
            continue


if __name__ == "__main__":
    from opts import *

    os.makedirs(opt.res_dir, exist_ok=True)

    seg_wsi_random_baseline(opt)