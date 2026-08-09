import os, sys
import numpy as np
import cv2
import openslide, pickle
from skimage import transform
import matplotlib
import torch
from typing import Tuple, List
from torch.utils.data import Dataset

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def check_kfb(filepath):
    try:
        assert os.path.exists(filepath), "Slide path doesnot exist"
        slide = openslide.open_slide(filepath)
        slide_width, slide_height = slide.level_dimensions[0]
        slide_name = os.path.basename(filepath)
        print("Slide {} original width: {}, height: {}".format(slide_name, slide_width, slide_height))

        return True
    except Exception as e:
        print(e)
        return False


class SlideDataset(Dataset):
    def __init__(self, slide_path: str, level: int = 1, imsize: int = 512,
                 to_real_scale: int = 4, overlap: int = 128) -> None:
        """
        Dataset for loading whole slide images in patches.

        Args:
            slide_path: Path to the slide file
            level: Slide level to read
            imsize: Size of output image
            to_real_scale: Scale factor
            overlap: Overlap between patches
        """
        assert os.path.exists(slide_path), f"Slide path does not exist: {slide_path}"

        self.level = level
        self.imsize = imsize
        self.scale = 1.0 / to_real_scale

        # Open slide
        self.slide = openslide.open_slide(slide_path)
        self.slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        w, h = self.slide.level_dimensions[self.level]

        # Calculate actual crop size
        level = 0 if to_real_scale > 1 else self.level
        self.level_ratio = self.slide.level_dimensions[0][0] // self.slide.level_dimensions[level][0]
        self.scale *= self.level_ratio
        self.actual_crop_size = int(self.imsize / self.scale)

        # Generate coordinate list
        y_coords = np.arange(0, h - self.actual_crop_size + 1, self.actual_crop_size - overlap)
        x_coords = np.arange(0, w - self.actual_crop_size + 1, self.actual_crop_size - overlap)
        self.locations = np.array(np.meshgrid(y_coords, x_coords)).T.reshape(-1, 2).astype(int)

        # Calculate output dimensions
        self.output_size = [
            int(h * self.scale),
            int(w * self.scale)
        ]

        # print(f'Dataset created for slide {self.slide_name}:')
        # print(f'- Dimensions: ({h}, {w}) at level {self.level}')
        # print(f'- Scale ratio: {self.level_ratio}, actual size: {self.actual_crop_size}')
        print(f'- Output size: {self.output_size}')
        print(f'- Total patches: {len(self.locations)}')

    def __len__(self) -> int:
        return len(self.locations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Get a single patch from the slide.

        Args:
            idx: Index of the patch

        Returns:
            Tuple containing:
                - img: Tensor of shape (3, H, W)
                - location: (y, x) coordinates
        """
        y, x = self.locations[idx]

        # Read region from slide
        origin_img = self.slide.read_region((x, y), self.level,
                                            (self.actual_crop_size, self.actual_crop_size))
        origin_img = np.asarray(origin_img)[:, :, 0:-1]  # remove alpha channel

        # Resize
        new_size = (int(origin_img.shape[1] * self.scale),
                    int(origin_img.shape[0] * self.scale))
        img = cv2.resize(origin_img, new_size, interpolation=cv2.INTER_NEAREST)

        # Convert to tensor with proper normalization
        img = img.astype(np.float32)  # Convert to float32 before creating tensor
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)

        return img, (int(y * self.scale), int(x * self.scale))

    def get_output_size(self) -> List[int]:
        """Returns the dimensions of the final output [height, width]"""
        return self.output_size

    def get_slide_name(self) -> str:
        """Returns the name of the slide without extension"""
        return self.slide_name


def collate_fn(batch: List[Tuple[torch.Tensor, Tuple[int, int]]]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Custom collate function to handle the batched data.

    Args:
        batch: List of (image, location) tuples

    Returns:
        Tuple containing:
            - Batched images tensor of shape (B, C, H, W)
            - List of (y, x) locations
    """
    images = torch.stack([item[0] for item in batch])
    locations = [item[1] for item in batch]
    return images, locations

def compute_grid_score(grid, threshold):
    grid[grid < threshold] = 0
    # return float(np.count_nonzero(grid))
    return float(grid.sum())


def patch_sampling(img_list, tot_samples=50, stride_ratio=0.1, sample_size=[256, 256], threshold=0.5):
    '''
    Direct sorting and selection for 256*256 patches.
    Input:
        img_list: [((x,y), seg, img), ...]
                (x, y) is the absolute coordinates respect to the whole slide
                seg: a probability segmentation
                img: is the corresponding image of seg
        tot_samples: number of patches to select
        stride_ratio: not used in this version
        sample_size: not used in this version
        threshold: probability threshold for filtering

    Output:
        samples: [((act_x, act_y), sample_img, sample_img_seg), ...] selected patches with coordinates
    '''
    print('patch_sampling', '{} patches'.format(len(img_list)))

    # Collect all patches and their scores
    all_candidates = []
    for (y, x), seg, img in img_list:
        # Calculate mean probability across the patch
        score = seg.mean()
        if score > threshold:  # Only keep patches with mean prob > threshold
            all_candidates.append({
                'score': score,
                'coords': (y, x),
                'img': img,
                'seg': seg
            })

    if not all_candidates:
        print('Warning: No valid candidates found!')
        return []

    # Sort by score in descending order
    sorted_candidates = sorted(all_candidates, key=lambda x: x['score'], reverse=True)

    # Take top tot_samples
    selected_candidates = sorted_candidates[:min(tot_samples, len(sorted_candidates))]

    # Generate final sample list
    samples = []
    for candidate in selected_candidates:
        act_y, act_x = candidate['coords']
        sample_img = candidate['img']
        sample_img_seg = candidate['seg']
        samples += [((act_x, act_y), sample_img, sample_img_seg)]

    print(f'Selected {len(samples)} patches out of {len(all_candidates)} candidates')
    return samples
#

# def patch_sampling(img_list, score_threshold, prob_threshold=0.5):
#     '''
#     Filter patches whose total score exceeds a threshold.
#
#     Input:
#     img_list: [((x,y), seg, img), ...] (x, y) are absolute coordinates relative to the whole slide
#     score_threshold: minimum total score to keep a patch
#     prob_threshold: probability threshold within the segmentation map
#
#     Output:
#     samples: [((act_x, act_y), sample_img, sample_img_seg), ...] selected patches with coordinates
#     '''
#     print('patch_sampling', '{} patches'.format(len(img_list)))
#
#     valid_candidates = []
#     for (y, x), seg, img in img_list:
#         seg = seg.copy()
#         seg[seg < prob_threshold] = 0
#
#         score = seg.sum()
#         if score > score_threshold:
#             valid_candidates.append({
#                 'score': score,
#                 'coords': (y, x),
#                 'img': img,
#                 'seg': seg
#             })
#
#     if not valid_candidates:
#         print('Warning: No valid candidates found!')
#         return []
#
#     samples = []
#     for candidate in valid_candidates:
#         act_y, act_x = candidate['coords']
#         sample_img = candidate['img']
#         sample_img_seg = candidate['seg']
#         samples += [((act_x, act_y), sample_img, sample_img_seg)]
#
#     print(f'Selected {len(samples)} patches out of {len(img_list)} total patches')
#     return samples
def gradient_merge(img1, img2, mask):
    y, x = np.where(mask != 0)
    y = np.unique(y)
    x = np.unique(x)
    res = img2.copy()
    if y.size > x.size:
        # overalpping is vertical
        for i, p in enumerate(x):
            w = i / (len(x) - 1)
            res[:, p] = res[:, p] * (1 - w) + w * img1[:, p]

    else:
        # overlapping is horizontal
        for i, p in enumerate(y):
            w = i / (len(y) - 1)
            res[p, :] = res[p, :] * (1 - w) + w * img2[p, :]

    return res


def visualize_sampling_points(target, locs):
    target_c = target.copy()
    # target is RGB (from openslide), convert to BGR for cv2
    target_c = cv2.cvtColor(target_c, cv2.COLOR_RGB2BGR)
    for x, y in locs:
        cv2.circle(target_c, (x + 128, y + 128), 100, [54, 47, 236], -1)
    # convert back to RGB for imageio
    target_c = cv2.cvtColor(target_c, cv2.COLOR_BGR2RGB)
    return target_c


def visualize_all_candidates_with_score(target, candidates, font_scale=1.5, thickness=2):
    """
    Visualize all candidate regions with their scores overlaid.

    Args:
        target: Background image (numpy array)
        candidates: List of [((y, x), seg, img), ...]
    """
    print(f"Visualizing {len(candidates)} candidates with scores...")
    target_c = target.copy()

    # 遍历所有 candidate
    for i, item in enumerate(candidates):
        # 解包数据
        (y, x) = item[0]
        seg_map = item[1]

        # 计算分数 (与 patch_sampling 逻辑保持一致)
        # 注意：这里的 seg_map 已经在主程序中经过了 tissue-weighted 过滤
        score = seg_map.sum()

        # 1. 画框 (使用不同颜色，例如青色，以区别于最终选出的绿色框)
        cv2.rectangle(target_c,
                      pt1=(x, y),
                      pt2=(x + 256, y + 256),  # 假设 patch size 为 256
                      color=(0, 255, 255),  # 黄色框
                      thickness=thickness,
                      lineType=8)

        # 2. 写分数
        # 格式化分数，保留1位小数
        text = f"{score:.1f}"

        # 为了文字清晰，先画黑色轮廓，再画文字
        cv2.putText(target_c, text, (x + 5, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)  # 黑色描边
        cv2.putText(target_c, text, (x + 5, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)  # 红色文字

    return target_c

def visualize_heatmap(wsi_seg, shape, stride, wsi_img, save_path):
    # import pdb; pdb.set_trace()
    # low_size = shape[0] // stride
    # res = np.zeros((low_size, low_size), dtype=np.float16)

    # for (x,y), im, seg in samples:
    #     v =  seg.sum() / (seg.shape[0] * seg.shape[1])
    #     x_, y_ = x // stride, y // stride
    #     res[y_, x_] = v
    # out = skimage.transform.pyramid_expand(res, upscale=stride, sigma=25)

    # seg_img = misc.imresize(wsi_seg, 0.02)
    seg_img = cv2.resize(wsi_seg, (0, 0), fx=0.02, fy=0.02)
    out = transform.pyramid_expand(seg_img, upscale=50, sigma=25)

    fig = plt.figure(figsize=(10, int(10 * (out.shape[0] / out.shape[1]))))

    # plt.imshow(wsi_img)
    # plt.imshow(out, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
    plt.imshow(out, cmap=plt.cm.jet)
    plt.axis('off')
    fig.tight_layout()
    fig.savefig(save_path + '_cam.jpg', bbox_inches='tight', pad_inches=0)
    # contour_img = vis_overlay(wsi_img, wsi_seg.astype(np.float32) / wsi_seg.max(), threshold=0.2)#目的是将wsi_seg的轮廓画在wsi_img上
    # misc.imsave(save_path + '_contour.jpg', contour_img)
    # cv2.imwrite(save_path + '_contour.jpg', contour_img)
    return out


def vis_overlay(im, pred, threshold=-1):
    res = []
    pred = pred > threshold
    im = im.astype(np.uint8)

    contours, hierarchy = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 10)

    return im


