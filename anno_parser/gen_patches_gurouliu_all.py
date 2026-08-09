import os, sys
import argparse
import openslide
from skimage import transform
from pycontour import poly_transform
from shapely.geometry import Point
from load_anno_gurouliu import load_annotation_gurouliu
import imageio
from tqdm import tqdm
import logging
import cv2
import numpy as np

import multiprocessing as mp



def get_save_dirs(args):

    if args.anno_type == "Neg":  # Neg annotations → label 3
        patch_label = "3"
    elif args.anno_type == "Pos":
        if args.WSI_type == "OS":
            patch_label="2"
        elif args.WSI_type == "NOS":
            patch_label="1"
        else:
            print("Unknown dataset {}".format(args.dset))
            sys.exit()
    else:
        print("Unknown annotation type {}".format(args.anno_type))
        sys.exit()

    img_save_dir = os.path.join(args.output_dir, "classification", args.dset, "img", patch_label)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    mask_save_dir = os.path.join(args.output_dir, "classification", args.dset, "groundTruth", patch_label)
    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    return img_save_dir, mask_save_dir
def gen_patch_mask(args):
    patch_mask = None
    if args.anno_type == "Neg":
        patch_mask = np.ones((args.crop_size, args.crop_size), dtype=np.uint8) * 0
    elif args.anno_type == "Pos":
        patch_mask = np.ones((args.crop_size, args.crop_size), dtype=np.uint8) * 0
    else:
        logging.error(f"Unknown annotation type {args.anno_type}")
        return None
    return patch_mask


def mask_polygon(patch_mask, coords, rand_w, rand_h, crop_size, anno_type):
    mask_value = 155 if anno_type == "Neg" else 255
    # Adjust coordinates relative to the crop window
    points = np.column_stack((coords[1, :] - rand_w, coords[0, :] - rand_h)).astype(np.int32)
    cv2.fillPoly(patch_mask, [points], mask_value)
    return patch_mask


def process_patch_batch(batch_data):
    slide_path, patches_info, coords, args, img_save_dir, mask_save_dir, cur_reg = batch_data
    slide_name = os.path.basename(slide_path)

    try:
        with openslide.OpenSlide(slide_path) as slide_head:
            for h, w, cen_h, cen_w in patches_info:
                cur_patch = slide_head.read_region((w, h), args.slide_level,
                                                   (args.crop_size, args.crop_size))

                cur_patch = np.asarray(cur_patch)[:, :, :3]

                patch_mask = gen_patch_mask(args)
                patch_mask = mask_polygon(patch_mask, coords, w, h, args.crop_size, args.anno_type)

                save_img = transform.resize(cur_patch, (args.save_size, args.save_size))
                save_mask = transform.resize(patch_mask, (args.save_size, args.save_size))
                save_mask = (save_mask * 255).astype(np.uint8)
                save_img = (save_img * 255).astype(np.uint8)

                # Use patch coordinates as a unique identifier in the filename
                img_fullname = f"{os.path.splitext(slide_name)[0]}_{cur_reg}_h{h}_w{w}.png"
                save_img_path = os.path.join(img_save_dir, img_fullname)
                save_mask_path = os.path.join(mask_save_dir, img_fullname)

                imageio.imwrite(save_img_path, save_img)
                imageio.imwrite(save_mask_path, save_mask)

                del cur_patch, patch_mask, save_img, save_mask

    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")

    return len(patches_info)

def process_single_region(region_data):
    """Process a single annotation region — extract patches and save them."""
    slide_path, cur_reg, coords, args, level_scale, level_dimensions = region_data
    slide_name = os.path.basename(slide_path)

    try:
        img_save_dir, mask_save_dir = get_save_dirs(args)

        coords = np.array(coords, dtype=np.float32)
        coords = coords / level_scale
        coords = coords.astype(np.int32)
        coords = np.transpose(coords)
        coords[[0, 1]] = coords[[1, 0]]

        # Compute bounding box
        min_h, max_h = np.min(coords[0, :]), np.max(coords[0, :])
        min_w, max_w = np.min(coords[1, :]), np.max(coords[1, :])

        # Snap bounds to grid alignment
        min_h = (min_h // args.crop_size) * args.crop_size
        min_w = (min_w // args.crop_size) * args.crop_size

        cur_poly = poly_transform.np_arr_to_poly(np.asarray(coords))

        patches_info = []
        h_positions = np.arange(min_h, max_h, args.crop_size)
        w_positions = np.arange(min_w, max_w, args.crop_size)

        for h in h_positions:
            for w in w_positions:
                cen_h = int(h + args.crop_size / 2)
                cen_w = int(w + args.crop_size / 2)

                if (h + args.crop_size >= level_dimensions[1] or
                        w + args.crop_size >= level_dimensions[0]):
                    continue

                if Point(cen_w, cen_h).within(cur_poly):
                    patches_info.append((h, w, cen_h, cen_w))

        # Split into batches for parallel patch extraction
        batch_size = 64
        patches_batches = [patches_info[i:i + batch_size]
                           for i in range(0, len(patches_info), batch_size)]

        total_processed = 0
        from multiprocessing.pool import ThreadPool
        inner_threads = max(1, mp.cpu_count() // 2)

        with ThreadPool(processes=inner_threads) as inner_pool:
            batch_results = list(inner_pool.imap_unordered(
                process_patch_batch,
                [(slide_path, batch, coords, args, img_save_dir, mask_save_dir, cur_reg)
                 for batch in patches_batches]
            ))

        # Garbage collect after each region to free memory
        import gc
        gc.collect()

        total_processed = sum(batch_results)
        return cur_reg, total_processed, None

    except Exception as e:
        return cur_reg, 0, str(e)

def process_single_slide(slide_path, anno_path, args):
    """Process a single WSI file — load annotations, extract patches from all regions."""
    slide_name = os.path.basename(slide_path)

    try:
        annotation_dict = load_annotation_gurouliu(anno_path)

        with openslide.OpenSlide(slide_path) as slide_head:
            if args.slide_level < 0 or args.slide_level >= slide_head.level_count:
                logging.error(f"Level {args.slide_level} not available in {slide_name}")
                return False
            level_scale = slide_head.level_downsamples[args.slide_level]
            level_dimensions = slide_head.level_dimensions[args.slide_level]

        # Prepare region data for parallel processing
        region_data = [(slide_path, reg, coords, args, level_scale, level_dimensions)
                       for reg, coords in annotation_dict.items()]

        num_processes = mp.cpu_count() - 1
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_region, region_data),
                total=len(region_data),
                desc=f"Processing regions in {slide_name}"
            ))

        # Aggregate results across regions
        total_patches = 0
        failed_regions = []
        for reg_id, patch_count, error in results:
            if error:
                logging.warning(f"Failed to process region {reg_id} in {slide_name}: {error}")
                failed_regions.append(reg_id)
            else:
                total_patches += patch_count

        return True

    except Exception as e:
        logging.error(f"Failed to process {slide_name}: {str(e)}")
        return False

def main():
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    slide_files = [f for f in os.listdir(args.slides_dir) if f.endswith('.tif')]
    print('slide_files:', slide_files)

    successful_files = []
    failed_files = []

    # Process WSI files sequentially; regions within each slide are parallelized
    for slide_file in slide_files:
        slide_name = os.path.splitext(slide_file)[0]
        slide_path = os.path.join(args.slides_dir, slide_file)
        anno_path = os.path.join(args.annos_dir, f"{slide_name}.tif.geojson")

        if not os.path.exists(anno_path):
            logging.warning(f"Annotation file not found for {slide_file}")
            failed_files.append(slide_file)
            continue

        success = process_single_slide(slide_path, anno_path, args)
        if success:
            successful_files.append(slide_file)
        else:
            failed_files.append(slide_file)

    # Print processing summary
    logging.info("\nProcessing Summary:")
    logging.info(f"Total files: {len(slide_files)}")
    logging.info(f"Successfully processed: {len(successful_files)}")
    logging.info(f"Failed: {len(failed_files)}")

    if failed_files:
        logging.info("\nFailed files:")
        for file in failed_files:
            logging.info(file)


def set_args():
    parser = argparse.ArgumentParser(description='Batch processing of WSI slides and annotations')
    parser.add_argument('--slides_dir', type=str, default='/home/pengxiao/disk/新增30张wsi&标注 ＋excel/NOS/两',
                        help="Directory containing .tif files")#TODO
    parser.add_argument('--annos_dir', type=str, default='/home/pengxiao/disk/新增30张wsi&标注 ＋excel/annotation（NOS25 OS5）/',
                        help="Directory containing annotation files")#TODO
    parser.add_argument('--output_dir', type=str, default='/data/pengxiao/Osteosarcoma_diagnosis/data/',
                        help="Directory to save output patches and masks")
    parser.add_argument('--anno_type', type=str, default="Pos", choices=["Pos", "Neg"])#    TODO
    parser.add_argument('--dset', type=str, default="train", choices=["train", "test"])#TODO
    parser.add_argument('--WSI_type',type=str,default='NOS')#TODO
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--save_size', type=int, default=256)
    parser.add_argument('--slide_level', type=int, default=0)
    parser.add_argument('--patches_per_region', type=int, default=10,
                        help="Number of patches to extract per region")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    mp.freeze_support()
    main()