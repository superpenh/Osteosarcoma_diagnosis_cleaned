import os, sys, pdb
import glob, random
import numpy as np
import itertools
from collections import Counter
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.color import rgb2hed, hed2rgb

from data_generator.image import ImageDataGenerator


def apply_he_augmentation_with_mask(image):
    """
    Apply H&E stain color augmentation to a histopathology image while preserving white background regions.

    Args:
        image: Input RGB image with values in [0, 1].

    Returns:
        Augmented RGB image.
    """
    # Create a mask for white/background regions (pixels where all RGB channels exceed threshold)
    white_threshold = 0.8
    white_mask = np.all(image > white_threshold, axis=2)

    # Preserve original white region colors for restoration after augmentation
    original_white = image.copy()

    hed = rgb2hed(np.clip(image, 0, 1.0))

    # Generate random linear transform parameters for Hematoxylin and Eosin channels
    ah = 0.98 + random.random() * 0.04
    bh = -0.02 + random.random() * 0.04
    ae = 0.98 + random.random() * 0.04
    be = -0.02 + random.random() * 0.04

    hed_augmented = hed.copy()
    hed_augmented[:, :, 0] = ah * hed[:, :, 0] + bh  # H channel
    hed_augmented[:, :, 1] = ae * hed[:, :, 1] + be  # E channel

    rgb_augmented = hed2rgb(hed_augmented)
    rgb_augmented = np.clip(rgb_augmented, 0, 1.0)

    # Restore white/background regions to their original colors
    rgb_augmented[white_mask] = original_white[white_mask]

    return rgb_augmented


def preprocess(img, mean, std, label, normalize_label=False):
    """Preprocess image and label."""
    out_img = img / img.max()  # scale to [0,1]
    out_img = (out_img - np.array(mean).reshape(1, 1, 3)) / np.array(std).reshape(1, 1, 3)

    if normalize_label:
        if np.unique(label).size > 2:
            print('WARNING: the label has more than 2 classes. Set normalize_label to False')
        label = label / label.max()
    return out_img, label.astype(np.int32)


def deprocess(img, mean, std, label):
    out_img = img / img.max()  # scale to [0,1]
    out_img = (out_img * np.array(std).reshape(1, 1, 3)) + np.array(mean).reshape(1, 1, 3)
    out_img = out_img * 255.0

    return out_img.astype(np.uint8), label.astype(np.uint8)


def he_augmentation_generator(generator):
    """
    Wrap an ImageDataGenerator to apply H&E stain augmentation on the fly.
    """
    for x, y in generator:
        x_normalized = x / 255.0

        for i in range(x.shape[0]):
            x_normalized[i] = apply_he_augmentation_with_mask(x_normalized[i])

        x_augmented = (x_normalized * 255.0).astype(np.uint8)

        yield x_augmented, y


def data_loader(path, batch_size, imSize,
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                ignore_val=0, pos_val=255, neg_val=155, pos_class=[0, 1], neg_class=[2],
                use_he_augmentation=True):
    """
    Data loader with optional H&E stain augmentation for histopathology images.
    """

    def imerge(img_gen, mask_gen):
        if use_he_augmentation:
            img_gen = he_augmentation_generator(img_gen)

        for (imgs, img_labels), (mask, mask_labels) in itertools.zip_longest(img_gen, mask_gen):
            # Compute weight to ignore particular pixels
            mask = mask[:, :, :, 0]
            weight = np.ones(mask.shape, np.float32)
            weight[mask == ignore_val] = 0

            for c, mask_label in enumerate(mask_labels):
                assert (mask_labels[c] == img_labels[c])
                mask_pointer = mask[c]
                if mask_label in pos_class:
                    mask_pointer[mask_pointer != pos_val] /= 255.0
                    mask_pointer[mask_pointer == pos_val] = 1
                elif mask_label in neg_class:
                    assert (np.where(mask_pointer == pos_val)[0].size == 0)
                    mask_pointer[mask_pointer != neg_val] /= 255.0
                    mask_pointer[mask_pointer == neg_val] = 0
                else:
                    print('WARNING: mask beyond the expected class range')
                    mask_pointer /= 255.0
                mask_pointer[mask_pointer == ignore_val] = 0

            assert np.all((mask >= 0) & (mask <= 1)), "mask contains values outside 0 and 1"

            yield imgs, mask, weight, img_labels

    # Data augmentation parameters optimized for H&E stained images
    train_data_gen_args = dict(
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='reflect')

    seed = 1234
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
        path + 'train/img',
        class_mode="sparse",
        target_size=(imSize, imSize),
        batch_size=batch_size,
        seed=seed)

    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
        path + 'train/groundTruth',
        class_mode="sparse",
        target_size=(imSize, imSize),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=seed)

    test_image_datagen = ImageDataGenerator().flow_from_directory(
        path + 'test/img',
        class_mode="sparse",
        target_size=(imSize, imSize),
        batch_size=batch_size,
        seed=seed)

    test_mask_datagen = ImageDataGenerator().flow_from_directory(
        path + 'test/groundTruth',
        class_mode="sparse",
        target_size=(imSize, imSize),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=seed)

    train_generator = imerge(train_image_datagen, train_mask_datagen)
    test_generator = imerge(test_image_datagen, test_mask_datagen)

    sys.stdout.flush()
    return train_generator, test_generator, train_image_datagen.samples, test_image_datagen.samples
