# create a data generator that uses both original and augmented data for training. ###
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A

class DataGenerator(Sequence):
    def __init__(self, image_dirs, mask_dirs, batch_size, img_size, augs=None):
        self.image_dirs = image_dirs
        self.mask_dirs = mask_dirs
        self.batch_size = batch_size
        self.img_size = img_size
        self.augs = augs
        self.image_filenames = []
        self.mask_filenames = []
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            self.image_filenames.extend([os.path.join(image_dir, fname) for fname in sorted(os.listdir(image_dir))])
            self.mask_filenames.extend([os.path.join(mask_dir, fname) for fname in sorted(os.listdir(mask_dir))])
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of range for data generator with length {len(self)}')
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, masks = [], []
        for img_file, mask_file in zip(batch_images, batch_masks):
            # Load and preprocess image and mask
            img = load_img(img_file, target_size=self.img_size, color_mode='grayscale')
            img = img_to_array(img) / 255.0  # Normalize images to [0, 1]
            mask = load_img(mask_file, target_size=self.img_size, color_mode='grayscale')
            mask = img_to_array(mask) / 255.0  # Normalize masks to [0, 1]
            mask = (mask > 0.5).astype(np.float32)  # Binarize masks
            # Apply augmentations if specified
            if self.augs:
                augmented = self.augs(image=img, mask=mask)
                img, mask = augmented['image'], augmented['mask']
            images.append(img)
            masks.append(mask)
        return np.array(images), np.array(masks)