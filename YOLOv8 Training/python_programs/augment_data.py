# I used this to augment data, this is not necessary to be used unless you want to increase the dataset

import os
import cv2
import albumentations as A
from PIL import Image
import numpy as np
import shutil

# Path to your image and label folders
input_image_folder = r"C:\Users\gabri\Desktop\dataset\images"
input_label_folder = r"C:\Users\gabri\Desktop\dataset\labels"
output_image_folder = r"C:\Users\gabri\Desktop\augumented_dataset\augmented_images"
output_label_folder = r"C:\Users\gabri\Desktop\augumented_dataset\augmented_labels"

# Make sure output folders exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

# Define the augmentation pipeline with bounding box support
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),       # Flip horizontally
    A.VerticalFlip(p=0.2),         # Flip vertically
    A.RandomRotate90(p=0.5),       # Rotate by 90 degrees
    A.RandomBrightnessContrast(p=0.3),  # Adjust brightness and contrast
    A.ShiftScaleRotate(p=0.2),     # Randomly shift, scale, and rotate
    A.Blur(p=0.2),                 # Add blur
    A.GaussNoise(p=0.2)            # Add Gaussian noise
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

# Number of augmentations per image to reach the target dataset size
augmentations_per_image = 4 

# Augment the dataset
for img_name in os.listdir(input_image_folder):
    if img_name.endswith('.jpg') or img_name.endswith('.png'):
        # Load the image and corresponding label
        img_path = os.path.join(input_image_folder, img_name)
        label_path = os.path.join(input_label_folder, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        if os.path.exists(label_path):
            # Read image
            image = np.array(Image.open(img_path))

            # Read label (YOLO format)
            bboxes = []
            category_ids = []
            with open(label_path, 'r') as label_file:
                for line in label_file:
                    label_data = line.strip().split()
                    category_ids.append(int(label_data[0]))  # Class ID
                    bbox = list(map(float, label_data[1:]))  # YOLO format: center_x, center_y, width, height
                    bboxes.append(bbox)

            # Apply augmentations multiple times
            for i in range(augmentations_per_image):
                # Apply augmentations with bounding boxes
                augmented = augmentor(image=image, bboxes=bboxes, category_ids=category_ids)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']

                # Save augmented image
                aug_img_name = f"aug_{img_name.replace('.jpg', f'_{i}.jpg').replace('.png', f'_{i}.png')}"
                aug_img_path = os.path.join(output_image_folder, aug_img_name)
                aug_image_pil = Image.fromarray(aug_image)
                aug_image_pil.save(aug_img_path)

                # Save augmented label
                aug_label_name = f"aug_{img_name.replace('.jpg', f'_{i}.txt').replace('.png', f'_{i}.txt')}"
                aug_label_path = os.path.join(output_label_folder, aug_label_name)
                with open(aug_label_path, 'w') as aug_label_file:
                    for bbox, category_id in zip(aug_bboxes, category_ids):
                        bbox_str = " ".join(map(str, bbox))
                        aug_label_file.write(f"{category_id} {bbox_str}\n")

                # Debug: Confirm files are saved
                print(f"Saved augmented image: {aug_img_name}")
                print(f"Saved augmented label: {aug_label_name}")
