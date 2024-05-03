import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def visualize_augmentations(dataset_dir):
    # Set up directories
    train_dir = os.path.join(dataset_dir, 'train')

    # Create a basic generator to fetch the original image
    basic_datagen = ImageDataGenerator()
    train_generator = basic_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        color_mode='grayscale',
        batch_size=1,
        class_mode='categorical',
        shuffle=True
    )

    # Fetch the original image
    original_img = next(train_generator)[0][0]

    # Individual Augmentation Generators
    rescale_gen = ImageDataGenerator(rescale=1. / 255)
    rotation_gen = ImageDataGenerator(rotation_range=10)
    width_shift_gen = ImageDataGenerator(width_shift_range=0.2)
    height_shift_gen = ImageDataGenerator(height_shift_range=0.2)
    zoom_gen = ImageDataGenerator(zoom_range=0.1)
    shear_gen = ImageDataGenerator(shear_range=0.2)
    horizontal_flip_gen = ImageDataGenerator(horizontal_flip=True)
    vertical_flip_gen = ImageDataGenerator(vertical_flip=True)

    generators = [rescale_gen, rotation_gen, width_shift_gen, height_shift_gen, zoom_gen, shear_gen,
                  horizontal_flip_gen, vertical_flip_gen]
    titles = ["Rescale", "Rotation", "Width Shift", "Height Shift", "Zoom", "Shear", "Horizontal Flip", "Vertical Flip"]

    # Visualization
    num_rows = 3
    num_cols = 3
    plt.figure(figsize=(21, 21))
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(original_img[:, :, 0], cmap='gray')
    plt.title("Original Image", fontsize=40)
    plt.axis('off')

    for i, (gen, title) in enumerate(zip(generators, titles), start=2):
        augmented_img = gen.flow(np.expand_dims(original_img, 0))[0][0]
        plt.subplot(num_rows, num_cols, i)
        plt.imshow(augmented_img[:, :, 0], cmap='gray')
        plt.title(title, fontsize=40)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_dir = r"D:\ECGData\ECGImages4C4LColorInversionLow"
    visualize_augmentations(dataset_dir)
