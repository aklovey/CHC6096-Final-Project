# Wyatt
import os
import random
from datetime import datetime

import h5py
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from keras.layers import DepthwiseConv2D, SeparableConv2D
from keras.models import Sequential, Model
from keras import optimizers, Input
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, label_binarize
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

# Define the custom activation function
def clip_relu(x):
    return K.clip(x, 0, 6)

def relu6(x):
    return Activation(clip_relu)(x)

def DepthwiseSeparableConvWithResidual_MobileNetV1(input_shape, number_classes):
    inputs = Input(shape=input_shape)

    # Initial layer
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = relu6(x)

    # Depthwise Separable Conv Block
    def ds_conv_block(x, channels):
        # Depthwise Convolution
        x = DepthwiseConv2D((3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = relu6(x)

        # Pointwise Convolution
        x = Conv2D(channels, (1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = relu6(x)

        return x

    # Blocks with residual connection
    residual = Conv2D(64, (1, 1), padding='same')(inputs)

    residual = MaxPooling2D(pool_size=(2, 2))(residual)
    x = ds_conv_block(x, 64)
    x = Add()([x, residual])

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same')(x)
    x = ds_conv_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Add()([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same')(x)
    x = ds_conv_block(x, 256)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Add()([x, residual])

    residual = Conv2D(512, (1, 1), strides=(2, 2), padding='same')(x)
    x = ds_conv_block(x, 512)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Add()([x, residual])

    # Fully Connected Layers
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(number_classes, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    return model

def analysisShap(save_dir):
        shap_samples = test_images_small[:10]  # 选择10个样本进行SHAP分析
        explainer = shap.GradientExplainer(model, shap_samples)
        shap_values = explainer.shap_values(shap_samples)

        # 绘制SHAP值
        shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
        plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'))  # 保存SHAP摘要图
        plt.show()



if __name__ == '__main__':

    dataset_dir = r'D:\ECGData\ECGImages303threshold=40.0'

    num_classes = 5

    input_shape = (224, 224, 1)
    batch_size = 16


    def load_data_from_h5(file_path):
        images = []
        labels = []

        with h5py.File(file_path, 'r') as f:
            for top_group_name, top_group in f.items():
                for subgroup_name, subgroup in top_group.items():
                    for group_name, group in subgroup.items():
                        if 'image' in group:
                            img = group['image'][:]
                            img_attrs = {label_name: value for label_name, value in group.attrs.items()}

                            # Assuming you want to store these attributes as a categorical array or similar
                            label_array = [img_attrs.get(label_name, 0.0) for label_name in sorted(img_attrs.keys())]

                            images.append(img)
                            labels.append(label_array)
                        else:
                            print(
                                f"Warning: Group '{group_name}' in '{subgroup_name}' in '{top_group_name}' does not contain an 'image' dataset.")

        return np.array(images), np.array(labels)


    # Load the data from the .h5 files
    train_images, train_labels = load_data_from_h5(os.path.join(dataset_dir, 'train.h5'))
    val_images, val_labels = load_data_from_h5(os.path.join(dataset_dir, 'val.h5'))
    test_images, test_labels = load_data_from_h5(os.path.join(dataset_dir, 'test.h5'))

    train_images = np.expand_dims(train_images, -1)
    val_images = np.expand_dims(val_images, -1)
    test_images = np.expand_dims(test_images, -1)

    # Print shapes for debugging
    print("Shape of train_images:", train_images.shape)
    print("Shape of train_labels:", train_labels.shape)
    print("Shape of val_images:", val_images.shape)
    print("Shape of val_labels:", val_labels.shape)
    print("Shape of test_images:", test_images.shape)
    print("Shape of test_labels:", test_labels.shape)

    # ... rest of the code ...

    # seed_value = 42
    seed_value = 23
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=10,
                                       width_shift_range=0.2,
                                       height_shift_range=0.1,
                                       # horizontal_flip=True,
                                       zoom_range=0.1,
                                       # seed=seed_value,
                                       # height_shift_range=0.2,
                                       # shear_range=0.2,
                                       )
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    import numpy as np



    # Assuming train_images, val_images, and test_images are numpy arrays
    # Calculate the indices for slicing the datasets to one-tenth
    train_idx = int(0.1 * train_images.shape[0])
    val_idx = int(0.1 * val_images.shape[0])
    test_idx = int(0.1 * test_images.shape[0])

    # Slice the datasets to keep only one-tenth of the data
    train_images_small = train_images[:train_idx]
    train_labels_small = train_labels[:train_idx]

    val_images_small = val_images[:val_idx]
    val_labels_small = val_labels[:val_idx]

    test_images_small = test_images[:test_idx]
    test_labels_small = test_labels[:test_idx]


    # Print shapes for debugging
    print("One tenth sample fast dataset:\n")
    print("Shape of train_images:", train_images_small.shape)
    print("Shape of train_labels:", train_labels_small.shape)
    print("Shape of val_images:", val_images_small.shape)
    print("Shape of val_labels:", val_labels_small.shape)
    print("Shape of test_images:", test_images_small.shape)
    print("Shape of test_labels:", test_labels_small.shape)

    # Continue with the ImageDataGenerator as before, but use the smaller datasets
    train_generator = train_datagen.flow(train_images_small, train_labels_small, batch_size=batch_size, seed=seed_value)
    validation_generator = val_datagen.flow(val_images_small, val_labels_small, batch_size=batch_size, shuffle=False)
    test_generator = test_datagen.flow(test_images_small, test_labels_small, batch_size=batch_size, shuffle=False)

    # Register the custom activation function with Keras
    get_custom_objects().update({'relu6': Activation(clip_relu)})

    model = DepthwiseSeparableConvWithResidual_MobileNetV1(input_shape, num_classes)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])



    best_weights_save_load_dir = f"DSCWithResnetForECGImages20240304_160145.h5"

    model.summary()

    import matplotlib.pyplot as plt


    def plot_images(images, labels, num_images=5):
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].squeeze(), cmap='gray')  # Adjust depending on your data format
            ax.title.set_text(f"Label: {labels[i]}")
            ax.axis('off')
        plt.show()


    # # Visualize original training images
    # plot_images(test_images_small, test_labels_small)
    # # Generate and visualize augmented images
    # sample_generator = train_datagen.flow(test_images_small, test_labels_small, batch_size=5, seed=42)
    # augmented_images, augmented_labels = next(sample_generator)
    # plot_images(augmented_images, augmented_labels)





    # save_dir = r"C:\Users\Wyatt\PycharmProjects\12-leadsECGuseingDSCwithResnetModel\Test result20240304_163854"


    # last_conv_layer_name = "conv2d_8"

    last_conv_layer_name = "conv2d"
    # # last_conv_layer_name = "  conv2d_13"

    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, save_dir=None):
        grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        # Convert the heatmap to numpy array and ensure it is float type for plotting
        heatmap = heatmap.numpy().astype('float')

        # Plotting
        if save_dir:
            fig, ax = plt.subplots()  # Create a figure and a set of subplots
            cax = ax.matshow(heatmap, cmap='jet')  # Display the heatmap
            plt.title("Grad-Cam Heatmap")
            fig.colorbar(cax)  # Add a color bar to the figure based on the heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'gradcam_heatmap_{timestamp}.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)  # Save the figure with a unique filename
            plt.close(fig)
        return heatmap


    def visualize_heatmap(sample_image, model, last_conv_layer_name, save_dir=None):
        # 显示原始图像
        plt.figure(figsize=(6, 6))
        plt.imshow(sample_image.squeeze(), cmap='gray')  # 确保图像是灰度显示
        plt.title("Original Image")
        plt.show()

        # 生成热图并显示
        heatmap = make_gradcam_heatmap(np.expand_dims(sample_image, axis=0), model, last_conv_layer_name,
                                       save_dir=save_dir)
        plt.figure(figsize=(6, 6))
        plt.imshow(sample_image.squeeze(), cmap='gray')  # 原始图像
        im = plt.imshow(heatmap, cmap='jet', alpha=0.5)  # 叠加热图
        plt.title("Heatmap Overlay")
        plt.colorbar(im)  # 为热图添加色标
        plt.show()

        # Visualize heatmap for the first test image


    visualize_heatmap(test_images_small[0], model, last_conv_layer_name,
                      save_dir=r"C:\Users\Wyatt\PycharmProjects\12-leadsECGuseingDSCwithResnetModel\Test result20240304_163854")

    # Visualize heatmap for the first test image
    visualize_heatmap(test_images_small[1], model, last_conv_layer_name,
                      save_dir=r"C:\Users\Wyatt\PycharmProjects\12-leadsECGuseingDSCwithResnetModel\Test result20240304_163854")

    # Visualize heatmap for the first test image
    visualize_heatmap(test_images_small[2], model, last_conv_layer_name,
                      save_dir=r"C:\Users\Wyatt\PycharmProjects\12-leadsECGuseingDSCwithResnetModel\Test result20240304_163854")

    # Visualize heatmap for the first test image
    visualize_heatmap(test_images_small[20], model, last_conv_layer_name,
                      save_dir=r"C:\Users\Wyatt\PycharmProjects\12-leadsECGuseingDSCwithResnetModel\Test result20240304_163854")









