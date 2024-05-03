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
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import LeakyReLU
import json
from tensorflow.keras.regularizers import l2
import time
from tensorflow.keras.callbacks import TensorBoard

def ShowHistory(history):
    # Create a directory for saving the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("Test result" + timestamp)
    os.makedirs(save_dir, exist_ok=True)


    # Training Accuracy and Validation accuracy graph
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation accuracy'], loc='lower right')
    plt.savefig(os.path.join(save_dir, "Model_Accuracy.png"))
    plt.show()

    # Loss
    plt.figure(figsize=(16, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.savefig(os.path.join(save_dir, "Model_Loss.png"))
    plt.show()

    return save_dir

from io import StringIO
import sys

def get_model_summary(model):
    stream = StringIO()
    sys.stdout = stream  # Redirect stdout to capture the printed model summary
    model.summary()  # This will print the summary to our stream buffer
    sys.stdout = sys.__stdout__  # Reset stdout
    return stream.getvalue()  # Return the captured string

def evaluateModel(model, test_generator,save_dir):
    classes = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
    # classes = ['CD', 'MI', 'NORM', 'STTC']
    num_classes = len(classes)

    start_time = time.time()

    # Evaluate the model using the test generator
    loss, accuracy = model.evaluate(test_generator)

    end_time = time.time()  # Capture the end time
    testing_duration = end_time - start_time  # Calculate the duration
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(test_generator)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Reset the generator
    test_generator.reset()

    # Determine number of steps per epoch
    steps_per_epoch = len(test_generator)

    y_true_list = []
    for i in range(steps_per_epoch):
        _, labels_batch = next(test_generator)
        y_true_list.extend(np.argmax(labels_batch, axis=1))

    y_true = np.array(y_true_list)

    confusion = confusion_matrix(y_true, y_pred_labels)


    # print("confusion_matrix: ")
    # print(confusion)



    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, cmap='YlGnBu', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, "Confusion_Matrix.png"))
    plt.show()

    f1 = f1_score(y_true, y_pred_labels, average=None)
    for i, score in enumerate(f1):
        print(f"F1 score for class {classes[i]}: {score}")

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    # y_pred_bin = label_binarize(y_pred_labels, classes=list(range(num_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Compute micro-average Precision-Recall curve and area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_pred.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    macro_avg_auc = np.mean([roc_auc[i] for i in range(num_classes)])
    print("AUC Scores:")
    for i, cls in enumerate(classes):
        print(f"AUC for class {cls}: {roc_auc[i]:.4f}")
    print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
    print(f"Macro-average AUC: {macro_avg_auc:.4f}\n")  # 打印宏平均AUC

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "Receiver Operating Characteristic Curve.png"))
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                                                ''.format(i, pr_auc[i]))


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "Precision-Recall Curve"))
    plt.show()



    # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate) for each class
    for i, cls in enumerate(classes):
        tp = confusion[i, i]
        fn = sum(confusion[i, :]) - tp
        fp = sum(confusion[:, i]) - tp
        tn = sum(sum(confusion)) - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        print(f"Sensitivity for {cls}: {sensitivity:.4f}")
        print(f"Specificity for {cls}: {specificity:.4f}")
        print("------")

    history_file_path = os.path.join(save_dir, "history.json")
    with open(history_file_path, "w") as hist_file:
        json.dump(history.history, hist_file)

    # Save the model configuration to a separate JSON file
    config_file_path = os.path.join(save_dir, "model_config.json")
    with open(config_file_path, "w") as config_file:
        config = model.get_config()
        json.dump(config, config_file, indent=4)  # Convert dictionary to pretty-printed JSON string
    # Save test loss, accuracy, F1 score, sensitivity, specificity, confusion matrix, ROC and Precision-Recall curve to a txt file
    with open(os.path.join(save_dir, "test_results.txt"), "w") as f:

        f.write(dataset_dir)
        f.write('\n')
        f.write(f"Test loss: {loss}\n")
        f.write(f"Test accuracy: {accuracy}\n\n")
        f.write(f"\nTesting Duration: {testing_duration:.2f} seconds\n")

        f.write("AUC Scores:\n")
        for i, cls in enumerate(classes):
            f.write(f"AUC for class {cls}: {roc_auc[i]:.4f}\n")
        f.write(f"Micro-average AUC: {roc_auc['micro']:.4f}\n")
        f.write(f"Macro-average AUC: {macro_avg_auc:.4f}\n\n")  # 保存宏平均AUC

        f.write("\nModel Architecture:\n")
        f.write(get_model_summary(model))

        f.write("Confusion Matrix:\n")
        for row in confusion:
            f.write(' '.join(map(str, row)) + '\n')
        f.write("\n")

        for i, score in enumerate(f1):
            f.write(f"F1 score for class {classes[i]}: {score}\n")

        for i, cls in enumerate(classes):
            tp = confusion[i, i]
            fn = sum(confusion[i, :]) - tp
            fp = sum(confusion[:, i]) - tp
            tn = sum(sum(confusion)) - (tp + fn + fp)

            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

            f.write(f"Sensitivity for {cls}: {sensitivity:.4f}\n")
            f.write(f"Specificity for {cls}: {specificity:.4f}\n")
        f.write("\n")

        for i in range(num_classes):
            f.write(f"ROC Curve for class {classes[i]}:\n")
            for j in range(len(fpr[i])):
                f.write(f"FPR: {fpr[i][j]}, TPR: {tpr[i][j]}\n")
            f.write("\n")

        for i in range(num_classes):
            f.write(f"Precision-Recall Curve for class {classes[i]}:\n")
            for j in range(len(precision[i])):
                f.write(f"Recall: {recall[i][j]}, Precision: {precision[i][j]}\n")
            f.write("\n")


import shap


def analysisShap(save_dir):
    shap_samples = test_images_small[:10]  # 选择10个样本进行SHAP分析
    explainer = shap.GradientExplainer(model, shap_samples)
    shap_values = explainer.shap_values(shap_samples)

    # 绘制SHAP值
    shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'))  # 保存SHAP摘要图
    plt.show()


from tensorflow.keras.models import Model



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

    # 绘制热图
    plt.matshow(heatmap)
    plt.savefig(os.path.join(save_dir, 'gradcam_heatmap.png'))  # 保存Grad-CAM热图
    plt.show()
    # plt.close()  # 关闭图表以避免在notebook中显示


def evaluateModel_NoWrite(model, test_generator):
    # classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    classes = ['CD', 'MI', 'NORM', 'STTC']
    num_classes = len(classes)

    # Evaluate the model using the test generator
    loss, accuracy = model.evaluate(test_generator)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(test_generator)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Determine number of steps per epoch
    steps_per_epoch = len(test_generator)

    y_true_list = []
    for i in range(steps_per_epoch):
        _, labels_batch = next(test_generator)
        y_true_list.extend(np.argmax(labels_batch, axis=1))

    y_true = np.array(y_true_list)

    confusion = confusion_matrix(y_true, y_pred_labels)
    print("confusion_matrix: ")
    print(confusion)



    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, cmap='YlGnBu', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    f1 = f1_score(y_true, y_pred_labels, average=None)
    for i, score in enumerate(f1):
        print(f"F1 score for class {classes[i]}: {score}")



    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_pred_bin = label_binarize(y_pred_labels, classes=list(range(num_classes)))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Compute micro-average Precision-Recall curve and area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                                                ''.format(i, pr_auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate) for each class
    for i, cls in enumerate(classes):
        tp = confusion[i, i]
        fn = sum(confusion[i, :]) - tp
        fp = sum(confusion[:, i]) - tp
        tn = sum(sum(confusion)) - (tp + fn + fp)

        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        print(f"Sensitivity for {cls}: {sensitivity:.4f}")
        print(f"Specificity for {cls}: {specificity:.4f}")
        print("------")



def DepthwiseSeparableConvWithResidual_M10D11_V1(input_shape, number_classes):
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 1
    residual = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])


    # Residual block 2
    residual = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    sx = Add()([x, residual])


    # Residual block 3
    residual = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(sx)
    residual = BatchNormalization()(residual)
    x = DepthwiseConv2D((3, 3), padding='same')(sx)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    ex = MaxPooling2D(pool_size=(2, 2))(sx)
    ex = BatchNormalization()(ex)
    concatenatedx = Concatenate(axis=-1)([x, ex])
    concatenatedx = Dropout(0.2)(concatenatedx)

    # Residual block 3
    residual = concatenatedx
    residual = BatchNormalization()(residual)
    x = DepthwiseConv2D((3, 3), padding='same')(concatenatedx)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(768, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    # Residual block 4
    residual = Conv2D(768, (3, 3), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(768, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])


    sx = MaxPooling2D(pool_size=(2, 2))(sx)
    sx = BatchNormalization()(sx)
    sx = MaxPooling2D(pool_size=(2, 2))(sx)
    sx = BatchNormalization()(sx)
    concatenatedx = Concatenate(axis=-1)([x, sx])
    concatenatedx = Dropout(0.2)(concatenatedx)


    # Residual block 4
    residual = concatenatedx

    # 或许不用降维了 1x1变通道就行

    residual = BatchNormalization()(residual)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = Conv2D(1024, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)
    x = Dropout(0.2)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    # After the last residual block, add GlobalMaxPooling to reduce the dimensions further.
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = Dense(1024, activation='swish',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='swish',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='swish',kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)


    outputs = Dense(number_classes, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs, outputs)

    return model

def DepthwiseSeparableConvWithResidual_M10D17_V1(input_shape, number_classes):
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)

    x = LeakyReLU(alpha=0.1)(x)

    # x = SeparableConv2D(128,(3,3),(1,1),padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Residual block 1
    residual = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)

    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (1, 1), padding='same')(x)

    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    # Residual block
    residual = x
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])
    residual = x
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])
    residual = x
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])


    # Residual block 2
    residual = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)

    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(256, (1, 1), padding='same')(x)

    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    sx = Add()([x, residual])


    # Residual block 3
    residual = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(sx)

    x = DepthwiseConv2D((3, 3), padding='same')(sx)

    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(512, (1, 1), padding='same')(x)

    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    ex = MaxPooling2D(pool_size=(2, 2))(sx)

    concatenatedx = Concatenate(axis=-1)([x, ex])
    concatenatedx = Dropout(0.2)(concatenatedx)

    # Residual block 3
    residual = concatenatedx

    x = DepthwiseConv2D((3, 3), padding='same')(concatenatedx)

    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(768, (1, 1), padding='same')(x)

    # x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    # Residual block 4
    residual = Conv2D(768, (3, 3), strides=(2, 2), padding='same')(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)

    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(768, (1, 1), padding='same')(x)

    # x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])


    sx = MaxPooling2D(pool_size=(2, 2))(sx)

    sx = MaxPooling2D(pool_size=(2, 2))(sx)

    concatenatedx = Concatenate(axis=-1)([x, sx])
    concatenatedx = Dropout(0.2)(concatenatedx)


    # Residual block 4
    residual = concatenatedx

    # 或许不用降维了 1x1变通道就行

    x = DepthwiseConv2D((3, 3), padding='same')(x)

    x = Activation('swish')(x)
    x = Conv2D(1024, (1, 1), padding='same')(x)

    x = Activation('swish')(x)
    x = Dropout(0.2)(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Add()([x, residual])

    # After the last residual block, add GlobalMaxPooling to reduce the dimensions further.
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Layers
    x = Dense(1024, activation='swish',kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)
    x = Dense(512, activation='swish',kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)
    x = Dense(256, activation='swish',kernel_initializer='he_normal')(x)



    outputs = Dense(number_classes, activation='sigmoid', kernel_initializer='glorot_uniform')(x)

    model = Model(inputs, outputs)

    return model



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

if __name__ == '__main__':

    # dataset_dir = r"D:\ECGData\ECGImages4C4LHighSoftLables"
    # dataset_dir = r'D:\ECGData\ECGImages4C4LHighSoftLablesColorInversion4C'
    # dataset_dir=  r'D:\ECGData\Binarization224x224'

    # dataset_dir =r'D:\ECGData\Binarization224x224Class5'
    # dataset_dir =r'D:\ECGData\ECGImages302threshold=20.0'
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

    # def load_data_from_h5(file_path):
    #     images = []
    #     labels = []
    #
    #     with h5py.File(file_path, 'r') as f:
    #         for top_group_name, top_group in f.items():
    #             for subgroup_name, subgroup in top_group.items():
    #                 for group_name, group in subgroup.items():
    #                     if 'image' in group:
    #                         img = group['image'][:]
    #                         img_attrs = {label_name: value for label_name, value in group.attrs.items() if
    #                                      label_name != "HYP"}  # Exclude 'HYP'
    #
    #                         # Assuming you want to store these attributes as a categorical array or similar
    #                         label_array = [img_attrs.get(label_name, 0.0) for label_name in sorted(img_attrs.keys())]
    #
    #                         images.append(img)
    #                         labels.append(label_array)
    #                     else:
    #                         print(
    #                             f"Warning: Group '{group_name}' in '{subgroup_name}' in '{top_group_name}' does not contain an 'image' dataset.")
    #
    #     return np.array(images), np.array(labels)


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



    # model = DepthwiseSeparableConvWithResidual_M10D17_V1(input_shape, num_classes)
    model = DepthwiseSeparableConvWithResidual_MobileNetV1(input_shape, num_classes)


    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_weights_save_load_dir = f"DSCWithResnetForECGImages{timestamp}.h5"
    # best_weights_save_load_dir = f"DSCWithResnetForECGImages20230928_220717.h5"

    model.summary()

    log_dir = "logs/fit/"  # 您可以选择其他目录
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Define the checkpoint callback
    checkpoint = ModelCheckpoint(best_weights_save_load_dir, monitor='val_accuracy', save_best_only=True,
                                 mode='max', verbose=1)

    # Define the Early Stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=200, verbose=1)


    history = model.fit(train_generator, epochs=200, validation_data=validation_generator, verbose=1,
                        callbacks=[checkpoint, early_stopping,tensorboard_callback])

    save_dir = ShowHistory(history)
    # from tensorflow.keras.utils import plot_model
    # import pydot
    # plot_file_path = os.path.join(save_dir, 'model_plot.png')
    # plot_model(model, to_file=plot_file_path, show_shapes=True, show_layer_names=True)

    model.load_weights(best_weights_save_load_dir)

    # evaluateModel_NoWrite(model, test_generator)

    evaluateModel(model, test_generator,save_dir)

    analysisShap(save_dir)

    # Grad-CAM 热图生成调用
    # 需要提供模型、要分析的图像、模型中最后一个卷积层的名称和保存目录
    last_conv_layer_name = "conv2d_8"
    # last_conv_layer_name = "  conv2d_13"

    img_array = test_images_small[0:1]  # 假设这是要分析的图像

    # 调用 make_gradcam_heatmap 函数
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, save_dir=save_dir)










