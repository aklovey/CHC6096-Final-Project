import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc, classification_report
import seaborn as sns
import pywt
from tf_keras_vis.gradcam import Gradcam


def evaluateModel(model, X_test, y_test):
    # Evaluate the model
    evaluation = model.evaluate(X_test, y_test)

    # Extract loss and accuracy
    loss = evaluation[0]
    accuracy = evaluation[1]

    # Print loss and accuracy
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # Extract additional metrics if available
    if len(evaluation) > 2:
        additional_metrics = evaluation[2:]
        print("Additional metrics:")
        for i, metric_value in enumerate(additional_metrics):
            print(f'Metric {i}: {metric_value}')

    y_pred = model.predict(X_test)
    # Convert the predictions to binary format
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Get the class labels for the true values
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred_labels)
    print("Classification Report:")
    print(report)

    # Compute the confusion matrix
    confusion = confusion_matrix(y_true, y_pred_labels)
    print("confusion_matrix: ")
    print(confusion)

    # Plot confusion matrix as an image
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion, annot=True, cmap='YlGnBu', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute F1 score for each class separately
    f1 = f1_score(y_true, y_pred_labels, average=None)
    for i, score in enumerate(f1):
        print(f"F1 score for class {i}: {score}")
    # # Compute the F1 score
    # f1 = f1_score(y_true, y_pred_labels,
    #               average='micro')  # average should be one of {None, 'micro', 'macro', 'weighted', 'samples'}
    # print("f1_score: " + str(f1))

    # Binarize the output (one-hot encoding)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
    y_pred_bin = label_binarize(y_pred_labels, classes=[0, 1, 2, 3, 4, 5])

    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute Precision-Recall curve and area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Compute micro-average Precision-Recall curve and area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 8))
    for i in range(n_classes):
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
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                                                ''.format(i, pr_auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Compute sensitivity and specificity
    sensitivity = tpr[1]
    specificity = 1 - fpr[1]
    print("Sensitivity: " + str(sensitivity))
    print("Specificity: " + str(specificity))


def accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def micro_average_precision(y_true, y_pred):
    """
    Calculate micro-average precision for multi-class classification.

    Args:
    y_true: True labels, one-hot encoded.
    y_pred: Predicted labels.

    Returns:
    Micro-average precision.
    """
    # Calculate TP and FP for each class
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)), axis=0)
    # Calculate micro-average precision
    micro_precision = tf.reduce_sum(true_positives) / (tf.reduce_sum(predicted_positives) + tf.keras.backend.epsilon())
    return micro_precision

def micro_average_recall(y_true, y_pred):
    """
    Calculate micro-average recall for multi-class classification.

    Args:
    y_true: True labels, one-hot encoded.
    y_pred: Predicted labels.

    Returns:
    Micro-average recall.
    """
    # Calculate TP and FN for each class
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)), axis=0)
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)), axis=0)
    # Calculate micro-average recall
    micro_recall = tf.reduce_sum(true_positives) / (tf.reduce_sum(possible_positives) + tf.keras.backend.epsilon())
    return micro_recall

def micro_average_f1_score(y_true, y_pred):
    """
    Calculate micro-average F1 score for multi-class classification.

    Args:
    y_true: True labels, one-hot encoded.
    y_pred: Predicted labels.

    Returns:
    Micro-average F1 score.
    """
    micro_precision = micro_average_precision(y_true, y_pred)
    micro_recall = micro_average_recall(y_true, y_pred)
    # Calculate micro-average F1 score
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + tf.keras.backend.epsilon())
    return micro_f1






def plot_training_history(history):
    """
    Plot the training and validation metrics from the training history.

    Args:
    history: A History object returned from the fit method of a keras model.
    """
    # Set up the matplotlib figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # Training and Validation Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation accuracy')
    axes[0, 0].set_title('Training and Validation Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    # Training and Validation Loss
    axes[0, 1].plot(history.history['loss'], label='Training loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation loss')
    axes[0, 1].set_title('Training and Validation Loss')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()

    # Micro Average Precision
    axes[0, 2].plot(history.history['micro_average_precision'], label='Training Micro Average Precision')
    axes[0, 2].plot(history.history['val_micro_average_precision'], label='Validation Micro Average Precision')
    axes[0, 2].set_title('Training and Validation Micro Average Precision')
    axes[0, 2].set_ylabel('Micro Average Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()

    # Micro Average Recall
    axes[1, 0].plot(history.history['micro_average_recall'], label='Training Micro Average Recall')
    axes[1, 0].plot(history.history['val_micro_average_recall'], label='Validation Micro Average Recall')
    axes[1, 0].set_title('Training and Validation Micro Average Recall')
    axes[1, 0].set_ylabel('Micro Average Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()

    # Micro Average F1 Score
    axes[1, 1].plot(history.history['micro_average_f1_score'], label='Training Micro Average F1 Score')
    axes[1, 1].plot(history.history['val_micro_average_f1_score'], label='Validation Micro Average F1 Score')
    axes[1, 1].set_title('Training and Validation Micro Average F1 Score')
    axes[1, 1].set_ylabel('Micro Average F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()

    # Remove the empty subplot (bottom right)
    fig.delaxes(axes[1, 2])

    # Adjust layout
    plt.tight_layout()
    plt.show()


def VGG_16(input_shape):
    # building the model: VGGNet 16: 16 Layers - 13 Conv layers, 5 MaxPooling Layer and 3 Fully-connected (FC) Layers
    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    # Fully connected layers
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))

    return model


# def compute_cwt(ecg_signal):
#     coefficients, _ = pywt.cwt(ecg_signal, scales, wavelet)
#     return np.abs(coefficients)


if __name__ == '__main__':
    # Read the data from the CSV files
    X_df = pd.read_csv('X_mitdb_MLII.csv')
    y_df = pd.read_csv('Y_mitdb_MLII.csv')

    # Convert the dataframes to numpy arrays
    segmented_beats = X_df.values
    segmented_labels = y_df.values
    print(segmented_beats.shape, segmented_labels.shape)

    entire_X = np.array(segmented_beats)  # float64 (110084,259)
    entire_y = np.array(segmented_labels)  # str (110084,1)

    smote = SMOTE()
    # Set your target sample sizes
    num_beats_per_class = 10000
    # Setup up samplers
    over_sampler = SMOTE(sampling_strategy={key: num_beats_per_class for key in ['L', 'R', 'V', 'A', '/']})
    under_sampler = RandomUnderSampler(sampling_strategy={key: num_beats_per_class for key in ['N']})
    # Define pipeline
    pipeline = Pipeline([('O', over_sampler), ('U', under_sampler)])
    # Apply the samplers through the pipeline
    X_resampled, y_resampled = pipeline.fit_resample(entire_X, entire_y,O__random_state=42, U__random_state=42)

    print(X_resampled.shape, y_resampled.shape, y_resampled)

    labelMap = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4, '/': 5}

    # Map the labels to integers based on labelMap
    y_resampled = np.vectorize(labelMap.get)(y_resampled)

    print(X_resampled.shape, y_resampled.shape, y_resampled)

    file_path = 'image_data.npy'

    # '''
    # This code is required for the first run
    # '''
    # data = X_resampled
    # Fs = 360
    # # After test a lot of range for 'cmorx.x-x.x' find that cmor1.5-0.5 can show both R-peak and T-peak
    # # At the same time different value of 'cmorx.x-x.x' could extract different version of features
    # scales = pywt.centrfrq('cmor1.5-0.5') * Fs / np.arange(1, 39)
    # wavelet = 'cmor1.5-0.5'
    #
    # # Initialize an empty array to store the images
    # image_data = np.empty((data.shape[0], len(scales), data.shape[1]))
    #
    # # Compute the images and store them in the image_data array
    # for i in range(data.shape[0]):
    #     image_data[i] = compute_cwt(data[i, :])
    #
    # # print(image_data.shape)
    # np.save(file_path, image_data)
    # '''
    # This code is required for the first run
    # '''

    # Load the data from the file
    loaded_image_data = np.load(file_path)
    # print(loaded_image_data.shape)

    X = loaded_image_data  # (60000, 38, 259)
    y = y_resampled  # (60000,)
    input_shape = (38, 259, 1)
    num_classes = 6

    # split the dataset into training (70% 42000), validation (15% 9000), and testing sets (15% 9000)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert the labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Reshape the input data
    X_train = X_train.reshape(-1, 38, 259, 1)
    X_val = X_val.reshape(-1, 38, 259, 1)
    X_test = X_test.reshape(-1, 38, 259, 1)

    model = VGG_16(input_shape)
    # compile the model
    # model.compile(optimizer=optimizers.Adam(learning_rate=0.0004),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=['accuracy', micro_average_precision, micro_average_recall, micro_average_f1_score])

    # model.compile(optimizer=optimizers.Adam(learning_rate=0.0004),
    #               loss=tf.keras.losses.CategoricalCrossentropy(),
    #               metrics=['accuracy'])

    model.summary()

    best_weights_save_load_dir = "CWT2Donvgg16.h5"

    # # Define the checkpoint callback
    # checkpoint = ModelCheckpoint(best_weights_save_load_dir, monitor='val_accuracy', save_best_only=True,
    #                              mode='max', verbose=1)
    #
    # # Define the Early Stopping callback
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)
    #
    # # Train the model with both callbacks
    # history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1, batch_size=64,
    #                     callbacks=[checkpoint, early_stopping])
    #
    # plot_training_history(history)

    model.load_weights(best_weights_save_load_dir)

    # evaluateModel(model, X_test, y_test)

    # def apply_grad_cam(model, X_test, y_test, layer_name='conv2d_9', class_index=0):
    #     from tf_explain.core.grad_cam import GradCAM
    #     import matplotlib.pyplot as plt
    #     import numpy as np  # Ensure numpy is imported
    #
    #     i = class_index
    #
    #     grad_cam = GradCAM()
    #     image = X_test[i]  # Get the image from the test set corresponding to the class index
    #     image = np.expand_dims(image, axis=0)  # Add batch dimension
    #     label_index = np.array([y_test[i].argmax()])  # Ensure label_index is in a batch
    #     data = (image, label_index)
    #
    #     cam = grad_cam.explain(data, model, class_index=label_index[0], layer_name=layer_name)  # Explain the image
    #
    #     print("CAM output shape:", cam.shape)  # Debug print to check the shape
    #
    #     fig, ax = plt.subplots(1, 2, figsize=(20, 8))  # Set up a figure with two subplots
    #
    #     # Plot the original image
    #     ax[0].imshow(image[0, :, :, 0].T, cmap='gray', aspect='auto')
    #     ax[0].set_title(f'Original Image for Class {class_index}')
    #     ax[0].axis('off')  # Hide the axes
    #
    #     # Plot the Grad-CAM heatmap
    #     ax[1].imshow(cam[0, :, :, 0].T, cmap='hot', aspect='auto')
    #     ax[1].set_title(f"Grad-CAM for Class {class_index}")
    #     ax[1].axis('off')  # Hide the axes
    #
    #     plt.colorbar(ax[1].imshow(cam[0, :, :, 0].T, cmap='hot', aspect='auto'), ax=ax[1])  # Add colorbar to heatmap
    #     plt.show()


    def apply_grad_cam(model, X_test, y_test, layer_name='conv2d_9', class_index=0):
        from tf_explain.core.grad_cam import GradCAM
        import matplotlib.pyplot as plt
        import numpy as np  # Ensure numpy is imported

        i = class_index

        grad_cam = GradCAM()
        image = X_test[i]  # Get the first image from the test set
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        label_index = np.array([y_test[i].argmax()])  # Ensure label_index is in a batch
        data = (image, label_index)

        cam = grad_cam.explain(data, model, class_index=label_index[0], layer_name=layer_name)  # Explain the image

        print("CAM output shape:", cam.shape)  # Debug print to check the shape

        plt.figure(figsize=(10, 8))
        # Assuming the Grad-CAM output matches the image spatial dimensions
        if cam.shape[-1] == 1:
            plt.imshow(cam[0, :, :, 0].T, cmap='hot', aspect='auto')  # Ensure correct access and orientation
        else:
            plt.imshow(cam[0].T, cmap='hot', aspect='auto')  # Ensure correct access and orientation
        plt.colorbar()
        plt.title(f"Grad-CAM for CWT images ")
        plt.show()


    apply_grad_cam(model, X_test,y_test,class_index=0)
    apply_grad_cam(model, X_test, y_test, class_index=1)
    apply_grad_cam(model, X_test, y_test, class_index=2)
    # apply_grad_cam(model, X_test, y_test, class_index=3)
    # apply_grad_cam(model, X_test, y_test, class_index=4)
    # apply_grad_cam(model, X_test, y_test, class_index=5)

    # from tf_keras_vis.gradcam import GradcamPlusPlus  # 确保使用正确的 GradCAM 实现
    #
    #
    # def apply_grad_cam(model, X_test, layer_name='conv2d_9'):
    #     # 检查数据形状
    #     if X_test.size == 0:
    #         raise ValueError("X_test cannot be empty.")
    #
    #     # 创建 GradCAM 实例
    #     grad_cam = GradcamPlusPlus(model, model_modifier=None, clone=False)
    #
    #     # 取第一张图片并调整形状以匹配模型输入要求
    #     image = X_test[0]
    #     if image.shape != (38, 259, 1):
    #         image = np.expand_dims(image, axis=0)  # 确保形状为 (1, 38, 259, 1)
    #
    #     # 生成 Grad-CAM 并进行显示
    #     cam = grad_cam(model, seed_input=image, penultimate_layer=layer_name)
    #     plt.figure(figsize=(10, 2))
    #     if cam.shape[-1] == 1:
    #         plt.imshow(cam[0, :, :, 0].T, cmap='hot', aspect='auto')  # 调整数组索引以适配数据形状
    #     else:
    #         plt.imshow(cam[0].T, cmap='hot', aspect='auto')  # 调整数组索引以适配数据形状
    #     plt.colorbar()
    #     plt.title("Grad-CAM")
    #     plt.show()
    #
    #
    # # 在函数外部调用
    # apply_grad_cam(model, X_test)




