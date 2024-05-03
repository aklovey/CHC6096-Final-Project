# Wyatt
import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import wfdb
import streamlit as st
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
import seaborn as sns
from sklearn.metrics import classification_report
class ECA_module(Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECA_module, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b
        self.conv = None

    def build(self, input_shape):
        N, L, C = input_shape
        t = int(abs((math.log(C, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        self.conv = tf.keras.layers.Conv1D(1, k, padding='same', use_bias=False)
        print(k)
        super(ECA_module, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        N, L, C = x.shape
        y = tf.reduce_mean(x, axis=1, keepdims=True)

        y = tf.keras.layers.Reshape((1, C))(y)

        y = self.conv(y)
        y = tf.keras.activations.sigmoid(y)
        return x * y

    def get_config(self):
        config = super(ECA_module, self).get_config()
        config.update({
            'gamma': self.gamma,
            'b': self.b
        })
        return config


def Layer(layer_in, Filter):
    # x = ZeroPadding1D()(layer_in)
    x = Conv1D(Filter, kernel_size=7)(layer_in)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def residual_block(layer_in, n_filters):
    # Shortcut layer
    shortcut = layer_in

    # If n_filters is 128, update the shortcut
    if n_filters == 128:
        shortcut = Conv1D(filters=n_filters, kernel_size=1)(shortcut)

    # Conv1
    x = Conv1D(filters=n_filters, kernel_size=1)(layer_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Conv2
    conv2 = Conv1D(filters=n_filters, kernel_size=1)(x)
    x = layers.BatchNormalization()(conv2)
    x = layers.Activation('relu')(x)

    # Conv3
    conv3 = Conv1D(filters=n_filters, kernel_size=1)(x)
    x = layers.BatchNormalization()(conv3)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    x = ECA_module()(x)

    return x




def ECANet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=15, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding1D(padding=10)(x)
    maxp1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    r1 = residual_block(maxp1, 64)


    x = Layer(r1, 64)
    x = ZeroPadding1D()(x)
    maxp2 = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    r2 = residual_block(maxp2, 64)


    x = Layer(r2, 128)
    x = ZeroPadding1D()(x)
    maxp3 = MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    r3 = residual_block(maxp3, 128)

    x = Layer(r3, 256)
    x = ZeroPadding1D(padding=3)(x)
    x = Concatenate(axis=-1)([r3, x])

    # Global Average pooling and the Fully connected Layer
    x = GlobalMaxPooling1D(keepdims=False)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    # output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

if __name__ == '__main__':
    acceptable_annotations = ['N', 'L', 'R', 'V', 'A', '/']
    # N: Non-ectopic peaks
    # L: Left bundle branch block beat
    # R: Right bundle branch block beat
    # V: Premature ventricular contraction
    # A: Atrial premature contraction
    # /: Paced beat

    # Read the data from the CSV files
    X_df = pd.read_csv('X_mitdb_MLII.csv')
    y_df = pd.read_csv('Y_mitdb_MLII.csv')

    # Convert the dataframes to numpy arrays
    segmented_beats = X_df.values
    segmented_labels = y_df.values

    # # Plot initial distribution
    # plot_class_distribution(segmented_labels, "Initial Class Distribution")

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

    # # Plot resampled distribution
    # plot_class_distribution(y_resampled, "Resampled Class Distribution")

    print(X_resampled.shape, y_resampled.shape, y_resampled)
    #
    labelMap = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4, '/': 5}
    #
    # Map the labels to integers based on labelMap
    y_resampled = np.vectorize(labelMap.get)(y_resampled)

    print(X_resampled.shape, y_resampled.shape, y_resampled)

    X = X_resampled  # (60000,259)
    y = y_resampled  # (60000,)
    num_classes = 6
    input_shape = (259, 1)
    #
    # split the dataset into training (70% 42000), validation (15% 9000), and testing sets (15% 9000)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert the labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    # y_test = to_categorical(y_test, num_classes=num_classes)

    # Reshape the input data
    X_train = X_train.reshape(-1, 259, 1)
    X_val = X_val.reshape(-1, 259, 1)
    X_test = X_test.reshape(-1, 259, 1)



    import streamlit as st
    import numpy as np
    import glob


    def load_model_weights():
        model = ECANet(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(r"C:\Users\Wyatt\PycharmProjects\ECGgit1\ECG1DonMITBIH.h5")
        return model


    model = load_model_weights()




    import shap
    import lime
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from matplotlib import pyplot as plt
    from tf_explain.core.grad_cam import GradCAM




    def apply_shap(model, X_test):
        X_test = X_test.reshape(-1, 259)
        num_features = X_test.shape[1]
        feature_names = [f"Feature{i + 1}" for i in range(num_features)]  # Create default feature names

        numoftestsample = 15
        # Initialize the explainer with a background dataset
        explainer = shap.KernelExplainer(model, X_test[:numoftestsample])
        x_test_sample = shap.sample(X_test, numoftestsample)
        # Compute SHAP values for the first sample
        shap_values = explainer.shap_values(x_test_sample)

        # Summary plot for the first 10 samples for an overview
        plt.figure(figsize=(10, 8), dpi=600)
        plt.title(f'SHAP Summary Plot On {numoftestsample} Test Samples', fontsize=16)
        shap.summary_plot(shap_values, x_test_sample, feature_names=feature_names)
        plt.show()


    # # Example usage
    # apply_shap(model, X_test)

    # LIME
    from lime.lime_tabular import LimeTabularExplainer


    def apply_lime(model, X_test):
        num_features = X_test.shape[1]
        feature_names = [f"Feature{i + 1}" for i in range(num_features)]  # Create default feature names
        explainer = LimeTabularExplainer(X_test, mode='classification', feature_names=feature_names)
        i = 0  # Example index to explain
        explanation = explainer.explain_instance(X_test[i], model.predict, num_features=10)
        explanation.show_in_notebook(show_table=True)


    # apply_lime(model, X_test)


    from tf_explain.core.grad_cam import GradCAM


    def apply_grad_cam(model, X_test, y_test,i, layer_name='conv1d_13'):
        grad_cam = GradCAM()
        image = np.expand_dims(X_test[i], axis=0)  # Add batch dimension
        label_index = np.array([y_test[i].argmax()])  # Ensure label_index is in a batch

        # Data structured for a batch of one
        data = (image, label_index)
        try:
            cam = grad_cam.explain(data, model, class_index=label_index[0], layer_name=layer_name)
            plt.figure(figsize=(10, 2))
            if cam.ndim == 3 and cam.shape[-1] == 1:
                plt.imshow(cam[0, :, :, 0].T, cmap='hot', aspect='auto')  # Adjust for batch dimension
            else:
                plt.imshow(cam[0].T, cmap='hot', aspect='auto')  # Adjust for batch dimension
            plt.colorbar()
            plt.title(f"Grad-CAM for class  {label_index[0]}")
            plt.show()
        except Exception as e:
            print(f"An error occurred: {e}")


    # Apply Explainability Methods
    apply_grad_cam(model, X_test, y_test,0)
    apply_grad_cam(model, X_test, y_test,1)
    apply_grad_cam(model, X_test, y_test,2)
    apply_grad_cam(model, X_test, y_test,3)
    apply_grad_cam(model, X_test, y_test,4)
    apply_grad_cam(model, X_test, y_test,5)
    apply_grad_cam(model, X_test, y_test,6)
    apply_grad_cam(model, X_test, y_test,7)
    apply_grad_cam(model, X_test, y_test,8)


