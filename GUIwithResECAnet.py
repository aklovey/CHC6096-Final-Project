# Wyatt
import math
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *

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
        # print(k)
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


    labelMap = {'N': 0, 'L': 1, 'R': 2, 'V': 3, 'A': 4, '/': 5}

    num_classes = 6
    input_shape = (259, 1)


    import os


    def save_sample_data(X, y, num_samples=5):
        category_descriptions = {
            'N': 'Non-ectopic_peaks',
            'L': 'Left_bundle_branch_block_beat',
            'R': 'Right_bundle_branch_block_beat',
            'V': 'Premature_ventricular_contraction',
            'A': 'Atrial_premature_contraction',
            '/': 'Paced_beat'
        }

        output_dir = "GUI1Dtestsamples"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for category, description in category_descriptions.items():
            category_indices = np.where(y == labelMap[category])[0]
            category_samples = np.random.choice(category_indices, num_samples, replace=False)

            category_dir = os.path.join(output_dir, description)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

            for i, idx in enumerate(category_samples):
                sample_path = os.path.join(category_dir, f"sample_{i + 1}.csv")
                np.savetxt(sample_path, X[idx].reshape(-1, 1), delimiter=",")


    # Assuming X_test and y_test are your testing data sets
    # save_sample_data(X_test, y_test)

    import streamlit as st
    import numpy as np
    import glob


    def load_model_weights():
        model = ECANet(input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(r"C:\Users\Wyatt\PycharmProjects\ECGgit1\ECG1DonMITBIH.h5")
        return model


    model = load_model_weights()

    import glob
    import matplotlib.pyplot as plt


    def predict_ecg(data):
        data = data.reshape(1, -1, 1)
        prediction = model.predict(data)
        return prediction


    st.title('ECG Sample Disease Diagnosis GUI')

    category_descriptions = {
        'N': 'Non-ectopic_peaks',
        'L': 'Left_bundle_branch_block_beat',
        'R': 'Right_bundle_branch_block_beat',
        'V': 'Premature_ventricular_contraction',
        'A': 'Atrial_premature_contraction',
        '/': 'Paced_beat'
    }

    selected_description = st.selectbox("Select Category:", list(category_descriptions.values()))

    sample_files = glob.glob(f"GUI1Dtestsamples/{selected_description}/*.csv")

    selected_file = st.selectbox("Choose a sample file:", sample_files)


    if st.button('Load Sample'):
        try:
            data = np.loadtxt(selected_file, delimiter=",")
            st.session_state['data'] = data

            # Plot the data
            plt.figure(figsize=(10, 4))
            plt.plot(data, label='ECG Data')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.title('ECG Sample Data')
            plt.legend()
            st.pyplot(plt)
            st.success('Sample successfully loaded.')
            st.session_state['data_loaded'] = True
        except Exception as e:
            st.error('Failed to load the sample: {}'.format(e))
            st.session_state['data_loaded'] = False

    if st.button('Predict') and st.session_state.get('data_loaded', False):
        try:
            prediction = predict_ecg(st.session_state['data'])
            max_prob_index = np.argmax(prediction[0])
            max_category = list(category_descriptions.keys())[max_prob_index]
            health_message = "Your heart appears healthy." if max_category == 'N' else "Potential issue: {}".format(
                category_descriptions[max_category])

            st.write(f'Prediction: {prediction}')
            st.markdown("### Probability per category:")
            for key, value in zip(category_descriptions.values(), prediction[0]):
                st.markdown(f"**{key}:**       {value:.6f}")
            st.markdown("## Diagnostic results:", unsafe_allow_html=True)
            st.markdown(f"### {health_message}", unsafe_allow_html=True)
        except Exception as e:
            st.error('Failed to make a prediction: {}'.format(e))







