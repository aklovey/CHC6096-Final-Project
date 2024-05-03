import os

import numpy as np
import time
import tflite_runtime.interpreter as tflite
from ecgdetectors import Detectors

# read the ECG data
# The total amount of data is freq * duration
def get_ecg(freq, duration):
    ecgs = []
    address = 0x48
    A0 = 0x40
    A1 = 0x41
    A2 = 0x42
    A3 = 0x43
    bus = smbus.SMBus(1)
    bus.write_byte(address, A2)
    for i in range(int(duration * freq)):
        value = bus.read_byte(address)
        normalized_value = value / 255.0  # Normalize the value
        ecgs.append(normalized_value)
        time.sleep(1.0 / freq)
    return ecgs


if __name__ == '__main__':
    f = 360
    detectors = Detectors(sampling_frequency=f)

    ecg_signals = get_ecg(360, 10)

    window_size = 259
    segmented_beats = []

    signal_normalized = (ecg_signals - np.min(ecg_signals)) / (np.max(ecg_signals) - np.min(ecg_signals))

    ecg_signal = signal_normalized

    # Detect R peaks
    r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg=ecg_signal.transpose(), MWA_name="cumulative")
    # r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg=record.p_signal.transpose())

    for r_peak in r_peaks:
        # Ensuring the window stays within the array bounds
        start_index = max(r_peak - 89, 0)  # One less before the peak
        end_index = min(r_peak + 170, len(ecg_signal))  # Keeping 170 after the peak

        # Segment the signal using R peak location and the window size
        segment = ecg_signal[start_index:end_index]
        # convert shape (259,1) to (259,)
        segment = segment.flatten()

        # If the segment size is less than window size, pad it with zeroes
        if len(segment) < window_size:
            pad = np.zeros(window_size - len(segment))
            segment = np.append(segment, pad)

        # make sure 259 samples per windows
        assert len(segment) == window_size, f"Segment length {len(segment)} != window size {window_size}"

        segmented_beats.append(segment)


    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="model.tflite")  # <--- Use tflite-runtime
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Initialize an empty list to store all predicted labels
    all_predicted_labels = []

    # Initialize variables to calculate heart rate
    total_time_in_seconds = 10  # You collect data for 10 seconds
    num_of_beats = 0  # Initialize the number of detected beats

    for segment in segmented_beats:
        # Increment the number of detected beats
        num_of_beats += 1
        # Reshape the segment for model input
        reshaped_segment = np.reshape(segment, (1, -1, 1)).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], reshaped_segment)

        # Invoke inference
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        labelMap = ['N', 'L', 'R', 'V', 'A', '/']
        # Translate output into label
        output_probabilities = output_data
        max_index = np.argmax(output_probabilities)
        predicted_label = labelMap[max_index]

        # Append the label to the list
        all_predicted_labels.append(predicted_label)

    # Calculate heart rate
    heart_rate_bpm = (num_of_beats / total_time_in_seconds) * 60
    print(f"Heart Rate: {heart_rate_bpm:.2f} BPM")

    # Now you have all labels in all_predicted_labels
    print("Predicted labels for all beats:", all_predicted_labels)

    from collections import Counter

    # Count the occurrences of each label
    label_counts = Counter(all_predicted_labels)

    # Initialize an empty string to accumulate the health conditions
    health_conditions_summary = ""

    # Check if all heartbeats are normal
    if label_counts.get('N', 0) == len(all_predicted_labels):
        print("The heartbeat appears to be normal. You are healthy.")
    else:
        # Loop through the unique labels and their counts
        for label, count in label_counts.items():
            if label == 'N':
                health_condition = "The heartbeat appears to be normal."
            elif label == 'L':
                health_condition = "Left bundle branch block beat detected."
            elif label == 'R':
                health_condition = "Right bundle branch block beat detected."
            elif label == 'V':
                health_condition = "Premature ventricular contraction detected."
            elif label == 'A':
                health_condition = "Atrial premature contraction detected."
            elif label == '/':
                health_condition = "Paced beat detected."

            # Add the health condition and its count to the summary
            health_conditions_summary += f"{health_condition} Occurrences: {count}\n"

        print(health_conditions_summary)

