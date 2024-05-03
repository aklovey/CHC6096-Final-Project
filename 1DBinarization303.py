# Wyatt
# Wyatt
import os
import io
import random
import time

import cv2
import h5py
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import pywt

from scipy.signal import iirnotch, filtfilt
from ecgdetectors import Detectors
from tqdm import tqdm


def moving_average_filter(signal, window_size):
    """Apply moving average filter to the signal."""
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def notch_filter(signal, f0, Q, fs):
    """Design a notch filter and apply it to the signal."""
    b, a = iirnotch(f0, Q, fs)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def process_ecg_signal(ecg_lead, ma_window_size=10, fs=500, f0=50.0, Q=30.0):
    """Process a single lead of ECG data."""
    ma_filtered = moving_average_filter(ecg_lead, ma_window_size)
    notch_filtered = notch_filter(ma_filtered, f0=f0, Q=Q, fs=fs)
    return notch_filtered


# def detect_r_peaks(signal):
#     detectors = Detectors(sampling_frequency=500)
#     # Here, you'd use your detectors.pan_tompkins_detector function.
#     # For demonstration purposes, I'll provide a dummy function.
#     return detectors.pan_tompkins_detector(unfiltered_ecg=signal.transpose())

def segment_data_around_rpeak(ecg_data, r_peak_idx, left=150, right=350):
    start_idx = max(r_peak_idx - left, 0)
    end_idx = min(r_peak_idx + right, ecg_data.shape[1])
    segment = ecg_data[:, start_idx:end_idx]
    if segment.shape[1] < left + right:
        pad_width = ((0, 0), (0, left + right - segment.shape[1]))
        segment = np.pad(segment, pad_width, mode='constant')
    return segment




def get_diff(ecg):
    ecg_diff = np.diff(ecg, prepend=ecg[0])
    return ecg_diff


def checkR(ecg, sampling_rate):
    ecg_diff = get_diff(ecg)
    max_val = np.max(ecg)
    min_val = np.min(ecg)
    threshold_val = (max_val - min_val) * 0.7 + min_val

    index = []
    for i in range(1, len(ecg) - 1):
        # 使用差分信号来检测R波
        if ecg_diff[i - 1] > 0 and ecg_diff[i] < 0 and ecg[i] > threshold_val:
            if not index or (
                    i - index[-1] >= 60.0 / 160.0 * sampling_rate and i - index[-1] <= 60.0 / 60.0 * sampling_rate):
                index.append(i)

    return np.array(index)


# def get_diff(ecg):
#     ecg_diff = np.diff(ecg, prepend=ecg[0])
#     return ecg_diff
#
# def checkR(ecg, sampling_rate):
#     ecg = ecg.flatten()  # 确保ecg是一维数组
#     ecg_diff = get_diff(ecg)
#     max_val = np.max(ecg)
#     min_val = np.min(ecg)
#     threshold_val = (max_val - min_val) * 0.7 + min_val
#     index = []
#     for i in range(1, len(ecg) - 1):
#         if ecg[i] == np.max(ecg[i-1:i+2]) and ecg[i] > threshold_val:
#             if index and i - index[-1] >= 60.0 / 160.0 * sampling_rate and i - index[-1] <= 60.0 / 60.0 * sampling_rate:
#                 index.append(i)
#             elif not index:
#                 index.append(i)
#     return np.array(index)





class ECGDataLoader:
    def __init__(self, data_path):
        self.path = data_path
        self.scp_statements = {
            'NDT': 'STTC', 'NORM': 'NORM', 'NST_': 'STTC', 'DIG': 'STTC', 'LNGQT': 'STTC',
            'IMI': 'MI', 'ASMI': 'MI', 'LVH': 'HYP', 'LAFB': 'CD', 'ISC_': 'STTC',
            'IRBBB': 'CD', '1AVB': 'CD', 'IVCD': 'CD', 'ISCAL': 'STTC', 'CRBBB': 'CD',
            'CLBBB': 'CD', 'ILMI': 'MI', 'LAO/LAE': 'HYP', 'AMI': 'MI', 'ALMI': 'MI',
            'ISCIN': 'STTC', 'INJAS': 'MI', 'LMI': 'MI', 'ISCIL': 'STTC', 'LPFB': 'CD',
            'ISCAS': 'STTC', 'INJAL': 'MI', 'ISCLA': 'STTC', 'RVH': 'HYP', 'ANEUR': 'STTC',
            'RAO/RAE': 'HYP', 'EL': 'STTC', 'WPW': 'CD', 'ILBBB': 'CD', 'IPLMI': 'MI',
            'ISCAN': 'STTC', 'IPMI': 'MI', 'SEHYP': 'HYP', 'INJIN': 'MI', 'INJLA': 'MI',
            'PMI': 'MI', '3AVB': 'CD', 'INJIL': 'MI', '2AVB': 'CD'
        }

    def load_single_record(self, filename, sampfrom, sampto):
        return wfdb.rdrecord(os.path.join(self.path, filename), sampfrom=sampfrom, sampto=sampto)

    def load_Y_data(self):
        Y = pd.read_csv(os.path.join(self.path, 'ptbxl_database.csv'), index_col='ecg_id')
        return Y


class ECGImageProcessor:
    def __init__(self, scp_mapping, superclass):
        self.scp_mapping = scp_mapping
        self.superclass = superclass
        self.lead_order_for_fourth_row = list(range(0, 12))

    def map_to_superclass(self, scp_code, threshold=50.0):
        """Convert the given scp_code into its superclass by first finding the maximum value for each superclass
        and then binarizing the result based on a specified threshold.
        Values >= threshold are set to 1, indicating significant presence, while values < threshold are set to 0, indicating absence or insignificance.
        """

        # Initialize the superclass_values dictionary with desired classes and zero values
        superclass_dict = {'CD': 0, 'HYP': 0, 'MI': 0, 'NORM': 0, 'STTC': 0}

        # First pass: Iterate through the scp_code dictionary and update to the maximum value
        for key, value in scp_code.items():
            superclass = self.scp_mapping.get(key)
            if superclass:
                # Only update if the new value is greater than the current value
                if value > superclass_dict[superclass]:
                    superclass_dict[superclass] = value

        # Second pass: Binarize the superclass_dict based on the specified threshold
        for key in superclass_dict:
            superclass_dict[key] = 1 if superclass_dict[key] >= threshold else 0

        return superclass_dict
    # def map_to_superclass(self, scp_code):
    #     """Convert the given scp_code into its superclass and accumulate the values."""
    #
    #     # Initialize the superclass_values dictionary with desired classes and zero values
    #     superclass_dict = {'CD': 0, 'HYP': 0, 'MI': 0, 'NORM': 0, 'STTC': 0}
    #
    #     # Flag to check if at least one scp_code was mapped to a superclass
    #     mapped_flag = False
    #
    #     # Iterate through the scp_code dictionary and map to the superclass
    #     for key, value in scp_code.items():
    #         superclass = self.scp_mapping.get(key)
    #         if superclass:
    #             superclass_dict[superclass] += value
    #             mapped_flag = True
    #
    #     # If no scp_code was mapped to a superclass, return None
    #     if not mapped_flag:
    #         return None
    #
    #     # Set values greater than 100.0 to 100.0
    #     for key in superclass_dict:
    #         if superclass_dict[key] > 100.0:
    #             superclass_dict[key] = 100.0
    #
    #     # If more than one superclass has non-zero values, return None
    #     # non_zero_values = sum(1 for value in superclass_dict.values() if value != 0)
    #     # if non_zero_values > 1:
    #     #     return None
    #
    #     # # If the maximum value in the superclass is below 100, return None
    #     # if max(superclass_dict.values()) < 100:
    #     #     return None
    #
    #     return superclass_dict

    def segment_data_around_rpeak(signal, r_peaks, left=150, right=350):
        segments = []
        for r_peak in r_peaks:
            start_index = max(r_peak - left, 0)
            end_index = min(r_peak + right, len(signal))
            segment = signal[start_index:end_index]
            if len(segment) < (left + right):
                pad = np.zeros((left + right) - len(segment))
                segment = np.append(segment, pad)
            segments.append(segment)
        return segments

    def save_ecg_signal_to_hdf5(self, signal, hdf5_file, superclass_values, filename, segment_idx):
        group_name = f"{filename}_{segment_idx}"
        if group_name not in hdf5_file:
            group = hdf5_file.create_group(group_name)
        else:
            group = hdf5_file[group_name]

        if "signal" not in group:
            group.create_dataset("signal", data=signal)

        for label_name, value in superclass_values.items():
            group.attrs[label_name] = value

    def process_and_save_data(self, Y, save_path):
        hdf5_files = {
            'train': h5py.File(os.path.join(save_path, 'train.h5'), 'a'),
            'val': h5py.File(os.path.join(save_path, 'val.h5'), 'a'),
            'test': h5py.File(os.path.join(save_path, 'test.h5'), 'a')
        }
        start_time = time.time()
        sample_count = 0
        for idx, row in tqdm(Y.iterrows(), total=len(Y), desc="Processing records"):
            filename = row['filename_hr']
            superclass_values = row['superclass_values']

            if row['strat_fold'] in [1, 2, 3, 4, 5, 6, 7, 8]:
                partition = 'train'
            elif row['strat_fold'] == 9:
                partition = 'val'
            else:
                partition = 'test'

            full_record = data_loader.load_single_record(filename, 0, 5000).p_signal.transpose()

            # 处理ECG信号
            processed_data = np.zeros((12, 5000))
            for lead_idx in range(12):
                processed_data[lead_idx, :] = process_ecg_signal(full_record[lead_idx, :])

            # 在II导联上进行R波检测
            ii_lead = processed_data[1, :]  # II导联的索引是1
            r_peaks = checkR(ii_lead, sampling_rate=500)


            # 确保选择的R波位置允许我们切割出长度为500的信号
            valid_r_peaks = [r_peak for r_peak in r_peaks if 150 <= r_peak <= 4650]

            if len(valid_r_peaks) > 0:
                # 随机选择一个R波位置
                selected_r_peak = random.choice(valid_r_peaks)

                sample_count += 1

                segment = processed_data[:, selected_r_peak - 150:selected_r_peak + 350]
                if segment.shape[1] == 500:
                    self.save_ecg_signal_to_hdf5(segment, hdf5_files[partition], superclass_values, filename,
                                                 selected_r_peak)
                if sample_count % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Processed {sample_count} samples in {elapsed_time:.2f} seconds.")

        for hdf5_file in hdf5_files.values():
            hdf5_file.close()





if __name__ == '__main__':
    path = r'D:/ECGData/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    save_path = r'D:\ECGData\ECG1D304'
    superclass = {'CD': 0, 'HYP': 0, 'MI': 0, 'NORM': 0, 'STTC': 0}
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_loader = ECGDataLoader(path)
    Y = data_loader.load_Y_data()

    Y['scp_codes_dict'] = Y['scp_codes'].apply(eval)
    processor = ECGImageProcessor(data_loader.scp_statements, superclass)
    Y['superclass_values'] = Y['scp_codes_dict'].apply(lambda x: processor.map_to_superclass(x))
    Y = Y.dropna(subset=['superclass_values'])

    processor.process_and_save_data(Y, save_path)



