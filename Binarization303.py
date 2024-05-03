# Wyatt
import os
import io
import cv2
import h5py
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import random
# Wyatt
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


    def generate_ecg_image(self, record, segment_start):
        lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        twelve_leads_full = {lead_names[i]: record.p_signal[:, i] for i in range(12)}

        # Standardizing each lead separately
        standardized_twelve_leads = {
            lead: (twelve_leads_full[lead] - np.mean(twelve_leads_full[lead])) / np.std(twelve_leads_full[lead]) for
            lead in lead_names}

        segment_end = segment_start + 250
        segments = {lead: signal[segment_start:segment_end] for lead, signal in standardized_twelve_leads.items()}

        fig, axs = plt.subplots(4, 1, figsize=(7, 7))

        for i in range(3):
            combined_x_data = []
            combined_y_data = []
            for j in range(4):
                lead = lead_names[(j * 3) + i]
                x_data = list(range(len(combined_x_data), len(combined_x_data) + 250))
                y_data = segments[lead]
                combined_x_data.extend(x_data)
                combined_y_data.extend(y_data)

            axs[i].plot(combined_x_data, combined_y_data, color='black')
            axs[i].axis('off')
            axs[i].set_ylim([-3, 3])
            axs[i].set_position([-0.045, i * 1 / 4, 1.09, 1 / 4])

        # 修改第四行以固定显示II型导联的数据
        x_data_full = list(range(0, 1000))
        y_data_full = standardized_twelve_leads['II'][0:1000]  # 固定使用II型导联
        axs[3].plot(x_data_full, y_data_full, color='black')
        axs[3].axis('off')
        axs[3].set_ylim([-3, 3])
        axs[3].set_position([-0.045, 3 * 1 / 4, 1.09, 1 / 4])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=32)
        buf.seek(0)
        plt.close()

        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        return img

    def save_ecg_image_to_hdf5(self, img, hdf5_file,  superclass_values, filename, segment_idx):
        # 用filename和segment_idx作为组名
        group_name = f"{filename}_{segment_idx}"
        if group_name not in hdf5_file:
            group = hdf5_file.create_group(group_name)
        else:
            group = hdf5_file[group_name]

        # 检查"image"数据集是否已经存在
        if "image" not in group:
            group.create_dataset("image", data=img)

        # 将Soft Labels存储到该组中
        for label_name, value in superclass_values.items():
            group.attrs[label_name] = value

    def process_and_save_images(self, Y, save_path, display_num=5, seed=42):
        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)  # If you're using numpy in other parts of your code

        # Use the keys from the superclass dictionary to initialize image_counter
        image_counter = {partition: {superclass: 0 for superclass in self.superclass.keys()}
                         for partition in ['train', 'val', 'test']}

        # Create or open HDF5 files for each partition
        hdf5_files = {
            'train': h5py.File(os.path.join(save_path, 'train.h5'), 'a'),
            'val': h5py.File(os.path.join(save_path, 'val.h5'), 'a'),
            'test': h5py.File(os.path.join(save_path, 'test.h5'), 'a')
        }

        segments_starts = [0, 250, 500, 750]
        display_indices = random.sample(range(len(Y)), min(len(Y), display_num))  # Determine random indices to display

        for idx, (index, row) in enumerate(Y.iterrows()):  # Ensure proper enumeration
            filename = row['filename_lr']
            superclass_values = row['superclass_values']

            # Skip the row if all superclass_values are zero
            if all(value == 0 for value in superclass_values.values()):
                continue  # Skip processing and saving this sample

            # Load the entire 10-second record
            full_record = data_loader.load_single_record(filename, 0, 1000)

            # Determine the dataset partition based on the strat_fold value
            if row['strat_fold'] in [1, 2, 3, 4, 5, 6, 7, 8]:
                partition = 'train'
            elif row['strat_fold'] == 9:
                partition = 'val'
            else:
                partition = 'test'

            for segment_start in segments_starts:
                # Generate the image for the segment
                img = self.generate_ecg_image(full_record, segment_start)
                segment_idx = segments_starts.index(segment_start)

                # Save the generated image to the HDF5 file
                self.save_ecg_image_to_hdf5(img, hdf5_files[partition], superclass_values, filename, segment_idx)

                # Update the image counter for the respective partition and superclass
                for superclass in self.superclass.keys():  # Use superclass keys from initialization
                    if superclass_values.get(superclass, 0) > 0:
                        image_counter[partition][superclass] += 1

                # Randomly display the image
                if idx in display_indices:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img, cmap='gray')
                    plt.title(f"Sample {idx}: {filename}")
                    plt.axis('off')
                    plt.show()

        # Close the HDF5 files
        for hdf5_file in hdf5_files.values():
            hdf5_file.close()

        return image_counter

path = r'D:/ECGData/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
superclass = {'CD': 0, 'HYP': 0, 'MI': 0, 'NORM': 0, 'STTC': 0}


data_loader = ECGDataLoader(path)
diagnostic_mapping = data_loader.scp_statements
Y = data_loader.load_Y_data()
print(Y.shape)
Y['scp_codes_dict'] = Y['scp_codes'].apply(eval)
processor = ECGImageProcessor(diagnostic_mapping, superclass)
Y['superclass_values'] = Y['scp_codes_dict'].apply(lambda x: processor.map_to_superclass(x, threshold=40.0))

Y = Y.dropna(subset=['superclass_values'])
print(Y.shape)
SavePath = r'D:\ECGData\ECGImages303threshold=40.0'

# Check and create the directory if it doesn't exist
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
image_counter = processor.process_and_save_images(Y, SavePath)

for partition, counts in image_counter.items():
    print(f"{partition.upper()}:")
    for cls, count in counts.items():
        print(f"  {cls}: {count} images")