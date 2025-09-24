from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- FASCICLE DICTIONARY --------------------
bundles_names = {
    0: 'AF_L', 1: 'AF_R', 2: 'CC_Oc', 3: 'CC_Fr_1', 4: 'CC_Fr_2',
    5: 'CC_Pa', 6: 'CC_Pr_Po', 7: 'CG_L', 8: 'CG_R', 9: 'FAT_L',
    10: 'FAT_R', 11: 'FPT_L', 12: 'FPT_R',
    15: 'IFOF_L', 16: 'IFOF_R', 17: 'ILF_L', 18: 'ILF_R', 19: 'MCP',
    20: 'MdLF_L', 21: 'MdLF_R', 22: 'OR_ML_L', 23: 'OR_ML_R',
    24: 'POPT_L', 25: 'POPT_R', 26: 'PYT_L', 27: 'PYT_R',
    28: 'SLF_L', 29: 'SLF_R', 30: 'UF_L', 31: 'UF_R'
}

# -------------------- PARAMETERS --------------------
train_path = r'Tractoinferno/trainset'
test_path = r'Tractoinferno/testset'
th = 15   # minimum streamline length
p = 15    # resample to this number of points

# -------------------- COUNT STREAMLINES --------------------
def count_streamlines_per_bundle(root_folder, fasciclesDict, th, p, subject_folders=None):
    bundle_counts = {fascicle: 0 for fascicle in fasciclesDict.keys()}
    print(f"Counting streamlines in: {root_folder}")
    
    if subject_folders is None:
        subject_folders = sorted(os.listdir(root_folder))
    
    for subject in subject_folders:
        subject_folder = os.path.join(root_folder, subject, 'tractography')
        if not os.path.exists(subject_folder):
            continue
        tract_files = [f for f in os.listdir(subject_folder) if f.endswith('.trk')]
        
        for tract_file in tract_files:
            tract_base = os.path.splitext(tract_file)[0]
            parts = tract_base.split('__')
            if len(parts) != 2:
                continue
            _, tract_name = parts
            if tract_name not in fasciclesDict:
                continue
            
            tract_path = os.path.join(subject_folder, tract_file)
            try:
                streams = load_tractogram(tract_path, 'same', bbox_valid_check=False)
                streams.remove_invalid_streamlines()
                filtered = [set_number_of_points(stream, p) 
                            for stream in streams.streamlines if len(stream) >= th]
                bundle_counts[tract_name] += len(filtered)
            except Exception as e:
                print(f"[ERROR] Could not load {tract_path}: {e}")
    return bundle_counts

# -------------------- PLOT COUNTS --------------------
def plot_streamline_counts_per_bundle(bundle_counts, title_suffix="", save_path="streamline_counts.png"):
    labels = list(bundle_counts.keys())
    counts = list(bundle_counts.values())

    plt.figure(figsize=(15, 6))
    bars = plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Bundles')
    plt.ylabel('Number of Streamlines')
    plt.title(f'Streamlines per Bundle {title_suffix}')
    plt.xticks(rotation=90)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                 ha='center', va='bottom', fontsize=8, color='black')
    plt.ylim(top=10000)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------------------- EXECUTION --------------------
train_subjects = sorted(os.listdir(train_path))
test_subjects = sorted(os.listdir(test_path))

# Train set
train_counts = count_streamlines_per_bundle(train_path, bundles_names, th, p, train_subjects)
print("Total streamlines per bundle in TRAIN:", train_counts)
plot_streamline_counts_per_bundle(train_counts, title_suffix="(Train)", save_path="plots/train_streamlines.png")

# Test set
test_counts = count_streamlines_per_bundle(test_path, bundles_names, th, p, test_subjects)
print("Total streamlines per bundle in TEST:", test_counts)
plot_streamline_counts_per_bundle(test_counts, title_suffix="(Test)", save_path="plots/test_streamlines.png")
