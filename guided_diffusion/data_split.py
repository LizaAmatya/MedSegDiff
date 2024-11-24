import os
import json
import random

# parent_dir = os.path.dirname(os.getcwd())
parent_dir = '/media/'      # Data folder
# print("curr dir", parent_dir)
base_dir = os.path.join(parent_dir, "M3Ddataset/M3D_Cap_npy/ct_quizze/")
op_dir = os.getcwd()
# print('output dir', op_dir)
output_json = os.path.join(op_dir + '/data/', "dataset_split.json") # Output JSON file
# print('output dir', op_dir)

train_data = []
test_data = []

for root, dirs, files in os.walk(base_dir):
    for file in files:
        print('file', file)
        if file.endswith(".npy"):
            image_path = os.path.join(root, file)
            # Get the path to the corresponding text file
            text_file = os.path.join(root, "text.txt")

            if os.path.exists(text_file):
                # Construct the data entry
                entry = {"image": image_path, "text": text_file}

                # Randomly assign to train or test (e.g., 80-20 split)
                if random.random() < 0.9:
                    train_data.append(entry)
                else:
                    test_data.append(entry)

# Organize data into the specified JSON format
dataset_split = {"train": train_data, "test": test_data}

# Write the JSON data to a file
with open(output_json, "w") as json_file:
    json.dump(dataset_split, json_file, indent=4)

print(f"Dataset split and saved to {output_json}")
