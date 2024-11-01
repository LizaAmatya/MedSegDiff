import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import Counter
import torchvision.transforms as ttf
from multiprocessing import Pool
from unidecode import unidecode
from monai.transforms import CropForeground


parent_dir = os.path.dirname(os.getcwd())
print("curr dir", parent_dir)
base_dir = os.path.join(parent_dir, 'M3Ddataset/')

print('base dir', base_dir)

input_dir = os.path.join(base_dir, "ct_quizze_00/")
output_dir = os.path.join(base_dir, 'M3D_Cap_npy/ct_quizze/')

# Get all subfolders [00001, 00002....]
subfolders = [folder for folder in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, folder))]


transform = ttf.Compose(
    [
        CropForeground(),
        ttf.Resize(size=(256, 256), interpolation=ttf.InterpolationMode.BILINEAR)
    ]
)


def process_subfolder(subfolder):
    output_id_folder = os.path.join(output_dir, subfolder)
    input_id_folder = os.path.join(input_dir, subfolder)

    os.makedirs(output_id_folder, exist_ok=True)

    for subsubfolder in os.listdir(input_id_folder):
        if subsubfolder.endswith('.txt'):
            text_path = os.path.join(input_dir, subfolder, subsubfolder)
            with open(text_path, 'r') as file:
                text_content = file.read()

            search_text = "study_findings:"
            index = text_content.find(search_text)

            if index != -1:
                filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
            else:
                print("Specified string not found")
                filtered_text = text_content.replace("\n", " ").strip()


            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                search_text = "discussion:"
                index = text_content.find(search_text)
                if index != -1:
                    filtered_text = text_content[index + len(search_text):].replace("\n", " ").strip()
                else:
                    print("Specified string not found")
                    filtered_text = text_content.replace("\n", " ").strip()


            if len(filtered_text.replace("\n", "").replace(" ", "")) < 5:
                filtered_text = text_content.replace("\n", " ").strip()


            new_text_path = os.path.join(output_dir, subfolder, subsubfolder)
            with open(new_text_path, 'w') as new_file:
                new_file.write(filtered_text)

        subsubfolder_path = os.path.join(input_dir, subfolder, subsubfolder)

        if os.path.isdir(subsubfolder_path):
            print('sub sub folder', subsubfolder_path)
            subsubfolder = unidecode(subsubfolder)
            output_path = os.path.join(output_dir, subfolder, f'{subsubfolder}.npy')

            image_files = [file for file in os.listdir(subsubfolder_path) if
                           file.endswith('.jpeg') or file.endswith('.png')]
            if len(image_files) == 0:
                continue

            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

            images_3d = []
            for image_file in image_files:
                image_path = os.path.join(subsubfolder_path, image_file)
                try:
                    img = Image.open(image_path)
                    img = img.convert("L")
                    img_array = np.array(img)
                    # normalization
                    img_array = img_array.astype(np.float32) / 255.0
                    images_3d.append(img_array[None])
                except Exception as e:
                    print("This image is error: ", e)

            images_3d_pure = []
            try:
                img_shapes = [img.shape for img in images_3d]
                item_counts = Counter(img_shapes)
                most_common_shape = item_counts.most_common(1)[0][0]
                for img in images_3d:
                    if img.shape == most_common_shape:
                        images_3d_pure.append(img)
                        print('3d image pure', img.shape)
                final_3d_image = np.vstack(images_3d_pure)
                print('3d images pure @@@@@@@', type(images_3d_pure))
                print("final 3d image @@@@@@@", type(final_3d_image), final_3d_image.shape)

                image = final_3d_image[np.newaxis, ...]
                print('here process 1', type(image), image.shape)
                image = image - image.min()
                image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
                print("here process 2", image.shape)

                img_trans = transform(image)
                # print('-----image trans', img_trans.shape)
                np.save(output_path, img_trans)
            except Exception as e:
                print([img.shape for img in images_3d])
                print("This folder is vstack error: ", e)
                
                return


if __name__ == "__main__":
    with Pool(processes=32) as pool:
        with tqdm(total=len(subfolders), desc="Processing") as pbar:
            for _ in pool.imap_unordered(process_subfolder, subfolders):
                pbar.update(1)
                