import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import tqdm

from huggingface_hub import login

# Authenticate using your Hugging Face token
login(token=os.getenv("DATASETTOKEN"))

# dataset = load_dataset("GoodBaiBai88/M3D-CAP", streaming=True)
# dataset = load_dataset("GoodBaiBai88/M3D-CAP", streaming=True, split="train[:0.5%]")

dataset = load_dataset(
    "GoodBaiBai88/M3D-CAP",
    split="test",
    streaming=True,
)
for example in tqdm(dataset, total=10):  # Adjust 'total' based on the subset size
    # Process each example
    print(example)
    pass

dataset = dataset.with_format('torch')

dataloader = DataLoader(dataset["test"], batch_size=1, shuffle=True)

print('dataloader', dataloader.dataset)


# https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap/tree/main/data_examples

# script to run

# python scripts/segmentation_train.py --data_name 'M3D_CAP' --data_dir /media/M3Ddataset/M3D_Cap_npy/ct_quizze/ --out_dir './output/' --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 4
