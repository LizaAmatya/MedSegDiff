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
