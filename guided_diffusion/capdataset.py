import os
import random
import torch
from torch.utils.data import Dataset
import json
import torchvision.transforms as ttf
import numpy as np


class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_json, "r") as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = ttf.Compose([
            ttf.RandomRotation(90),  # RandRotate90 equivalent (though not axis-specific)
            ttf.RandomHorizontalFlip(p=0.10),  # RandFlip for spatial_axis=0
            ttf.RandomVerticalFlip(p=0.10),  # RandFlip for spatial_axis=1 (Vertical Flip)
            ttf.RandomApply([ttf.RandomRotation([90])], p=0.10),  # Rough equivalent for 3D flips (manual setup needed)
            ttf.RandomApply([ttf.RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.5),  # RandScaleIntensity equivalent
            ttf.RandomApply([ttf.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),  # RandShiftIntensity equivalent
            ttf.ToTensor()  # ToTensor (MONAI's dtype option not needed here as default float is used)
        ])

        val_transform = ttf.Compose(
            [
                ttf.ToTensor(dtype=torch.float),
            ]
        )
        # set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif mode == "test":
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)
                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, "r") as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)