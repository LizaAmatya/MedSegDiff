import os
import random
import clip
import torch
from torch.utils.data import Dataset
import json
import torchvision.transforms as ttf
import numpy as np


class CapDataset(Dataset):
    def __init__(self, args, clip_model, processor, mode="train"):
        self.args = args
        self.data_root = args.data_dir
        self.clip_model = clip_model
        self.processor = processor
        self.mode = mode
        self.dtype = torch.float16 if args.use_fp16 else torch.float32
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_json, "r") as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = ttf.Compose([
            ttf.RandomRotation(90),  # RandRotate90 equivalent (though not axis-specific)
            ttf.RandomHorizontalFlip(p=0.10),  # RandFlip for spatial_axis=0
            ttf.RandomVerticalFlip(p=0.10),  # RandFlip for spatial_axis=1 (Vertical Flip)
            ttf.RandomApply([ttf.RandomRotation(degrees=(-90, 90))], p=0.10), # Rough equivalent for 3D flips (manual setup needed)
            ttf.RandomApply([ttf.RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.5),  # RandScaleIntensity equivalent
            ttf.RandomApply([ttf.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),  # RandShiftIntensity equivalent
            ttf.ToTensor()  # ToTensor (MONAI's dtype option not needed here as default float is used)
        ])

        val_transform = ttf.Compose(
            [
                ttf.ToTensor(),
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
                # image_abs_path = os.path.join(self.data_root, image_path)
                try:
                    image = np.load(image_path)
                    
                    # if self.transform:
                    #     image = self.transform(image)
                    #     print('image shape----- after transform', image.shape)

                    image = torch.tensor(image, dtype=self.dtype)
                    print(f"Image size (tensor): {image.shape}")

                except Exception as e:
                    raise ValueError(f"Error loading image at {image_path}: {e}")
                
                text_path = data["text"]
                # text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_path, "r") as text_file:
                    raw_text = text_file.read()
                
                chunks = self.split_text(raw_text)
                chunk_tensors = [clip.tokenize([chunk]).squeeze(0).to(self.device) for chunk in chunks]
                text_tensor = torch.cat(chunk_tensors, dim=0) 
                
                # text_tensor = clip.tokenize([raw_text]).squeeze(0).to(self.device)
                ret = (image, text_tensor, image_path)        #image, condition, name (metadata)
                
                return ret

                # answer = raw_text

                # prompt_question = random.choice(self.caption_prompts)

                # question = self.image_tokens + prompt_question

                # text_tensor = self.tokenizer(
                #     raw_text,
                #     max_length=self.args.max_length,
                #     truncation=True,
                #     padding="max_length",
                #     return_tensors="pt",
                # )

                # input_id = text_tensor["input_ids"][0]
                # attention_mask = text_tensor["attention_mask"][0]

                # valid_len = torch.sum(attention_mask)
                # if valid_len < len(input_id):
                #     input_id[valid_len] = self.tokenizer.eos_token_id

                # question_tensor = self.tokenizer(
                #     question,
                #     max_length=self.args.max_length,
                #     truncation=True,
                #     padding="max_length",
                #     return_tensors="pt",
                # )
                # question_len = torch.sum(question_tensor["attention_mask"][0])

                # label = input_id.clone()
                # label[label == self.tokenizer.pad_token_id] = -100
                # label[:question_len] = -100

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

    def split_text(raw_text: str, max_tokens=77):
        words = raw_text.split()
        chunks = []
        for i in range(0, len(words), max_tokens - 1):  # Reserve one token for [EOS]
            chunk = " ".join(words[i : i + max_tokens - 1])
            chunks.append(chunk)
        return chunks
    