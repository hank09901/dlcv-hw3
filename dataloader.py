import os
import json
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from tokenizer import BPETokenizer

class pb2_dataset(Dataset):
    def __init__(self, data_dir,  mode, json_dir=None, batch_size=32):
        self.data_dir = data_dir
        self.tokenizer = BPETokenizer('./encoder.json', './vocab.bpe')
        self.max_cap_length = 100
        self.mode = mode
        if mode == "train" and json_dir is not None:
            json_file = os.path.join(json_dir, "train.json") if mode == "train" else os.path.join(json_dir, "val.json")
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            # sort annotations_list by image_id
            self.annotations_list = json_data['annotations']
            # self.annotations_list = sorted(annotations_list, key=lambda x: x['image_id'])

            # print("annotations_list: ", self.annotations_list)
            images_name_list = []
            for annotation_dict in self.annotations_list:
                image_id = annotation_dict['image_id']
                # images_name_list += [image_dict['file_name'] for image_dict in json_data['images'] if image_dict['id'] == image_id]
                for image_dict in json_data['images']:
                    if image_dict['id'] == image_id:
                        images_name_list.append(image_dict["file_name"])
            self.images_name_list = images_name_list
            # print("images_name_list: ", self.images_name_list)
            # create a sorted list for id
            # self.id_list = [image_dict['id'] for image_dict in self.images_name_list] 
            # print("id_list: ", self.id_list)
            self.image_pth = [ os.path.join(data_dir, img) for img in self.images_name_list]
            # idx = np.random.choice(2000, 2, replace=False)
            # self.image_pth = [os.path.join(data_dir, img) for i, img in enumerate(os.listdir(data_dir)) if i in idx]
            # print("self.image_pth: ", self.image_pth)
            # print("self.annotations_list: ", self.annotations_list)
            # print("self.images_name_list: ", self.images_name_list)
        elif mode == "val":
            # self.image_pth = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
            # use one image for validation
            # random choose five images from 2000 image in val directory for validation
            # idx = np.random.choice(2000, batch_size, replace=False)
            # self.image_pth = [os.path.join(data_dir, img) for i, img in enumerate(os.listdir(data_dir)) if i in idx]
            # self.image_pth = [os.path.join(data_dir, "000000000000.jpg")]
            self.image_pth = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
            print("self.image_pth: ", self.image_pth)


        else:
            self.image_pth = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
            # self.images_name_list = [img.split('/')[-1].split('.')[0] for img in os.listdir(data_dir)]
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(),
            # transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.image_pth)

    def __getitem__(self, idx):
        image_name = self.image_pth[idx]
        image = Image.open(image_name).convert('RGB')
        name = image_name.split('/')[-1].split('.')[0]
        if self.mode == "train":
            image = self.train_transform(image)
        else:
            image = self.test_transform(image)
        
        if self.mode == "train":
            # print("image_name: ", name)
        
            caption = self.annotations_list[idx]['caption']
            # print("caption: ", caption)
            caption_input = self.tokenizer.encode(caption)
            
            caption_input.insert(0, 50256)
            caption_input = np.array(caption_input)
            # print("caption_input: ", caption_input)
            # print("caption_input shape: ", caption_input.shape)
            # caption_input size: (batch_size, max_cap_length)
            caption_input = np.pad(caption_input, (0, self.max_cap_length - len(caption_input)), 'constant', constant_values=50256)
            # print("caption_input shape: ", caption_input.shape) #100
            caption_input = torch.tensor(caption_input)

            caption_gt = self.tokenizer.encode(caption)
            caption_gt.append(50256)
            caption_gt = np.array(caption_gt)
            # 197 is encoder token for <pad>
            caption_gt = np.pad(caption_gt, (257, self.max_cap_length - len(caption_gt)), 'constant', constant_values=-100)
            # caption_gt = np.pad(caption_gt, (257, self.max_cap_length - len(caption_gt)), 'constant', constant_values=50256)
            caption_gt = torch.tensor(caption_gt)
            # print("caption_gt shape: ", caption_gt.shape) # 297
            return name, image, caption_input, caption_gt
        
        return (name, image)