from torch.utils.data.dataset import Dataset

from data.image_folder import make_dataset

import os
from PIL import Image
from glob import glob as glob
import numpy as np
import random
import torch


class RegularDataset(Dataset):
    def __init__(self, opt, augment):
        self.opt = opt
        self.root = opt.dataroot
        self.transforms = augment

        # input A (label maps)
        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        # input B (label images)
        dir_B = '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        # input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        A = self.parsing_embedding(A_path, 'seg')  # channel(20), H, W
        # A_tensor = self.transforms['1'](A)
        A_tensor = torch.from_numpy(A)

        # input B (images)
        B_path = self.B_paths[index]
        B = Image.open(B_path)
        B = np.array(B)
        B_tensor = self.transforms['1'](B)

        # original seg mask
        seg_mask = Image.open(A_path)
        seg_mask = np.array(seg_mask)
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)

        input_dict = {'seg_map': A_tensor, 'target': B_tensor, 'seg_map_path': A_path,
                      'target_path': B_path, 'seg_mask': seg_mask}

        return input_dict

    def parsing_embedding(self, parse_path, parse_type = "seg"):
        if parse_type == "seg":
            parse = Image.open(parse_path)
            parse = np.array(parse)
            parse_channel = 20

        parse_emb = []
        for i in range(parse_channel):
            parse_emb.append((parse == i).astype(np.float32).tolist())
            
        parse = np.array(parse_emb).astype(np.float32)
        return parse  # (channel,H,W)

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'RegularDataset'
