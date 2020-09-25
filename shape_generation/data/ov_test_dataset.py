import torch
from torch.utils.data.dataset import Dataset

from data.image_folder import make_dataset

import os
from PIL import Image
from glob import glob as glob
import numpy as np


class TestDataset(Dataset):
    
    def __init__(self, opt, augment):
        self.opt = opt
        self.root = opt.dataroot
        self.transforms = augment

        # input A (label maps)
        dir_A = '_query_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        # input B (label images)
        dir_B = '_ref_label'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        # densepose maps
        self.dir_densepose = os.path.join(
            opt.dataroot, opt.phase + '_densepose')
        self.densepose_paths = sorted(glob(self.dir_densepose + '/*'))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        # input A (label maps)
        A_path = self.A_paths[index]
        A = self.parsing_embedding(A_path, 'seg')
        A_tensor = torch.from_numpy(A)

        # input B (label maps)
        B_path = self.B_paths[index]
        B = self.parsing_embedding(B_path, 'seg')
        B_tensor = torch.from_numpy(B)

        # densepose maps
        dense_path = self.densepose_paths[index]
        dense_img = np.load(dense_path).astype('uint8')  # channel last
        dense_img_parts_embeddings = self.parsing_embedding(
            dense_img[:, :, 0], 'densemap')
        dense_img_parts_embeddings = np.transpose(dense_img_parts_embeddings,axes= (1,2,0))
        dense_img_final = np.concatenate((dense_img_parts_embeddings,dense_img[:, :, 1:]), axis=-1)  # channel(27), H, W
        dense_img_final = torch.from_numpy(np.transpose(dense_img_final,axes= (2,0,1)))

        input_dict = {'query': A_tensor, 'dense_map': dense_img_final,
                      'ref': B_tensor, 'query_path': A_path,'ref_path': B_path}

        return input_dict

    def custom_transform(self, input_image, per_channel_transform=True, input_type="densepose"):

        if per_channel_transform:
            num_channel_img = input_image.shape[0]
            tform_input_image_np = np.zeros(
                shape=input_image.shape, dtype=input_image.dtype)
            if input_type == "densepose":
                for i in range(num_channel_img):
                    if i > 24:
                        tform_input_image_np[i] = self.transforms['1'](
                            input_image[i].astype('uint8'))
                    else:
                        tform_input_image_np[i] = input_image[i]

            return torch.from_numpy(tform_input_image_np)

    def parsing_embedding(self, parse_obj, parse_type):
        if parse_type == "seg":
            parse = Image.open(parse_obj)
            parse = np.array(parse)
            parse_channel = 20

        elif parse_type == "densemap":
            parse = np.array(parse_obj)
            parse_channel = 25

        parse_emb = []

        for i in range(parse_channel):
            parse_emb.append((parse == i).astype(np.float32).tolist())

        parse = np.array(parse_emb).astype(np.float32)
        return parse

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'TestDataset'
