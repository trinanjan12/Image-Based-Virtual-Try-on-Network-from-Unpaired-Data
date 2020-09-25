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

        # input A (label maps source)
        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        # input B (label images target)
        dir_B = '_label'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        # densepose maps
        self.dir_densepose = os.path.join(
            opt.dataroot, opt.phase + '_densepose')
        self.densepose_paths = sorted(glob(self.dir_densepose + '/*'))

        self.dataset_size = len(self.A_paths)

    def custom_transform(self, input_image, per_channel_transform):

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        if per_channel_transform:
            num_channel_image = input_image.shape[0]
            tform_input_image_np = np.zeros(
                shape=input_image.shape, dtype=input_image.dtype)

            for i in range(num_channel_image):
                # TODO check why i!=5 makes a big difference in the output
                if i != 1 and i != 2 and i != 4 and i != 5 and i != 13:
                    # if i != 0 and i != 1 and i != 2 and i != 4 and i != 13:
                    tform_input_image_np[i] = self.transforms['1'](
                        input_image[i])
                else:
                    tform_input_image_np[i] = self.transforms['2'](
                        input_image[i])

        return torch.from_numpy(tform_input_image_np)

    def __getitem__(self, index):

        # input A (label maps source)
        A_path = self.A_paths[index]
        A = self.parsing_embedding(A_path, 'seg')  # channel(20), H, W

        # input B (label maps target)
        B_path = self.B_paths[index]
        B = self.parsing_embedding(B_path, 'seg')  # channel(20), H, W

        # densepose maps
        dense_path = self.densepose_paths[index]
        dense_img = np.load(dense_path).astype('uint8')
        dense_img_parts_embeddings = self.parsing_embedding(
            dense_img[:, :, 0], 'densemap')

        dense_img_parts_embeddings = np.transpose(
            dense_img_parts_embeddings, axes=(1, 2, 0))
        dense_img_final = np.concatenate(
            (dense_img_parts_embeddings, dense_img[:, :, 1:]), axis=-1)  # channel(27), H, W

        # original seg mask
        seg_mask = Image.open(A_path)
        seg_mask = np.array(seg_mask)
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)

        # final returns
        A_tensor = self.custom_transform(A, True)
        B_tensor = torch.from_numpy(B)
        dense_img_final = torch.from_numpy(
            np.transpose(dense_img_final, axes=(2, 0, 1)))

        input_dict = {'seg_map': A_tensor, 'dense_map': dense_img_final, 'target': B_tensor, 'seg_map_path': A_path,
                      'target_path': A_path, 'densepose_path': dense_path, 'seg_mask': seg_mask}

        return input_dict

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
        return 'RegularDataset'

