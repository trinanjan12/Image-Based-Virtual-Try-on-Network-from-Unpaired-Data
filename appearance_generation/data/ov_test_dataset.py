from torch.utils.data.dataset import Dataset

from data.image_folder import make_dataset

import os
from PIL import Image
from glob import glob as glob
import numpy as np
import random
import torch


class TestDataset(Dataset):
    def __init__(self, opt, augment):

        self.opt = opt
        self.root = opt.dataroot
        self.transforms = augment

        # query label (label maps)
        dir_query_label = '_query_label'
        self.dir_query_label = os.path.join(
            opt.dataroot, opt.phase + dir_query_label)
        self.query_label_paths = sorted(make_dataset(self.dir_query_label))

        # ref label (label images)
        dir_ref_label = '_ref_label'
        self.dir_ref_label = os.path.join(
            opt.dataroot, opt.phase + dir_ref_label)
        self.ref_label_paths = sorted(make_dataset(self.dir_ref_label))

        # query img (RGB maps)
        dir_query_img = '_query_img'
        self.dir_query_img = os.path.join(
            opt.dataroot, opt.phase + dir_query_img)
        self.query_img_paths = sorted(make_dataset(self.dir_query_img))

        # ref img (RGB images)
        dir_ref_img = '_ref_img'
        self.dir_ref_img = os.path.join(
            opt.dataroot, opt.phase + dir_ref_img)
        self.ref_img_paths = sorted(make_dataset(self.dir_ref_img))

        # generated segmentation from shape_generation (label maps)
        dir_query_ref_label = '_query_ref_label'
        self.dir_query_ref_label = os.path.join(
            opt.dataroot, opt.phase + dir_query_ref_label)
        self.query_ref_label_paths = sorted(
            make_dataset(self.dir_query_ref_label))

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

    def __getitem__(self, index):

        # query label (label maps)
        query_label_path = self.query_label_paths[index]
        query_label_parse = self.parsing_embedding(query_label_path, 'seg')
        query_label_parse = torch.from_numpy(query_label_parse)  # channel(20), H, W

        query_label_seg_mask = Image.open(query_label_path)
        query_label_seg_mask = np.array(query_label_seg_mask)
        query_label_seg_mask = torch.tensor(query_label_seg_mask, dtype=torch.long)

        # ref label (label maps)
        ref_label_path = self.ref_label_paths[index]
        ref_label_parse = self.parsing_embedding(ref_label_path, 'seg')
        ref_label_parse = torch.from_numpy(
            ref_label_parse)  # channel(20), H, W

        ref_label_seg_mask = Image.open(ref_label_path)
        ref_label_seg_mask = np.array(ref_label_seg_mask)
        ref_label_seg_mask = torch.tensor(ref_label_seg_mask, dtype=torch.long)

        # input B (images)
        query_img_path = self.query_img_paths[index]
        query_img = Image.open(query_img_path)
        query_img = self.transforms['1'](query_img)

        # input B (images)
        ref_img_path = self.ref_img_paths[index]
        ref_img = Image.open(ref_img_path)
        ref_img = self.transforms['1'](ref_img)

        # input A (label maps)
        query_ref_label_path = self.query_ref_label_paths[index]
        query_ref_label_parse = self.parsing_embedding(query_ref_label_path, 'seg')  # channel(20), H, W
        C_tensor = torch.from_numpy(query_ref_label_parse)
        
        query_ref_label_seg_mask = Image.open(query_ref_label_path)
        query_ref_label_seg_mask = np.array(query_ref_label_seg_mask)
        query_ref_label_seg_mask = torch.tensor(query_ref_label_seg_mask, dtype=torch.long)

            
        input_dict = {
            'query_parse_map': query_label_parse,
            'ref_parse_map': ref_label_parse,
            'query_seg_map': query_label_seg_mask,
            'ref_seg_map': ref_label_seg_mask,
            'query_img': query_img,
            'ref_img': ref_img,
            'C_tensor_parse_map': C_tensor,
            'C_tensor_seg_map': query_ref_label_seg_mask,
            'path': query_label_path
        }

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
        return parse  # (channel,H,W)

    def __len__(self):
        return len(self.query_label_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'TestDataset'
