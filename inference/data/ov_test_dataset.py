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

        if self.opt.shape_generation:
            # densepose maps
            dir_densepose = '_densepose'
            self.dir_densepose = os.path.join(
                opt.dataroot, opt.phase + dir_densepose)
            self.densepose_paths = sorted(glob(self.dir_densepose + '/*'))

        if self.opt.appearance_generation:
            # generated segmentation from shape_generation (label maps)
            dir_query_ref_label = '_query_ref_label'
            self.dir_query_ref_label = os.path.join(
                opt.dataroot, opt.phase + dir_query_ref_label)
            self.query_ref_label_paths = sorted(
                make_dataset(self.dir_query_ref_label))

    def custom_tranform(self, A_input_label, B_input_label, A_input_img, B_input_img, C_input, per_channel_transform=False):
        # NOTE 1
        # The idea of writing custom transformation function is
        # Apply same transformation to input and target(https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/6)
        # There should be better ways to apply same random transfomation instead of resetting the seed  ?
        # Apply channel wise transformation as needed for our case

        # NOTE 2
        # Currently we are using no affine transformation on the target
        # Only affine transformation is applied to input segmentation image
        # The target segmentation image, densepose has only totensor() and normalization transformation

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        '''
            1. A_input_label --> Label Map Query
            2. B_input_label --> Label Map Ref
            3. A_input_img --> Image Query
            4. B_input_img --> Image Query
            5. C_input --> Densepose/query_ref segmentation
        '''

        if per_channel_transform:

            num_channel_label = A_input_label.shape[0]

            tform_A_input_label = np.zeros(
                shape=A_input_label.shape, dtype=A_input_label.dtype)
            tform_B_input_label = np.zeros(
                shape=B_input_label.shape, dtype=B_input_label.dtype)

            # tform_A_input_img = np.zeros(
            #     shape=A_input_img.shape, dtype=A_input_img.dtype)
            # tform_B_input_img = np.zeros(
            #     shape=B_input_img.shape, dtype=B_input_img.dtype)

            tform_C_input = np.zeros(shape=C_input.shape, dtype=C_input.dtype)

            for i in range(num_channel_label):
                tform_A_input_label[i] = self.transforms['2'](A_input_label[i])
                tform_B_input_label[i] = self.transforms['2'](B_input_label[i])

            tform_A_input_img = self.transforms['2'](A_input_img)
            tform_B_input_img = self.transforms['2'](B_input_img)

        if self.opt.shape_generation:
            num_channel_dense = C_input.shape[0]
            for i in range(num_channel_dense):
                tform_C_input[i] = self.transforms['2'](C_input[i])

        if self.opt.appearance_generation:
            for i in range(num_channel_label):
                tform_C_input[i] = self.transforms['2'](C_input[i])

        A_tensor_label, B_tensor_label, A_tensor_img, B_tensor_img, C_tensor = torch.from_numpy(tform_A_input_label), torch.from_numpy(
            tform_B_input_label), tform_A_input_img, tform_B_input_img, torch.from_numpy(tform_C_input)

        return A_tensor_label, B_tensor_label, A_tensor_img, B_tensor_img, C_tensor

    def __getitem__(self, index):

        # query label (label maps)
        query_label_path = self.query_label_paths[index]
        query_label_parse = self.parsing_embedding(
            query_label_path, 'seg')  # channel(20), H, W

        query_label_seg_mask = Image.open(query_label_path)
        query_label_seg_mask = np.array(query_label_seg_mask)
        query_label_seg_mask = torch.tensor(
            query_label_seg_mask, dtype=torch.long)

        # ref label (label maps)
        ref_label_path = self.ref_label_paths[index]
        ref_label_parse = self.parsing_embedding(
            ref_label_path, 'seg')  # channel(20), H, W

        ref_label_seg_mask = Image.open(ref_label_path)
        ref_label_seg_mask = np.array(ref_label_seg_mask)
        ref_label_seg_mask = torch.tensor(ref_label_seg_mask, dtype=torch.long)

        # input B (images)
        query_img_path = self.query_img_paths[index]
        query_img = Image.open(query_img_path)

        # input B (images)
        ref_img_path = self.ref_img_paths[index]
        ref_img = Image.open(ref_img_path)

        if self.opt.shape_generation:
            # densepose maps
            dense_path = self.densepose_paths[index]
            dense_img = np.load(dense_path)  # channel last
            dense_img_parts_embeddings = self.parsing_embedding(
                dense_img[:, :, 0], 'densemap')
            dense_img_final = np.concatenate((dense_img_parts_embeddings, np.transpose(
                (dense_img[:, :, 1:]), axes=(2, 0, 1))), axis=0)  # channel(27), H, W

            C_tesor = dense_img_final

        if self.opt.appearance_generation:
            # input A (label maps)
            query_ref_label_path = self.query_ref_label_paths[index]
            query_ref_label_parse = self.parsing_embedding(
                query_ref_label_path, 'seg')  # channel(20), H, W
            # query_ref_label_seg_mask = Image.open(query_ref_label_path)
            # query_ref_label_seg_mask = np.array(query_ref_label_seg_mask)
            # query_ref_label_seg_mask = torch.tensor(
            #     query_ref_label_seg_mask, dtype=torch.long)

            C_tesor = query_ref_label_parse

        A_tensor_label, B_tensor_label, A_tensor_img, B_tensor_img, C_tensor = self.custom_tranform(
            query_label_parse, ref_label_parse, query_img, ref_img, C_tesor, True)

        input_dict = {
            'query_parse_map': A_tensor_label,
            'ref_parse_map': B_tensor_label,
            'query_seg_map': query_label_seg_mask,
            'ref_seg_map': ref_label_seg_mask,
            'query_img': A_tensor_img,
            'ref_img': B_tensor_img,
            'C_tensor': C_tensor,
            'path' : query_label_path
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
