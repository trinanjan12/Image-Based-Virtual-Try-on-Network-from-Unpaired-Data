from torch.utils.data.dataset import Dataset

from data.image_folder import make_dataset

import os
from PIL import Image
from glob import glob as glob
import numpy as np
import random
import torch

import sys
import pickle
sys.path.append("/media/pintu/BACKUP/Trinanjan/current_project/virtual_try_on/piktorlabs/detectron2/projects/DensePose")
from densepose.data.structures import DensePoseResult


class RegularDataset(Dataset):
    def __init__(self, opt, augment):
        
        self.opt = opt
        self.root = opt.dataroot
        self.transforms = augment
        
        with open('./datasets/deepfashion1/deepfashion1_densepose.pkl', 'rb') as f:
            self.densepose_pkl_data = pickle.load(f)
            
        print(len(self.densepose_pkl_data),self.densepose_pkl_data[0])
        
        # input A (label maps)
        dir_A = '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        
        # input B (label images)
        dir_B = '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        
        self.dataset_size = len(self.densepose_pkl_data)
        self.img_height = 512
        self.img_width = 256

    def __getitem__(self, index):
        
        img_file_path = self.densepose_pkl_data[index]['file_name']
        img_name = img_file_path.split('img_highres')[-1].replace('/','').split('.')[0]
        
        
        # input A (label maps)
        A_path = os.path.join(self.dir_A, img_name + '.png')
        A = Image.open(A_path)
        temp_w,temp_h = Image.open(os.path.join(self.dir_B, img_name + '.jpg')).size
        A = self.parsing_embedding(A_path, 'seg')  # channel(20), H, W

        # input B (label maps)
        B_path = os.path.join(self.dir_A, img_name + '.png')
        B = Image.open(B_path)
        B = self.parsing_embedding(B_path, 'seg')  # channel(20), H, W
        #print(A_path,A_path,img_file_path)

        # densepose processings
        iuv_arr,bbox_xywh  = self.get_iuv_arr(self.densepose_pkl_data[index])
        x, y, w, h = int(bbox_xywh[0]), int(bbox_xywh[1]), int(bbox_xywh[2]), int(bbox_xywh[3])
        img_final_arr =  np.zeros((temp_h,temp_w,3))
        mask = np.transpose(iuv_arr,(1,2,0))
        img_final_arr[y:y+h,x:x+w,:] = mask
        dense_img = img_final_arr
        
        dense_img = Image.fromarray(dense_img.astype(np.uint8))
        dense_img = dense_img.resize((self.img_width,self.img_height),Image.BICUBIC)
        dense_img = np.array(dense_img)
        dense_img_parts_embeddings = self.parsing_embedding(
            dense_img[:, :, 0], 'densemap')
        dense_img_final = np.concatenate((dense_img_parts_embeddings, np.transpose(
            (dense_img[:, :, 1:]), axes=(2, 0, 1))), axis=0)  # channel(27), H, W
        A_tensor, B_tensor, dense_img_final = self.custom_transform(
            A, B, dense_img_final, True)

        # original seg mask
        seg_mask = Image.open(A_path)
        seg_mask = seg_mask.resize((self.img_width,self.img_height),Image.BICUBIC)
        seg_mask = np.array(seg_mask)
        seg_mask = torch.tensor(seg_mask, dtype=torch.long)

        input_dict = {'seg_map': A_tensor, 'dense_map': dense_img_final, 'target': B_tensor, 'seg_map_path': A_path,
                      'target_path': A_path,'seg_mask': seg_mask}

        return input_dict

    def custom_transform(self, input_image, target, densepose, per_channel_transform):
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

        if per_channel_transform:
            num_channel_image = input_image.shape[0]
            num_channel_dense = densepose.shape[0]
            tform_input_image_np = np.zeros(
                shape=input_image.shape, dtype=input_image.dtype)
            tform_target_np = np.zeros(shape=target.shape, dtype=target.dtype)
            tform_dense_np = np.zeros(
                shape=densepose.shape, dtype=densepose.dtype)

            for i in range(num_channel_image):
                if i != 0 and i != 1 and i != 2 and i != 4 and i != 13:
                    tform_input_image_np[i] = self.transforms['1'](
                        input_image[i])
                else:
                    tform_input_image_np[i] = self.transforms['2'](
                        input_image[i])
                tform_target_np[i] = self.transforms['2'](target[i])
            for i in range(num_channel_dense):
                if i > 24:
                    tform_dense_np[i] = self.transforms['2'](densepose[i].astype('uint8'))
                else:
                    tform_dense_np[i] = densepose[i]

        return torch.from_numpy(tform_input_image_np), torch.from_numpy(tform_target_np), torch.from_numpy(tform_dense_np)

    def parsing_embedding(self, parse_obj, parse_type):
        if parse_type == "seg":
            parse = Image.open(parse_obj)
            parse = parse.resize((self.img_width,self.img_height),Image.NEAREST)
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
    
    def get_iuv_arr(self,pose_data):
        result_encoded = pose_data['pred_densepose'].results[0]
        iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
        bbox_xywh = pose_data['pred_densepose'].boxes_xywh[0]
        return iuv_arr,bbox_xywh
    
    def __len__(self):
        return len(self.densepose_pkl_data) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'RegularDataset'
    
    