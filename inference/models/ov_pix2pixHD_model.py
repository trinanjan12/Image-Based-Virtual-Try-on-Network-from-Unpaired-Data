import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    # TODO this can be replaced by __init__

    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.opt = opt
        input_nc = opt.label_nc  # Number of input image channel/ 20 for segmentation / 3 for RGB image

        # Define Networks for shape generation
        if self.opt.shape_generation:
            netG_input_nc = 0  # generator input channel
            if opt.densepose_nc != 0:
                netG_input_nc += 27  # 26 channel for densepose

            netE_input_nc = 1
            netE_output_nc = 10
            netG_input_nc += netE_output_nc * input_nc   # 10 according to the paper

            # Generator network
            use_last_activation = opt.use_generator_last_activation
            self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                          opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                          opt.n_blocks_local, opt.norm, use_last_activation, gpu_ids=self.gpu_ids)
            # Encoder network
            self.netE = networks.define_G(netE_input_nc, netE_output_nc, opt.nef, 'encoder',
                                              opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        # Define Networks for app generation
        if self.opt.appearance_generation:
            netG_input_nc = 0  # generator input channel
            netE_input_nc = 3 + input_nc
            netE_output_nc = 30
            netG_input_nc = input_nc
            netG_input_nc += netE_output_nc   # 30 according to the paper

            # Generator network
            use_last_activation = True
            self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                            opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                            opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

            # Encoder network
            self.netE = networks.define_G(netE_input_nc, netE_output_nc, opt.nef, 'encoder',
                                                  opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        # FOR DEBUGGING
        print('netG_input_nc, netG_output_nc', netG_input_nc, opt.output_nc)
        print('netE_input_nc, netE_output_nc ', netE_input_nc, netE_output_nc)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)
            if self.opt.verbose:
                print('---------- Networks loaded -------------')

    def inference_forward_shape(self, query, ref, dense_map):

        query = query.float().cuda()
        dense_map = dense_map.float().cuda()
        ref = ref.float().cuda()

        query_ref_mixed = torch.cat(
            (query[:, 0:5, :, :], ref[:, 5:8, :, :], query[:, 8:, :, :]), axis=1)

        # query_ref_mixed = torch.cat(
        #     (query[:, 0:9, :, :],  ref[:, 5:8, :, :] , query[:, 8:9, :, :], ref[:, 9:10, :, :], query[:, 10:12, :, :], ref[:, 12:13, :, :],
        #     query[:, 13:16, :, :],ref[:, 16:20, :, :]), axis=1)

        # query_ref_mixed = torch.cat((query[:, 0:9, :, :], ref[:, 9:10, :, :], query[:, 10:12, :, :],
        #                              ref[:, 12:13, :, :], query[:, 13:16, :, :], ref[:, 16:20, :, :]), axis=1)

        feat_map_total = []
        for each_class in range(self.opt.label_nc):
            # bs, 1, H, w
            inp_enc = query_ref_mixed[:, each_class:each_class+1, :, :]
            with torch.no_grad():
                feat_map_each_class = self.netE.forward(
                    inp_enc)  # bs, 10, H, w
            feat_map_total.append(feat_map_each_class)

        feat_map_total = torch.cat([i for i in feat_map_total], dim=1)

        # local pooling step
        local_avg_pool_fn = nn.AvgPool2d((64, 64))
        feat_map_each_class_pooled = local_avg_pool_fn(feat_map_total)

        # Upscaling
        upscale_fn = Upsample(scale_factor=64, mode='nearest')
        feat_map_final = upscale_fn(feat_map_each_class_pooled)

        input_concat = torch.cat((dense_map, feat_map_final), dim=1)

        with torch.no_grad():
            fake_image = self.netG.forward(input_concat)

        return query_ref_mixed, fake_image

    def inference_forward_appearance(self, query_img, query_parse_map, query_seg_map, ref_img, ref_parse_map, ref_seg_map, C_tensor):

        query_img = query_img.float().cuda()
        query_parse_map = query_parse_map.float().cuda()
        query_seg_map = query_seg_map.float().cuda()
        ref_img = ref_img.float().cuda()
        ref_parse_map = ref_parse_map.float().cuda()
        ref_seg_map = ref_seg_map.float().cuda()
        generated_parse_map = C_tensor.float().cuda()

        
        app_feature_map = torch.zeros((1, 30, 512, 256)).float().cuda()

        # query image
        selected_seg_map_tensor = query_seg_map
        selected_seg_map_tensor = torch.unsqueeze(
            selected_seg_map_tensor, 0)
        selected_img_tensor = query_img
        selected_seg_parse_map = query_parse_map

        input_encoder = torch.cat(
            (selected_img_tensor, selected_seg_parse_map), 1).cuda()
        with torch.no_grad():
            y_query_enc = self.netE.forward(input_encoder)

        # ref image
        selected_seg_map_tensor = ref_seg_map
        selected_seg_map_tensor = torch.unsqueeze(
            selected_seg_map_tensor, 0)
        selected_img_tensor = ref_img
        selected_seg_parse_map = ref_parse_map

        input_encoder = torch.cat(
            (selected_img_tensor, selected_seg_parse_map), 1).cuda()
        with torch.no_grad():
            y_ref_enc = self.netE.forward(input_encoder)
        
        for num_seg_channel in range(20):
            if 4 < num_seg_channel < 8:
                selected_seg_map_tensor = torch.unsqueeze(ref_seg_map, 0)
                app_feature_vec_temp = y_ref_enc
            else:
                selected_seg_map_tensor = torch.unsqueeze(query_seg_map, 0)
                app_feature_vec_temp = y_query_enc


            indices = (selected_seg_map_tensor == int(
                num_seg_channel)).nonzero()  # nx4

            for enc_channel in range(30):
                region_of_interest = app_feature_vec_temp[indices[:, 0],
                                                            indices[:, 1] + enc_channel, indices[:, 2], indices[:, 3]]
                enc_each_channel_mean = torch.mean(
                    region_of_interest).expand_as(region_of_interest)
                app_feature_map[indices[:, 0], indices[:, 1] + enc_channel,
                                indices[:, 2], indices[:, 3]] = enc_each_channel_mean

        input_concat = torch.cat(
            (generated_parse_map, app_feature_map), dim=1).cuda()

        with torch.no_grad():
            fake_image = self.netG.forward(input_concat)

        return fake_image


class InferenceModel(Pix2PixHDModel):
    def forward(self, query_parse_map, ref_parse_map, query_seg_map, ref_seg_map, query_img, ref_img, C_tensor):
        if self.opt.shape_generation:
            return self.inference_forward_shape(query_parse_map, ref_parse_map, C_tensor)
        if self.opt.appearance_generation:
            return self.inference_forward_appearance(query_img, query_parse_map, query_seg_map, ref_img, ref_parse_map, ref_seg_map, C_tensor)
