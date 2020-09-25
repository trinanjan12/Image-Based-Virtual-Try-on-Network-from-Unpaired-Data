import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample
from torchvision import transforms

########################################
'''
    Generator network 
        input_nc = label_nc(20) +  30 encoder output
        output_nc = 3 --> tryon RGB image

    Discriminator network
        input_nc = label_nc(20) + output_nc(20)
        output_nc = opt.ndf(64) patchgan #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py#L32
    Encoder Network
        input_nc = 3+20  e*t = (x,s(x)) suggested in the paper
        output_nc = 30 encoder embedding dimension
    
    run the code with python train_new.py --name name_of_exp --dataroot ./datasets/dataroot/ --tf_log

'''
########################################


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    # TODO this can be replaced by __init__
    #################### INITIALIZE ####################
    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain  # Train/Test mode
        self.opt = opt  # options

        #################### DEFINE NETWORKS ####################

        '''
            Initialize Generator Network
        '''

        # Define Networks
        # Generator network
        netG_input_nc = opt.label_nc  # 20 (segmentation class)
        netG_input_nc += opt.feat_num   # 30 encoder embedding
        netG_output_nc = opt.output_nc  # 3 RGB TRYON Image

        self.netG = networks.define_G(netG_input_nc, netG_output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        '''
            Initialize Discriminator Network
        '''

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.label_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        '''
            Initialize Encoder Network
        '''

        # encoder takes the rgb image along with one hot encoding 3 + 20
        netE_input_nc = 3 + opt.label_nc
        netE_output_nc = opt.feat_num  # 30 encoder embedding
        self.netE = networks.define_G(netE_input_nc, netE_output_nc, opt.nef, 'encoder',
                                      opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        print('netG_input_nc -- 50, netG_output_nc -- 3, netD_input_nc -- 23, netD_output_nc',
              netG_input_nc, opt.output_nc, netD_input_nc, opt.ndf)
        print('netE_input_nc -- 23, netE_output_nc -- 30',
              netE_input_nc, netE_output_nc)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        #################### LOAD NETWORKS ####################

        pretrained_path = '' if not self.isTrain else opt.load_pretrain
        self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
        self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
        self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        #################### SET LOSS FUNCTIONS AND OPTIMIZERS ####################

        # TODO https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/75
        # why Imagepool is used / Need to understand

        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError(
                    "Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter()

            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            self.criterionFeat = torch.nn.L1Loss()
            self.criterionCE = torch.nn.CrossEntropyLoss()

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter(
                'G_GAN_REF', 'G_GAN_QUERY', 'G_VGG')

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(
                params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(
                params, lr=opt.lr, betas=(opt.beta1, 0.999))
    
    #################### DISCRIMINATOR HELPER FUNCTION ####################
    # def discriminate(self, input_label, test_image, use_pool=False):
    #     input_concat = torch.cat((input_label, test_image.detach()), dim=1)
    #     if use_pool:
    #         fake_query = self.fake_pool.query(input_concat)
    #         with torch.no_grad():
    #             discriminator_output = self.netD.forward(fake_query)
    #         return discriminator_output
    #     else:
    #         with torch.no_grad():
    #             discriminator_output = self.netD.forward(fake_query)
    #         return discriminator_output

    #################### FILTER LOSSES AS NEEDED ####################
    def init_loss_filter(self):
        flags = (True, True, True)

        def loss_filter(g_gan_ref, g_gan_query, g_vgg):
            return [l for (l, f) in zip((g_gan_ref, g_gan_query, g_vgg), flags) if f]
        return loss_filter

    #################### THE MAIN FORWARD FUNCTION ####################
    def forward(self, query_img, query_parse_map, query_seg_map, ref_img, ref_parse_map, ref_seg_map, C_tensor, infer=False):

        #################### Define Inputs ####################
        query_img = query_img.float().cuda()
        query_parse_map = query_parse_map.float().cuda()
        query_seg_map = query_seg_map.float().cuda()
        ref_img = ref_img.float().cuda()
        ref_parse_map = ref_parse_map.float().cuda()
        ref_seg_map = ref_seg_map.float().cuda()
        generated_parse_map = C_tensor.float().cuda() # This is the segmentation parse map generated by the shape generation section

        #################### e^tm (appearance feature map) ####################
        # Query Image
        app_feature_map = torch.zeros((1, 30, 512, 256)).float().cuda()
        selected_seg_mask_tensor = query_seg_map
        selected_seg_mask_tensor = torch.unsqueeze(selected_seg_mask_tensor, 0)
        selected_img_tensor = query_img
        selected_seg_parse_map = query_parse_map
        input_encoder = torch.cat(
            (selected_img_tensor, selected_seg_parse_map), 1).cuda()
        with torch.no_grad():
            y1 = self.netE.forward(input_encoder)

        # Ref Image
        app_feature_map_ref = torch.zeros((1, 30, 512, 256)).float().cuda()
        selected_seg_mask_tensor = ref_seg_map
        selected_seg_mask_tensor = torch.unsqueeze(selected_seg_mask_tensor, 0)
        selected_img_tensor = ref_img
        selected_seg_parse_map = ref_parse_map
        input_encoder = torch.cat(
            (selected_img_tensor, selected_seg_parse_map), 1).cuda()
        with torch.no_grad():
            y2 = self.netE.forward(input_encoder)

        # Find appearance feature map for query ref mix 
        for num_seg_channel in range(20):
            if 4 < num_seg_channel < 8:
                selected_seg_mask_tensor = torch.unsqueeze(ref_seg_map, 0)
                app_feature_vec_temp = y2
            else:
                selected_seg_mask_tensor = torch.unsqueeze(query_seg_map, 0)
                app_feature_vec_temp = y1

            indices = (selected_seg_mask_tensor == int(
                num_seg_channel)).nonzero()  # nx4

            for enc_channel in range(30):
                region_of_interest = app_feature_vec_temp[indices[:, 0],
                                                          indices[:, 1] + enc_channel, indices[:, 2], indices[:, 3]]
                enc_each_channel_mean = torch.mean(
                    region_of_interest).expand_as(region_of_interest)
                app_feature_map[indices[:, 0], indices[:, 1] + enc_channel,
                                indices[:, 2], indices[:, 3]] = enc_each_channel_mean

        # Find appearance feature map for ref image
        app_feature_vec_temp = y2.clone()
        selected_seg_mask_tensor = torch.unsqueeze(ref_seg_map, 0)
        for num_seg_channel in range(20):
            indices = (selected_seg_mask_tensor == int(num_seg_channel)).nonzero() # nx4
            for enc_channel in range(30):
                region_of_interest = app_feature_vec_temp[indices[:,0], indices[:,1] + enc_channel, indices[:,2], indices[:,3]]
                enc_each_channel_mean = torch.mean(region_of_interest).expand_as(region_of_interest) 
                app_feature_map_ref[indices[:,0], indices[:,1] + enc_channel, indices[:,2], indices[:,3]] = enc_each_channel_mean

        #################### Online loss calculation ####################
        '''
            online loss has 2 component Lref and Lquery
        '''
        #################### Lref calculation ####################

        ref_parse_map_online = torch.zeros((1, 20, 512, 256)).float().cuda()
        for num_seg_channel in range(20):
            if 4 < num_seg_channel < 8:
                ref_parse_map_online[:, num_seg_channel, :,
                                     :] = ref_parse_map[:, num_seg_channel, :, :]

        # localization of loss using binary mask filter    
        ref_parse_map_online = (ref_parse_map_online + 1)/2
        filter_part = ref_parse_map_online[:, 5, :, :] + ref_parse_map_online[:,6, :, :] + ref_parse_map_online[:, 7, :, :]  # 5,6,7,
        filter_part = filter_part > 0
        filter_part = torch.unsqueeze(filter_part, 0).float().cuda()

        input_concat = torch.cat(
            (ref_parse_map_online, app_feature_map_ref), dim=1).cuda()

        fake_image = self.netG.forward(input_concat)
        fake_image = fake_image * filter_part

        # GAN loss (Fake Passability Loss)
        with torch.no_grad():
            pred_fake = self.netD.forward(torch.cat((ref_parse_map_online, fake_image), dim=1))
        loss_G_GAN_ref = self.criterionGAN(pred_fake, True)

        # VGG feature matching loss
        target = (ref_img + 1)/2
        target = target * filter_part
        norm = transforms.Normalize((0.5,), (0.5,))
        target = norm(torch.squeeze(target))
        target = torch.unsqueeze(target,0)

        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(
                fake_image, target) * self.opt.lambda_feat

        #################### Lquery calculation ####################
        # generated_parse_map_online = torch.zeros((1, 20, 512, 256)).float().cuda()
        # for num_seg_channel in range(20):
        #     if 4 < num_seg_channel < 8:
        #         generated_parse_map_online[:, num_seg_channel, :,
        #                              :] = generated_parse_map[:, num_seg_channel, :, :]

        # # localization of loss using binary mask filter    
        # generated_parse_map_online = (generated_parse_map_online + 1)/2
        # filter_part = generated_parse_map_online[:, 5, :, :] + generated_parse_map_online[:,6, :, :] + generated_parse_map_online[:, 7, :, :]  # 5,6,7,
        # filter_part = filter_part > 0
        # filter_part = torch.unsqueeze(filter_part, 0).float().cuda()

        input_concat = torch.cat(
            (generated_parse_map, app_feature_map), dim=1).cuda()

        fake_image = self.netG.forward(input_concat)
        fake_image_org = fake_image.clone()
        # fake_image = fake_image * filter_part

        # GAN loss (Fake Passability Loss)
        with torch.no_grad():
            pred_fake = self.netD.forward(torch.cat((generated_parse_map, fake_image_org), dim=1))
        loss_G_GAN_query = self.criterionGAN(pred_fake, True)


        return [self.loss_filter(loss_G_GAN_ref, loss_G_GAN_query, loss_G_VGG), None if not infer else fake_image_org]

    #################### UPDATE LEARNING RATE ####################
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    #################### SAVE NETWORKS ####################  
    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)


