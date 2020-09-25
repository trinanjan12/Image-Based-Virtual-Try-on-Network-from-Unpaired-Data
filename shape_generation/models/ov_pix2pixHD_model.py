import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.nn.modules.upsampling import Upsample


########################################
'''
    Generator network 
        input_nc = label_nc(20) +  27 for densepose
        output_nc = 20 --> segmentation output

    Discriminator network
        input_nc = label_nc(20) + output_nc(20)
        output_nc = opt.ndf(64) patchgan #https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py#L32
    Encoder Network
        input_nc = 1 channel binary mask for each class
        output_nc = 10(encoder embedding dimension for each class)

    ** since we are using categorical cross entropy loss remove tanh from GlobalGenerator class output
    ** add explanation for the above assumption
    
    run the code with python train_new.py --name name_of_exp --dataroot ./datasets/dataroot/ --tf_log

'''
########################################


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain  # Train/Test mode
        self.opt = opt  # options

        #################### DEFINE NETWORKS ####################

        '''
            Initialize Generator Network
        '''

        # (10*20) embeding for all the class
        netG_input_nc = opt.feat_num * opt.label_nc
        netG_input_nc += opt.densepose_nc  # 27 channel for densepose
        netG_output_nc = opt.output_nc  # 20 segmentation class

        self.netG = networks.define_G(netG_input_nc, netG_output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        '''
            Initialize Discriminator Network
        '''

        # if self.isTrain:
        use_sigmoid = opt.no_lsgan
        netD_input_nc = opt.label_nc + opt.output_nc  # (20 + 20)
        # self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
        #                                 opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                      opt.num_D, False, gpu_ids=self.gpu_ids)

        '''
            Initialize Encoder Network
        '''

        netE_input_nc = 1  # 1 channel binary mask for each class
        # embeding dim(10) for each segmentation class
        netE_output_nc = opt.feat_num
        self.netE = networks.define_G(netE_input_nc, netE_output_nc, opt.nef, 'encoder',
                                      opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        print('netG_input_nc -- 227, netG_output_nc -- 20, netD_input_nc -- 40, netD_output_nc',
              netG_input_nc, opt.output_nc, netD_input_nc, opt.ndf)
        print('netE_input_nc, netE_output_nc ', netE_input_nc, netE_output_nc)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')
        #################### LOAD NETWORKS ####################

        if not self.isTrain or opt.continue_train or opt.load_pretrain:

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

            # Define loss functions
            self.loss_filter = self.init_loss_filter()

            self.criterionGAN = networks.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            self.criterionCE = torch.nn.CrossEntropyLoss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter(
                'G_GAN', 'G_CE', 'D_real', 'D_fake')

            # Initialize optimizers
            # Optimizer G
            params = list(self.netG.parameters())
            params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(
                params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(
                params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    # Filter and return losses as needed
    def init_loss_filter(self):
        flags = (True, True, True, True)

        def loss_filter(g_gan, g_ce, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_ce, d_real, d_fake), flags) if f]

        return loss_filter

    # Forward function for the entire network
    def forward(self, seg_map, dense_map, target, seg_mask, infer=False):
        seg_map = seg_map.float().cuda()
        dense_map = dense_map.float().cuda()
        target = target.float().cuda()

        feat_map_total = []
        for each_class in range(self.opt.label_nc):
            inp_enc = seg_map[:, each_class:each_class+1, :, :]
            feat_map_each_class = self.netE.forward(
                inp_enc)  # bs, 10, H, w
            feat_map_total.append(feat_map_each_class)
        feat_map_total = torch.cat([i for i in feat_map_total], dim=1)

        # local pooling step and Upscaling
        local_avg_pool_fn = nn.AvgPool2d((64, 64))
        feat_map_each_class_pooled = local_avg_pool_fn(feat_map_total)
        upscale_fn = Upsample(scale_factor=64, mode='nearest')
        feat_map_final = upscale_fn(feat_map_each_class_pooled)

        # Gan Input
        input_concat = torch.cat((dense_map, feat_map_final), dim=1).cuda()
        fake_image = self.netG.forward(input_concat)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(
            seg_map, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(seg_map, target)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(
            torch.cat((seg_map, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        ###############################
        # Crossentropy loss
        loss_G_CE = 0
        loss_G_CE = self.criterionCE(fake_image, seg_mask)

        return [self.loss_filter(loss_G_GAN, loss_G_CE, loss_D_real, loss_D_fake), None if not infer else fake_image]

    # Update Learning rate function
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

    # Save model function
    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    # Inference function
    def inference_enc(self, query, dense_map, ref, cloth_part='uppercloth'):
        query = query.float().cuda()
        dense_map = dense_map.float().cuda()
        ref = ref.float().cuda()

        # Cloth part to mix
        if cloth_part == 'uppercloth':
            query_ref_mixed = torch.cat(
                (query[:, 0:5, :, :], ref[:, 5:8, :, :], query[:, 8:, :, :]), axis=1)

        elif cloth_part == 'bottomcloth':
            query_ref_mixed = torch.cat((query[:, 0:9, :, :], ref[:, 9:10, :, :], query[:, 10:12, :, :],
                                         ref[:, 12:13, :, :], query[:, 13:16, :, :], ref[:, 16:20, :, :]), axis=1)
        # Encoder
        feat_map_total = []
        for each_class in range(self.opt.label_nc):
            inp_enc = query_ref_mixed[:, each_class:each_class+1, :, :]
            with torch.no_grad():
                feat_map_each_class = self.netE.forward(
                    inp_enc)  # bs, 10, H, w
            feat_map_total.append(feat_map_each_class)
        feat_map_total = torch.cat([i for i in feat_map_total], dim=1)

        # Local pooling step and Upscaling
        local_avg_pool_fn = nn.AvgPool2d((64, 64))
        feat_map_each_class_pooled = local_avg_pool_fn(feat_map_total)
        upscale_fn = Upsample(scale_factor=64, mode='nearest')
        feat_map_final = upscale_fn(feat_map_each_class_pooled)

        # GAN
        input_concat = torch.cat((dense_map, feat_map_final), dim=1)
        with torch.no_grad():
            fake_image = self.netG.forward(input_concat)

        return query_ref_mixed, fake_image

# Inference class


class InferenceModel(Pix2PixHDModel):
    def forward(self, query, dense_map, ref, cloth_part):
        return self.inference_enc(query, dense_map, ref, cloth_part)
