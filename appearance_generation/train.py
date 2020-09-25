'''
    run the code with python train.py --name name_of_exp --dataroot ./datasets/dataroot/ --tf_log
'''
from options.train_options import TrainOptions
from util.visualizer import Visualizer
import util.util as util
from models.models import create_model
from data.ov_train_dataset import RegularDataset
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from subprocess import call
import fractions
from torch.utils.tensorboard import SummaryWriter


def lcm(a, b): return abs(a * b)/fractions.gcd(a, b) if a and b else 0


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(
            iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# NEW DATALOADER
augment = {}
augment['1'] = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))])

# augment['1'] = transforms.Compose(
#     [
#         transforms.ToTensor()])

train_dataset = RegularDataset(opt, augment)

train_dataloader = DataLoader(train_dataset,
                              batch_size=opt.batchSize,
                              shuffle=True,
                              num_workers=int(opt.nThreads),
                              pin_memory=True)

# FOR DEBUGGING
print(" #Checking  the dimension and type of data")
for key in train_dataset[0].keys():
    try:
        x = train_dataset[0][key]
        print("name of the input and shape -- > ", key, x.shape)
        print("type,dtype,and min max -- >", type(x),
              x.dtype, torch.min(x), torch.max(x))
    except Exception as e:
        print("name of the input -- > ", key, train_dataset[0][key])
    print('----------------')

dataset_size = len(train_dataset)
print('#training images = %d' % dataset_size)

# Initialize Networks
model = create_model(opt)

# Training Visualizer
visualizer = Visualizer(opt)

# Optimizers
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

# Train related additional details
total_steps = (start_epoch-1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(train_dataloader, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        # Reference Purpose
        # input_dict = {'seg_map': A_tensor, 'dense_map': dense_img, 'target': B_tensor, 'seg_map_path': A_path,
        # 'target_path': A_path, 'densepose_path': dense_path }
        # print( data['seg_mask'].shape)
        losses, generated = model(
            data['seg_map'], data['target'], data['seg_mask'], infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int)
                  else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_VGG', 0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ############## Display results and errors ##########
        # print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(
                v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

        # display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['seg_map'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(
                                       generated.data[0])),
                                   ('real_image', util.tensor2im(data['target'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

#     ### instead of only training the local enhancer, train the entire network after certain iterations
#     if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
#         model.module.update_fixed_params()

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
