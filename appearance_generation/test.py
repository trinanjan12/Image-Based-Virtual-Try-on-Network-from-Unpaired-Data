'''
    run the code with python test.py --name name_of_exp --dataroot ./datasets/dataroot/ 
'''
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.ov_test_dataset import TestDataset
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

augment = {}
# augment['1'] = transforms.Compose(
#     [transforms.ToTensor()])  # change to [C, H, W]

augment['1'] = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))])

test_dataset = TestDataset(opt, augment)
test_dataloader = DataLoader(test_dataset,
                             shuffle=False,
                             num_workers=int(opt.nThreads),
                             batch_size=opt.batchSize,
                             pin_memory=True)

dataset_size = len(test_dataset)
print('#testing images = %d' % dataset_size)

# Create and Load Model
model = create_model(opt)

# Initialize visualizer
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
for i, data in enumerate(test_dataloader):
    if i >= opt.how_many:
        break
    generated = model.inference_forward_appearance(data['query_img'],data['query_parse_map'],
                                                              data['query_seg_map'],data['ref_img'],
                                                              data['ref_parse_map'],data['ref_seg_map'],
                                                              data['C_tensor_parse_map'],data['C_tensor_seg_map'])
    visuals = OrderedDict([('query_img', util.tensor2im(data['query_img'][0])),
                           ('ref_image', util.tensor2im(data['ref_img'][0])),
                           ('generated_parse_map', util.tensor2label(data['C_tensor_parse_map'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
