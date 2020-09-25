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
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, ), (0.5, ))])  # change to [C, H, W]
augment['1'] = transforms.Compose(
    [transforms.ToTensor()])  # change to [C, H, W]

test_dataset = TestDataset(opt, augment)
test_dataloader = DataLoader(test_dataset,
                             shuffle=False,
                             num_workers=int(opt.nThreads),
                             batch_size=opt.batchSize,
                             pin_memory=True)
dataset_size = len(test_dataset)
print('#testing images = %d' % dataset_size)

for key in test_dataset[0].keys():
    try:
        x = test_dataset[0][key]
        print("name of the input and shape -- > ", key, x.shape)
        print("type,dtype,and min max -- >", type(x),
              x.dtype, torch.min(x), torch.max(x))
    except Exception as e:
        print("name of the input -- > ", key, test_dataset[0][key])
    print('----------------')

# Create and Load Model
model = create_model(opt)

# Initialize visualizer
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' %
                       (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
    opt.name, opt.phase, opt.which_epoch))
for i, data in enumerate(test_dataloader):
    if i >= opt.how_many:
        break
    query_ref_mixed, generated = model.inference_enc(data['query'], data['dense_map'],
                                                     data['ref'], cloth_part= opt.cloth_part)
    visuals = OrderedDict([('query', util.tensor2label(data['query'][0], opt.label_nc)),
                           ('ref', util.tensor2label(
                               data['ref'][0], opt.label_nc)),
                           ('query_ref_mixed', util.tensor2label(
                               query_ref_mixed.data[0], opt.label_nc)),
                           ('synthesized_image', util.tensor2label(
                               generated.data[0], opt.label_nc)),
                           ('synthesized_image_edgemap', util.tensor2edgemap(torch.softmax(generated.data[0], dim=0)))])
    img_path = data['query_path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
