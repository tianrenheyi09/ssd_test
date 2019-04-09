from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import torch.nn.functional as F

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


lr = 0.001

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 0.001 * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if device == 'cuda' else False

# if torch.cuda.is_available():
#     if cuda:
#         torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     if not cuda:
#         print("WARNING: It looks like you have a CUDA device, but aren't " +
#               "using CUDA.\nRun with --cuda for optimal training speed.")
#         torch.set_default_tensor_type('torch.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')


cfg = voc
dataset = VOCDetection(root='data/',
                       transform=SSDAugmentation(cfg['min_dim'],
                                                 MEANS))


ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net




if cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True


vgg_weights = torch.load('weights/'+'vgg16_reducedfc.pth')

print('Loading base network...')
net.vgg.load_state_dict(vgg_weights)


net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=5e-4)

criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                         False,cuda)


net.train()
# loss counters
loc_loss = 0
conf_loss = 0
epoch = 0
print('Loading the dataset...')

batch_size = 1
epoch_size = len(dataset) // batch_size
print('Training SSD on:', dataset.name)
print('Using the specified args:')

step_index = 0
num_workers = 4
start_iter = 0
gamma = 0.1

data_loader = data.DataLoader(dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)
# create batch iterator
batch_iterator = iter(data_loader)




for iteration in range(start_iter, cfg['max_iter']):
        # reset epoch loss counters
    loc_loss = 0
    conf_loss = 0
    epoch += 1

    if iteration in cfg['lr_steps']:
        step_index += 1
        adjust_learning_rate(optimizer, gamma, step_index)
    # load train data
    images, targets = next(batch_iterator)

    # forward
    images = images.to(device)
    targets = [ann.to(device) for ann in targets]
    t0 = time.time()
    out = net(images)
    # backprop
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()
    loc_loss += loss_l.item()
    conf_loss += loss_c.item()
    if iteration % 1 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + str(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

    if iteration != 0 and iteration % 5000 == 0:
        print('Saving state, iter:', iteration)
        torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                   repr(iteration) + '.pth')




