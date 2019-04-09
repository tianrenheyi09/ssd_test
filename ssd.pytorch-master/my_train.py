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

cfg = voc
dataset = VOCDetection(root='data/',
                       transform=SSDAugmentation(cfg['min_dim'],
                                                 MEANS))

ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net

vgg_weights = torch.load('weights/'+'vgg16_reducedfc.pth')

print('Loading base network...')
net.vgg.load_state_dict(vgg_weights)


net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=5e-4)

criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                         False, False)

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


# loa train data
images, targets = next(batch_iterator)

###################
import matplotlib.pyplot as plt
# img1 = cv2.imread('data/VOC2007/JPEGImages/000005.jpg')
# cv2.imshow('ii',img1)
#
# plt.imshow(img1)
#
# from skimage import io,data
# a = images.squeeze(0)
# a = a.view(300,300,3).type(torch.ByteTensor)
# a = a.numpy()
# # cv2.imshow('a',a.numpy())
# plt.imshow(a)

# images = Variable(images)
# targets = [Variable(ann, volatile=True) for ann in targets]

# forward
images = images.to(device)
targets = [ann.to(device) for ann in targets]
t0 = time.time()
out = net(images)

loc_data, conf_data, priors = out
num = loc_data.size(0)
priors = priors[:loc_data.size(1), :]
num_priors = (priors.size(0))
num_classes = 21



from layers.box_utils import match, log_sum_exp,point_form,center_size,intersect,jaccard,encode,decode,nms
from data import coco as cfg

loc_t = torch.Tensor(num, num_priors, 4)
conf_t = torch.LongTensor(num, num_priors)
MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                         False, False)
for idx in range(num):
    truths = targets[idx][:, :-1].data
    labels = targets[idx][:, -1].data
    defaults = priors.data
    match(0.5, truths, defaults, cfg['variance'], labels,
          loc_t, conf_t, idx)

# wrap targets
loc_t = Variable(loc_t, requires_grad=False)
conf_t = Variable(conf_t, requires_grad=False)

pos = conf_t > 0

num_pos = pos.sum(dim=1, keepdim=True)

# Localization Loss (Smooth L1)
# Shape: [batch,num_priors,4]
pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
loc_p = loc_data[pos_idx].view(-1, 4)
loc_t = loc_t[pos_idx].view(-1, 4)
loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)


# Compute max conf across batch for hard negative mining
batch_conf = conf_data.view(-1, 21)
loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))#########gathers函数去axis=1轴上conf_t对应数字的那个数

# Hard Negative Mining
loss_c[pos.view(loss_c.shape)] = 0  # filter out pos boxes for now
loss_c = loss_c.view(num, -1)################zhe这个loss_c主要是位下面分数排序选择负样本做准备
_, loss_idx = loss_c.sort(1, descending=True)
_, idx_rank = loss_idx.sort(1)
num_pos = pos.long().sum(1, keepdim=True)
num_neg = torch.clamp(3*num_pos, max=pos.size(1)-1)
neg = idx_rank < num_neg.expand_as(idx_rank)

# Confidence Loss Including Positive and Negative Examples
pos_idx = pos.unsqueeze(2).expand_as(conf_data)
neg_idx = neg.unsqueeze(2).expand_as(conf_data)
conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,21)
targets_weighted = conf_t[(pos+neg).gt(0)]
loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

# Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

N = num_pos.data.sum()
loss_l /= N
loss_c /= N

# backprop
optimizer.zero_grad()
loss_l, loss_c = criterion(out, targets)







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

    # images = Variable(images)
    # targets = [Variable(ann, volatile=True) for ann in targets]

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
    loc_loss += loss_l.data[0]
    conf_loss += loss_c.data[0]
    if iteration % 10 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')

    if iteration != 0 and iteration % 5000 == 0:
        print('Saving state, iter:', iteration)
        torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                   repr(iteration) + '.pth')







#
#
#
#
#
#
# def train():
#     if args.dataset == 'COCO':
#         if args.dataset_root == VOC_ROOT:
#             if not os.path.exists(COCO_ROOT):
#                 parser.error('Must specify dataset_root if specifying dataset')
#             print("WARNING: Using default COCO dataset_root because " +
#                   "--dataset_root was not specified.")
#             args.dataset_root = COCO_ROOT
#         cfg = coco
#         dataset = COCODetection(root=args.dataset_root,
#                                 transform=SSDAugmentation(cfg['min_dim'],
#                                                           MEANS))
#     elif args.dataset == 'VOC':
#         if args.dataset_root == COCO_ROOT:
#             parser.error('Must specify dataset if specifying dataset_root')
#         cfg = voc
#         dataset = VOCDetection(root=args.dataset_root,
#                                transform=SSDAugmentation(cfg['min_dim'],
#                                                          MEANS))
#
#     if args.visdom:
#         import visdom
#         viz = visdom.Visdom()
#
#     ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
#     net = ssd_net
#
#     if args.cuda:
#         net = torch.nn.DataParallel(ssd_net)
#         cudnn.benchmark = True
#
#     if args.resume:
#         print('Resuming training, loading {}...'.format(args.resume))
#         ssd_net.load_weights(args.resume)
#     else:
#         vgg_weights = torch.load(args.save_folder + args.basenet)
#         print('Loading base network...')
#         ssd_net.vgg.load_state_dict(vgg_weights)
#
#     if args.cuda:
#         net = net.cuda()
#
#     if not args.resume:
#         print('Initializing weights...')
#         # initialize newly added layers' weights with xavier method
#         ssd_net.extras.apply(weights_init)
#         ssd_net.loc.apply(weights_init)
#         ssd_net.conf.apply(weights_init)
#
#     optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
#                           weight_decay=args.weight_decay)
#     criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
#                              False, args.cuda)
#
#     net.train()
#     # loss counters
#     loc_loss = 0
#     conf_loss = 0
#     epoch = 0
#     print('Loading the dataset...')
#
#     epoch_size = len(dataset) // args.batch_size
#     print('Training SSD on:', dataset.name)
#     print('Using the specified args:')
#     print(args)
#
#     step_index = 0
#
#     if args.visdom:
#         vis_title = 'SSD.PyTorch on ' + dataset.name
#         vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
#         iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
#         epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
#
#     data_loader = data.DataLoader(dataset, args.batch_size,
#                                   num_workers=args.num_workers,
#                                   shuffle=True, collate_fn=detection_collate,
#                                   pin_memory=True)
#     # create batch iterator
#     batch_iterator = iter(data_loader)
#     for iteration in range(args.start_iter, cfg['max_iter']):
#         if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
#             update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
#                             'append', epoch_size)
#             # reset epoch loss counters
#             loc_loss = 0
#             conf_loss = 0
#             epoch += 1
#
#         if iteration in cfg['lr_steps']:
#             step_index += 1
#             adjust_learning_rate(optimizer, args.gamma, step_index)
#
#         # load train data
#         images, targets = next(batch_iterator)
#
#         if args.cuda:
#             images = Variable(images.cuda())
#             targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
#         else:
#             images = Variable(images)
#             targets = [Variable(ann, volatile=True) for ann in targets]
#         # forward
#         t0 = time.time()
#         out = net(images)
#         # backprop
#         optimizer.zero_grad()
#         loss_l, loss_c = criterion(out, targets)
#         loss = loss_l + loss_c
#         loss.backward()
#         optimizer.step()
#         t1 = time.time()
#         loc_loss += loss_l.data[0]
#         conf_loss += loss_c.data[0]
#
#         if iteration % 10 == 0:
#             print('timer: %.4f sec.' % (t1 - t0))
#             print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
#
#         if args.visdom:
#             update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
#                             iter_plot, epoch_plot, 'append')
#
#         if iteration != 0 and iteration % 5000 == 0:
#             print('Saving state, iter:', iteration)
#             torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
#                        repr(iteration) + '.pth')
#     torch.save(ssd_net.state_dict(),
#                args.save_folder + '' + args.dataset + '.pth')
#
#
# def adjust_learning_rate(optimizer, gamma, step):
#     """Sets the learning rate to the initial LR decayed by 10 at every
#         specified step
#     # Adapted from PyTorch Imagenet example:
#     # https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     lr = args.lr * (gamma ** (step))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# def xavier(param):
#     init.xavier_uniform(param)
#
#
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         xavier(m.weight.data)
#         m.bias.data.zero_()
#
# #
# # def create_vis_plot(_xlabel, _ylabel, _title, _legend):
# #     return viz.line(
# #         X=torch.zeros((1,)).cpu(),
# #         Y=torch.zeros((1, 3)).cpu(),
# #         opts=dict(
# #             xlabel=_xlabel,
# #             ylabel=_ylabel,
# #             title=_title,
# #             legend=_legend
# #         )
# #     )
# #
# #
# # def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
# #                     epoch_size=1):
# #     viz.line(
# #         X=torch.ones((1, 3)).cpu() * iteration,
# #         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
# #         win=window1,
# #         update=update_type
# #     )
# #     # initialize epoch plot on first iteration
# #     if iteration == 0:
# #         viz.line(
# #             X=torch.zeros((1, 3)).cpu(),
# #             Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
# #             win=window2,
# #             update=True
# #         )
#
#
# if __name__ == '__main__':
#     train()
