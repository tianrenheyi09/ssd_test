import os
import sys
module_path = os.path.abspath(os.path.join('...'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

net = build_ssd('train', 300, 21)    # initialize SSD
net.load_weights('weights/ssd300_mAP_77.43_v2.pth')

# image = cv2.imread('./data/example.jpg', cv2.IMREAD_COLOR)  # uncomment if dataset not downloaded

from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train')
testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
img_id = 60
image = testset.pull_image(img_id)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# View the sampled input image before transform
plt.figure(figsize=(10,10))
plt.imshow(rgb_image)
plt.show()


x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
plt.imshow(x)
x = torch.from_numpy(x).permute(2, 0, 1)


xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)

'''
#####################--------------dection函数进行测试
from layers.box_utils import decode, nms,his_nms
from data import voc as cfg
import torch.nn as nn
loc_data = y[0]
conf_data = nn.Softmax(dim=-1)(y[1])
prior_data = y[2]
num_classes = 21
top_k  =200
conf_thresh = 0.01
nms_thresh = 0.45
################################
from layers import *
out = Detect(num_classes, 0, 200, 0.01, 0.45)(loc_data,conf_data,prior_data)

#############################3----------------------测试nms


num = loc_data.size(0)  # batch size
num_priors = prior_data.size(0)
output = torch.zeros(num, num_classes, top_k, 5)
conf_preds = conf_data.view(num, num_priors,
                            num_classes).transpose(2, 1)

decoded_boxes = decode(loc_data[0], prior_data, cfg['variance'])
# For each class, perform nms
conf_scores = conf_preds[0].clone()

c_mask = conf_scores[16].gt(conf_thresh)
scores = conf_scores[16][c_mask]

l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
boxes = decoded_boxes[l_mask].view(-1, 4)
ids, count = nms(boxes.detach(), scores.detach(), 0.45, 200)#############必须加上detach否则会带入梯度

output[0, 16, :count] = \
            torch.cat((scores[ids[:count]].unsqueeze(1),
                       boxes[ids[:count]]), 1)



# Decode predictions into bboxes.
for i in range(num):
    decoded_boxes = decode(loc_data[i], prior_data, cfg['variance'] )
    # For each class, perform nms
    conf_scores = conf_preds[i].clone()

    for cl in range(1, num_classes):
        c_mask = conf_scores[cl].gt(conf_thresh)
        scores = conf_scores[cl][c_mask]
        if scores.size(0) == 0:
            continue
        l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
        boxes = decoded_boxes[l_mask].view(-1, 4)
        # idx of highest scoring and non-overlapping boxes per class
        ids, count = nms(boxes.detach(), scores.detach(), nms_thresh, top_k)
        output[i, cl, :count] = \
            torch.cat((scores[ids[:count]].unsqueeze(1),
                       boxes[ids[:count]]), 1)
flt = output.contiguous().view(num, -1, 5)
_, idx = flt[:, :, 0].sort(1, descending=True)
_, rank = idx.sort(1)
flt[(rank < top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
#
'''

from data import VOC_CLASSES as labels
top_k=10
plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data##########train模式下使用output,test模式下直接使用y
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1