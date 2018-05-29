import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image

import sys
import os 
import time
import numpy as np
import cv2
import argparse
import yaml
import json
import random
import math
import copy
from tqdm import tqdm
from easydict import EasyDict as edict

parser = argparse.ArgumentParser(description='Training code')
parser.add_argument('--config', default='config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
CONFIG = edict(yaml.load(open(args.config, 'r')))
print ('==> CONFIG is: \n', CONFIG, '\n')

LOGDIR = '%s/%s_%d'%(CONFIG.LOGS.LOG_DIR, CONFIG.NAME, int(time.time()))
SNAPSHOTDIR = '%s/%s_%d'%(CONFIG.LOGS.SNAPSHOT_DIR, CONFIG.NAME, int(time.time()))
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists(SNAPSHOTDIR):
    os.makedirs(SNAPSHOTDIR)

def to_varabile(arr, requires_grad=False, is_cuda=True):
    if type(arr) == np.ndarray:
        tensor = torch.from_numpy(arr)
    else:
        tensor = arr
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

MEAN_var = to_varabile(np.array(CONFIG.DATASET.MEAN, dtype=np.float32)[:,np.newaxis,np.newaxis], requires_grad=False, is_cuda=True)
        
######################################################################################################################
#                           "Globally and Locally Consistent Image Completion" Model
######################################################################################################################

def AffineAlignOp(features, idxs, aligned_height, aligned_width, Hs):
    def _transform_matrix(Hs, w, h):
        _Hs = np.zeros(Hs.shape, dtype = np.float32)
        for i, H in enumerate(Hs):
            H0 = np.concatenate((H, np.array([[0, 0, 1]])), axis=0)
            A = np.array([[2.0 / w, 0, -1], [0, 2.0 / h, -1], [0, 0, 1]])
            A_inv = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h/ 2.0], [0, 0, 1]])
            H0 = A.dot(H0).dot(A_inv)
            H0 = np.linalg.inv(H0)
            _Hs[i] = H0[:-1]
        return _Hs
    bz, C_feat, H_feat, W_feat = features.size()
    N = len(idxs)
    feature_select = features[idxs] # (N, feature_channel, feature_size, feature_size)
    Hs_new = _transform_matrix(Hs, w=W_feat, h=H_feat) # return (N, 2, 3)
    Hs_var = Variable(torch.from_numpy(Hs_new), requires_grad=False).cuda()
    flow = F.affine_grid(theta=Hs_var, size=(N, C_feat, H_feat, W_feat)).float().cuda()
    flow = flow[:,:aligned_height, :aligned_width, :]
    rois = F.grid_sample(feature_select, flow, mode='bilinear', padding_mode='border') # 'zeros' | 'border' 
    return rois
    
def CropAlignOp(feature_var, rois_var, aligned_height, aligned_width, spatial_scale):
    rois_np = rois_var.data.cpu().numpy()
    #idxs = rois_np[:,0]
    affinematrixs_feat = []
    for roi in rois_np:
        #x1, y1, x2, y2 = roi[1:] * float(spatial_scale)
        x1, y1, x2, y2 = roi * float(spatial_scale)
        matrix = np.array([[aligned_width/(x2-x1), 0, -aligned_width/(x2-x1)*x1],
                           [0, aligned_height/(y2-y1), -aligned_height/(y2-y1)*y1]
                          ])
        affinematrixs_feat.append(matrix)
    affinematrixs_feat = np.array(affinematrixs_feat)
    feature_rois = AffineAlignOp(feature_var, np.array(range(rois_var.size(0))), 
                                 aligned_height, aligned_width, affinematrixs_feat)
    return feature_rois


class ConvBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim, 
                 kernel_size=3, stride=1, dilation=1, group=1,
                 bias = True, bn = True, relu = True):
        super(ConvBnRelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, (kernel_size-1)//2+(dilation-1), dilation, group, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class DeconvBnRelu(nn.Module):
    def __init__(self, inp_dim, out_dim, 
                 kernel_size=4, stride=2,
                 bias = True, bn = True, relu = True):
        super(DeconvBnRelu, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.ConvTranspose2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class GLCIC_G(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_G, self).__init__()
        self.conv1_1 = ConvBnRelu(4, 64, kernel_size=5, stride=1, bias=bias_in_conv)
        self.conv1_2 = ConvBnRelu(64, 128, kernel_size=3, stride=2, bias=bias_in_conv)
        self.conv1_3 = ConvBnRelu(128, 128, kernel_size=3, stride=1, bias=bias_in_conv)
        
        self.conv2_1 = ConvBnRelu(128, 256, kernel_size=3, stride=2, bias=bias_in_conv)
        self.conv2_2 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)
        self.conv2_3 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)
        
        self.conv3_1 = ConvBnRelu(256, 256, kernel_size=3, dilation=2, stride=1, bias=bias_in_conv)
        self.conv3_2 = ConvBnRelu(256, 256, kernel_size=3, dilation=4, stride=1, bias=bias_in_conv)
        self.conv3_3 = ConvBnRelu(256, 256, kernel_size=3, dilation=8, stride=1, bias=bias_in_conv)
        self.conv3_4 = ConvBnRelu(256, 256, kernel_size=3, dilation=16, stride=1, bias=bias_in_conv)
        
        self.conv4_1 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)
        self.conv4_2 = ConvBnRelu(256, 256, kernel_size=3, stride=1, bias=bias_in_conv)
        
        self.decoder1_1 = DeconvBnRelu(256, 128, kernel_size=4, stride=2, bias=bias_in_conv)
        self.decoder1_2 = ConvBnRelu(128, 128, kernel_size=3, stride=1, bias=bias_in_conv)
        
        self.decoder2_1 = DeconvBnRelu(128, 64, kernel_size=4, stride=2, bias=bias_in_conv)
        self.decoder2_2 = ConvBnRelu(64, 32, kernel_size=3, stride=1, bias=bias_in_conv)
        self.decoder2_3 = ConvBnRelu(32, 3, kernel_size=3, stride=1, bias=bias_in_conv, bn = False, relu = False)
        
        self.init(pretrainfile)
        
    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()
                    
        elif 'completionnet_places2.t7' in pretrainfile:
            mapping = {'conv1_1.conv': 0, 'conv1_1.bn': 1, 'conv1_2.conv': 3, 'conv1_2.bn': 4, 'conv1_3.conv': 6, 'conv1_3.bn': 7, 'conv2_1.conv': 9, 'conv2_1.bn': 10, 'conv2_2.conv': 12, 'conv2_2.bn': 13, 'conv2_3.conv': 15, 'conv2_3.bn': 16, 'conv3_1.conv': 18, 'conv3_1.bn': 19, 'conv3_2.conv': 21, 'conv3_2.bn': 22, 'conv3_3.conv': 24, 'conv3_3.bn': 25, 'conv3_4.conv': 27, 'conv3_4.bn': 28, 'conv4_1.conv': 30, 'conv4_1.bn': 31, 'conv4_2.conv': 33, 'conv4_2.bn': 34, 'decoder1_1.conv': 36, 'decoder1_1.bn': 37, 'decoder1_2.conv': 39, 'decoder1_2.bn': 40, 'decoder2_1.conv': 42, 'decoder2_1.bn': 43, 'decoder2_2.conv': 45, 'decoder2_2.bn': 46, 'decoder2_3.conv': 48}
            from torch.utils.serialization import load_lua
            pretrain = load_lua(pretrainfile).model
            pretrained_dict = {}
            for key, mapidx in mapping.items():
                if '.conv' in key:
                    pretrained_dict[key+'.weight'] = pretrain.modules[mapidx].weight
                    pretrained_dict[key+'.bias'] = pretrain.modules[mapidx].bias
                elif '.bn' in key:
                    pretrained_dict[key+'.weight'] = pretrain.modules[mapidx].weight
                    pretrained_dict[key+'.bias'] = pretrain.modules[mapidx].bias
                    pretrained_dict[key+'.running_var'] = pretrain.modules[mapidx].running_var
                    pretrained_dict[key+'.running_mean'] = pretrain.modules[mapidx].running_mean
            model_dict = self.state_dict()
            print ('==> [netG] load official weight as pretrain. init %d/%d layers.'%(len(pretrained_dict), len(model_dict)))
            model_dict.update(pretrained_dict) 
            self.load_state_dict(pretrained_dict)
        else:
            self.load_state_dict(torch.load(pretrainfile, map_location=lambda storage, loc: storage))
            print ('==> [netG] load self-train weight as pretrain.')
    
    def forward(self, input):
        x = self.conv1_1(input)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.decoder1_1(x)
        x = self.decoder1_2(x)
        x = self.decoder2_1(x)
        x = self.decoder2_2(x)
        x = self.decoder2_3(x)
        x = F.sigmoid(x)
        return x

    def calc_loss(self, pred, gt):
        loss = torch.nn.MSELoss()(pred, gt)
        return loss

class GLCIC_D(nn.Module):
    def __init__(self, bias_in_conv=True, pretrainfile=None):
        super(GLCIC_D, self).__init__()
        # local D
        self.local_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.local_fc = nn.Linear(8192, 1024)
        # global D
        self.global_conv1 = ConvBnRelu(3, 64, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv2 = ConvBnRelu(64, 128, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv3 = ConvBnRelu(128, 256, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv4 = ConvBnRelu(256, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv5 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_conv6 = ConvBnRelu(512, 512, kernel_size=5, stride=2, bias=bias_in_conv)
        self.global_fc = nn.Linear(8192, 1024)
        # after concat
        self.fc = nn.Linear(2048, 1)
        
        self.init(pretrainfile)
        
    def init(self, pretrainfile=None):
        if pretrainfile is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, .1)
                    m.bias.data.zero_()
        else:
            self.load_state_dict(torch.load(pretrainfile))
            print ('==> [netD] load self-train weight as pretrain.')

    
    def forward(self, input_local, input_global):
        x_local = self._forward_local(input_local)
        x_global = self._forward_global(input_global)
        x = torch.cat([x_local, x_global], 1)
        x = self.fc(x)
        return x
        
    def _forward_local(self, input):
        x = self.local_conv1(input)
        x = self.local_conv2(x)
        x = self.local_conv3(x)
        x = self.local_conv4(x)
        x = self.local_conv5(x)
        x = x.view(x.size(0), -1)
        x = self.local_fc(x)
        return x
    
    def _forward_global(self, input):
        x = self.global_conv1(input)
        x = self.global_conv2(x)
        x = self.global_conv3(x)
        x = self.global_conv4(x)
        x = self.global_conv5(x)
        x = self.global_conv6(x)
        x = x.view(x.size(0), -1)
        x = self.global_fc(x)
        return x

    def calc_loss(self, pred, gt):
        loss = nn.BCEWithLogitsLoss()(pred, gt)
        return loss
    
    
######################################################################################################################
#                                                    Dataset: ATR/LIP
######################################################################################################################
# CONFIG.DATASET.TRAINDIR
# CONFIG.DATASET.VALDIR   
# CONFIG.DATASET.INPUT_RES
# CONFIG.DATASET.MEAN

class MyDataset(object):
    def __init__(self, ImageDir, istrain=True):
        self.istrain = istrain
        self.imgdir = ImageDir
        self.imglist = os.listdir(ImageDir)
        print ('==> Load Dataset: \n', {'dataset': ImageDir, 'istrain:': istrain, 'len': self.__len__()}, '\n')
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        path = os.path.join(self.imgdir, self.imglist[idx])
        return self.loadImage(path)
    
    def loadImage(self, path):
        image = cv2.imread(path)
        image = image[:,:,::-1]
        image = cv2.resize(image, (CONFIG.DATASET.INPUT_RES, CONFIG.DATASET.INPUT_RES), interpolation=cv2.INTER_LINEAR)
        input = (image.astype(np.float32)/255.0 - CONFIG.DATASET.MEAN)
        input = input.transpose(2,0,1)
        
        if self.istrain:
            bbox_c, mask_c = self.randommask(image.shape[0], image.shape[1])
            bbox_d, mask_d = self.randommask(image.shape[0], image.shape[1])
        else:
            mask_c = cv2.imread(path.replace('images', 'masks')).astype(np.float32)
            bbox_c = 0
            mask_d = 0
            bbox_d = 0
        return np.float32(input), np.float32(mask_c), bbox_c, np.float32(mask_d), bbox_d
    
    def randommask(self, height, width):
        x1, y1 = np.random.randint(0, CONFIG.DATASET.INPUT_RES - CONFIG.DATASET.LOCAL_RES + 1, 2)
        x2, y2 = np.array([x1, y1]) + CONFIG.DATASET.LOCAL_RES
        w, h = np.random.randint(CONFIG.DATASET.HOLE_MIN, CONFIG.DATASET.HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, CONFIG.DATASET.LOCAL_RES - w)
        q1 = y1 + np.random.randint(0, CONFIG.DATASET.LOCAL_RES - h)
        p2 = p1 + w
        q2 = q1 + h
        mask = np.zeros((height, width), dtype=np.float32)
        mask[q1:q2 + 1, p1:p2 + 1] = 1.0
        bbox = np.array([x1, y1, x1+CONFIG.DATASET.LOCAL_RES, y1+CONFIG.DATASET.LOCAL_RES], dtype=np.int32)
        return bbox, mask[np.newaxis, :,:]

    
######################################################################################################################
#                                                   Training
######################################################################################################################
def train(dataLoader, model_G, model_D, epoch):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses_G = AverageMeter('losses_G')
    losses_D = AverageMeter('losses_D')
    
    losses_G_L2 = AverageMeter('losses_G_L2')
    losses_G_real = AverageMeter('losses_G_real')
    losses_D_real = AverageMeter('losses_D_real')
    losses_D_fake = AverageMeter('losses_D_fake')
    
        
    # switch to train mode
    model_G.train()
    model_D.train()

    end = time.time()
    for i, data in enumerate(dataLoader):
        # measure data loading time
        data_time.update(time.time() - end)

        input3ch, mask_c, bbox_c, mask_d, bbox_d = data
        input4ch = torch.cat([input3ch * (1 - mask_c), mask_c], dim=1)
        
        input3ch_var = to_varabile(input3ch, requires_grad=False, is_cuda=True)
        input4ch_var = to_varabile(input4ch, requires_grad=True, is_cuda=True)
        bbox_c_var = to_varabile(bbox_c, requires_grad=False, is_cuda=True)
        
        out_G = model_G(input4ch_var)
        loss_G_L2 = model_G.calc_loss(out_G, input3ch_var)
        losses_G_L2.update(loss_G_L2.data[0], input3ch.size(0))
        
        completion = (input3ch_var + MEAN_var)*(1 - mask_c.cuda()) + out_G * mask_c.cuda()
#         local_completion = completion[:,:,bbox_c[0][1]:bbox_c[0][3], bbox_c[0][0]:bbox_c[0][2]]
#         local_input3ch = input3ch_var[:,:,bbox_c[0][1]:bbox_c[0][3], bbox_c[0][0]:bbox_c[0][2]]
        local_completion = CropAlignOp(completion, bbox_c_var, 
                                       CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)
        local_input3ch = CropAlignOp(input3ch_var, bbox_c_var, 
                                       CONFIG.DATASET.LOCAL_RES, CONFIG.DATASET.LOCAL_RES, spatial_scale=1.0)

        
        out_D_fake = model_D(local_completion, completion)
        loss_D_fake = model_D.calc_loss(out_D_fake, torch.zeros_like(out_D_fake))
        losses_D_fake.update(loss_D_fake.data[0], input3ch.size(0))
        
        out_D_real = model_D(local_input3ch+MEAN_var, input3ch_var+MEAN_var)
        loss_D_real = model_D.calc_loss(out_D_real, torch.ones_like(out_D_real))
        losses_D_real.update(loss_D_real.data[0], input3ch.size(0))
        
        out_G_real = model_D(local_completion, completion)
        loss_G_real = model_D.calc_loss(out_G_real, torch.ones_like(out_G_real))
        losses_G_real.update(loss_G_real.data[0], input3ch.size(0))
        
        if epoch <= CONFIG.TRAIN_G_EPOCHES:
            optimizer = torch.optim.Adam(model_G.parameters(), 
                                         CONFIG.SOLVER.LR, weight_decay=CONFIG.SOLVER.WEIGHTDECAY) 
            loss_G = losses_G_L2
            losses_G.update(loss_G.data[0], input3ch.size(0))
            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()
            
        elif epoch <= CONFIG.TRAIN_D_EPOCHES:
            optimizer = torch.optim.Adam(model_D.parameters(), 
                                         CONFIG.SOLVER.LR, weight_decay=CONFIG.SOLVER.WEIGHTDECAY) 
            loss_D = loss_D_fake + loss_D_real
            losses_D.update(loss_D.data[0], input3ch.size(0))
            optimizer.zero_grad()
            loss_D.backward()
            optimizer.step()
            
        else:
            optimizer_G = torch.optim.Adam(model_G.parameters(), CONFIG.SOLVER.LR, weight_decay=CONFIG.SOLVER.WEIGHTDECAY) 
            optimizer_D = torch.optim.Adam(model_D.parameters(), CONFIG.SOLVER.LR, weight_decay=CONFIG.SOLVER.WEIGHTDECAY) 
            loss_G = loss_G_L2 + CONFIG.LOSS.ALPHA * loss_G_real
            loss_D = loss_D_fake + loss_D_real
            losses_G.update(loss_G.data[0], input3ch.size(0))
            losses_D.update(loss_D.data[0], input3ch.size(0))
        
            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()
            
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % CONFIG.LOGS.PRINT_FREQ == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'G {loss_G.val:.4f} ({loss_G.avg:.4f})\t'
                  'D {loss_D.val:.4f} ({loss_D.avg:.4f})\t'
                  'G_L2 {G_L2.val:.4f} ({G_L2.avg:.4f})\t'
                  'G_real {G_real.val:.4f} ({G_real.avg:.4f})\t'
                  'D_fake {D_fake.val:.4f} ({D_fake.avg:.4f})\t'
                  'D_real {D_real.val:.4f} ({D_real.avg:.4f})\t'.format(
                   epoch, i, len(dataLoader), batch_time=batch_time, #data_time=data_time, 
                   loss_G=losses_G, loss_D=losses_D,
                   G_L2 = losses_G_L2, G_real=losses_G_real,
                   D_fake=losses_D_fake, D_real=losses_D_real ))
        
        if i % CONFIG.LOGS.LOG_FREQ == 0:
            vis = torch.cat([input3ch_var * (1 - mask_c.cuda()) + MEAN_var,
                             completion], dim=0)
            save_image(vis, os.path.join(LOGDIR, 'epoch%d_%d_vis.jpg'%(epoch, i)), nrow=input3ch.size(0), padding=2,
                       normalize=True, range=None, scale_each=True, pad_value=0)
        
            vis = torch.cat([local_input3ch + MEAN_var, local_completion], dim=0)
            save_image(vis, os.path.join(LOGDIR, 'epoch%d_%d_vis_crop.jpg'%(epoch, i)), nrow=input3ch.size(0), padding=2,
                       normalize=True, range=None, scale_each=True, pad_value=0)
            
        if i % CONFIG.LOGS.SNAPSHOT_FREQ == 0:
            torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, 'G_%d_%d.pkl'%(epoch,i)))
            torch.save(model_D.state_dict(), os.path.join(SNAPSHOTDIR, 'D_%d_%d.pkl'%(epoch,i)))
    
    if epoch == CONFIG.TRAIN_G_EPOCHES:
        torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, 'preG_%d_%d.pkl'%(epoch,i)))
        
    if epoch == CONFIG.TRAIN_D_EPOCHES:
        torch.save(model_G.state_dict(), os.path.join(SNAPSHOTDIR, 'preD_%d_%d.pkl'%(epoch,i)))
            
def main():
    dataset = MyDataset(ImageDir=CONFIG.DATASET.TRAINDIR, istrain=True)
    
    BATCHSIZE = CONFIG.SOLVER.IMG_PER_GPU * len(CONFIG.SOLVER.GPU_IDS)
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=CONFIG.SOLVER.WORKERS, pin_memory=False)
    
    model_G = GLCIC_G(bias_in_conv=True, pretrainfile=CONFIG.INIT).cuda()
    model_D = GLCIC_D(bias_in_conv=True).cuda()
    
    epoches = 200
    for epoch in range(epoches):
        print ('===========>   [Epoch %d] training    <==========='%epoch)
        train(dataLoader, model_G, model_D, epoch)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    