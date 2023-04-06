import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from collections import OrderedDict
import models.modules.module_util as mutil
from models.modules.non_parameteric_regression import KernelEstimation
import sys
import math
from scipy.stats import norm
from torchvision.utils import make_grid
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from skimage import io, color
from scipy.stats import gaussian_kde
import cv2



class multiregression(nn.Module):
    ''' stage1, train MANet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=4, pca_path='./pca_matrix_aniso21_15_x2.pth',
                 code_length=15, kernel_size=21, manet_nf=256, manet_nb=1, split=2, features = 4):
        super(multiregression, self).__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.features = features
        self.kernel_estimation = KernelEstimation(in_nc=in_nc, kernel_size=kernel_size)
        self.parameter_extractor = ConvNet()
        self.localpool = nn.AvgPool2d(20, stride=20, padding=0)
        self.avgpoool = nn.AdaptiveAvgPool2d(1)
        self.avgpoool2 = nn.AdaptiveAvgPool2d(84)

        
    def forward(self, x, gt_K):
        # non-parametric part
        kernel = self.kernel_estimation(x)
        kernelx_p = self.avgpoool(kernel)
        kernelx_p = kernelx_p.view(1, -1, self.kernel_size, self.kernel_size)
        kernelx_p_p = F.interpolate(kernelx_p, scale_factor=self.scale, mode='nearest')
        kernel_p = self.localpool(kernel)
        # kernel_p = F.interpolate(kernel_p, scale_factor=20, mode='nearest')
        
        b_,c,p_1,p_2 = kernel_p.size()
            
        param_m = torch.zeros((b_,p_1*p_2,3), device = 'cuda:0', requires_grad=True).clone()
        kernel_p = kernel_p.permute(0,2,3,1).view(b_,p_1*p_2,21,21)
        
            
            
        ## parametric part
        #### SV parametric
        for b in range(0, kernel_p.size(0)) :
            i = 0
            for kernelx in kernel_p[b] :
                kernelx = kernelx.contiguous().view(1,1,21,21)
                kernelx = F.interpolate(kernelx, scale_factor=self.scale, mode='nearest')
                param = self.parameter_extractor(kernelx*100)
                param_m[b, i, :] = param
                i = i + 1
        # param_m = F.interpolate(param_m.view(param_m.size(0),param_m.size(2),param_m.size(1)).contiguous(), scale_factor=self.scale*4, mode='nearest').flatten(2).permute(0, 2, 1)                     
        # param_m = F.interpolate(param_m, scale_factor=20, mode='nearest').permute(0, 2, 1)
        # print(param_m.size())
        # sys.exit()
        
        ### non sv mode
        param = self.parameter_extractor(kernelx_p_p*100)
        # no meaning
        with torch.no_grad():
            out = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return out, kernel_p, param_m


class ConvNet(nn.Module):
    def __init__(self): 
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=2) 
        self.conv2 = nn.Conv2d(40, 80, kernel_size=5, stride=2) 
        self.conv3 = nn.Conv2d(80, 100, kernel_size=5, stride=2)

        self.relu = nn.ReLU(inplace=True)

        self.drop2D = nn.Dropout2d(p=0.25, inplace=False) 
        self.fc1 = nn.Linear(4900,100) 
        self.fc2 = nn.Linear(100,1) 
        
        self.fc21 = nn.Linear(4900,100) 
        self.fc22 = nn.Linear(100,1) 
        
        self.fc31 = nn.Linear(4900,100)
        self.fc32 = nn.Linear(100,1) 
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        b, _,_,_ = x.size()
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x)) 
        x = x.view(x.size(0), -1) 
        x1 = self.fc1(x) 
        x1 = self.fc2(self.relu(x1)) 

        x2 = self.fc21(x) 
        x2 = self.fc22(self.relu(x2)) 
        
        x3 = self.fc31(x) 
        x3 = self.fc32(self.relu(x3)) 
        
        result = torch.stack([x1,x2,x3],dim=1).view(b,3)
        return result 
    