from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


#from torch.utils.data import DataLoader
#import torchvision.transforms as transforms
#import torchvision
#import json
#import torch.optim as optim
import torch
##import os.path
#import argparse
#import h5py
#import time
#import numpy as np



class CNNModel(nn.Module):
    """docstring for ClassName"""
    
    def __init__(self, args):
        super(CNNModel, self).__init__()
        ##-----------------------------------------------------------
        ## define the model architecture here
        ## MNIST image input size batch * 28 * 28 (one input channel)
        ##-----------------------------------------------------------
        
        ## define CNN layers below
        ##nn.conv2D(in_channels, out_channels, kernel_size, stride, padding)
        self.conv = nn.Sequential(nn.Conv2d(1, 14, args.k_size, args.stride, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.BatchNorm2d(14),
                                    nn.Conv2d(14, 10, args.k_size, args.stride, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.4),
                                    nn.BatchNorm2d(10),
                                    nn.Conv2d(10, 6, args.k_size, args.stride, 1),
                                    nn.ReLU(),
                                    nn.Dropout(0.4),
                                    nn.BatchNorm2d(6),
                                    nn.MaxPool2d(args.pooling_size,args.pooling_size),
                                )
        
        

        ##------------------------------------------------
        ## write code to define fully connected layer below
        ##------------------------------------------------
        in_size = 864
        between_size = 196
        out_size = 10
        self.fc1 = nn.Linear(in_size, between_size)
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(between_size, out_size)
        )
            

    '''feed features to the model'''
    def forward(self, x):  #default
        
        ##---------------------------------------------------------
        ## write code to feed input features to the CNN models defined above
        ##---------------------------------------------------------
        x_out = self.conv(x)

        ## write flatten tensor code below (it is done)
        x = torch.flatten(x_out,1) # x_out is output of last layer
        
        ## ---------------------------------------------------
        ## write fully connected layer (Linear layer) below
        ## ---------------------------------------------------
        result = self.fc1(x)
        result = self.fc2(result)
        
        return result
        