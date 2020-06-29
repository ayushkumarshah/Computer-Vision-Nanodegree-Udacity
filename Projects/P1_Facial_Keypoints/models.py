## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        self.depth = [1, 32, 64, 128, 256, 512]

        # convolutional layer (sees 96x96x1 image tensor)
        self.conv_1 = nn.Conv2d(self.depth[0], self.depth[1], 5)
        
        # convolutional layer (sees 46x46x32 image tensor)
        self.conv_2 = nn.Conv2d(self.depth[1], self.depth[2], 5)
#         self.batchnorm_2 = nn.BatchNorm2d(self.depth[2])
        
#         # convolutional layer (sees 53x53x64 image tensor)
#         self.conv_3 = nn.Conv2d(self.depth[2], self.depth[3], 5)
# #         self.batchnorm_3 = nn.BatchNorm2d(self.depth[3])
        
#         # convolutional layer (sees 24x24x128 image tensor)
#         self.conv_4 = nn.Conv2d(self.depth[3], self.depth[4], 5)
# #         self.batchnorm_4 = nn.BatchNorm2d(self.depth[4])
        
#         # convolutional layer (sees 10x10x256 image tensor)
#         self.conv_5 = nn.Conv2d(self.depth[4], self.depth[5], 5) 
# #         self.batchnorm_5 = nn.BatchNorm2d(self.depth[5])
        
        
        # linear layer 64x21x21 -> 1000
        self.fc1 = nn.Linear(self.depth[2] * 21 * 21, 1000)
        # linear layer 1000 -> 512
        self.fc2 = nn.Linear(1000, 512)
        # linear layer 512 -> 136
        self.fc3 = nn.Linear(512, 136)
        
        # Dropout
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.MaxPool2d(2, 2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu((self.conv_2(x))))
#         x = self.pool(F.relu((self.conv_3(x))))
#         x = self.pool(F.relu((self.conv_4(x))))
#         x = self.pool(F.relu((self.conv_5(x))))
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x = x.view(-1, self.depth[5] * 3 * 3)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
