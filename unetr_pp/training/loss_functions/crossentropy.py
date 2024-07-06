from torch import nn, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from monai.metrics import compute_hausdorff_distance

from kornia.losses import HausdorffERLoss3D


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
    
class HausdorffDistance(nn.Module):
    def __init__(self):
        super(HausdorffDistance, self).__init__()
            
        self.weight1 = 0.05
        self.weight2 = 1
        self.threshold = 0.5
        self.Hausdorff_loss_3D = HausdorffERLoss3D()
        
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        '''
        https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/hausdorff.html
        '''
        input = torch.where(input >= self.threshold, 1.0, 0.0)
        #print(input.type()) # torch.cuda.LongTensor
        #print(target.type()) # torch.cuda.FloatTensor
        # print(input.shape) # torch.Size([8, 1, 64, 128, 128])
        
        
        hausdorff_dist_loss = self.Hausdorff_loss_3D(input, target) # background 계산 x
        if input.shape[2] == 64:
            hausdorff_dist_loss = self.weight1 * hausdorff_dist_loss
        else:
            hausdorff_dist_loss = self.weight2 * hausdorff_dist_loss

        return hausdorff_dist_loss
        