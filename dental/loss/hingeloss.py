import torch
import torch.nn as nn
from torch.autograd import Variable 


class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, size_average=True, sign=1.0):
        super(HingeLoss, self).__init__()
        self.sign = sign
        self.margin = margin
        self.size_average = size_average
 
    def forward(self, input, target):
        #
        input = input.view(-1)
        target = target.view(-1)

        #
        # assert input.dim() == target.dim()
        # for i in range(input.dim()): 
        #     assert input.size(i) == target.size(i)

        output = self.margin - torch.mul(target, input)
   
        if 'cuda' in input.data.type():
            mask = torch.cuda.FloatTensor(input.size()).zero_()
        else:
            mask = torch.FloatTensor(input.size()).zero_()
        mask = Variable(mask)
        mask[torch.gt(output, 0.0)] = 1.0

        #
        output = torch.mul(output, mask)

        # size average
        if self.size_average:
            output = torch.mul(output, 1.0 / input.nelement())

        # sum
        output = output.sum()

        # apply sign
        output = torch.mul(output, self.sign)
        return output