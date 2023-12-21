import torch.nn.functional as F
import torch
from torch import nn

class fusion_part_module(nn.Module):
    def __init__(self, channel, numclass=3000, sptial=34):
        super().__init__()

        self.sptial = sptial

        self.conv1 =  nn.Conv1d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        self.bn1 = nn.BatchNorm1d(channel)
        self.conv1_1 = nn.Conv1d(channel, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm1d(channel)
        

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x_1, y_1):
        input = torch.cat([x_1, y_1], 0) # n c p
        input = input.permute(1, 2, 0).contiguous() #n,c,p

        out = F.relu(self.bn1(self.conv1(input)))
        out = F.relu(self.bn1_1(self.conv1_1(out))) #n,c,p
        
        return out



  
