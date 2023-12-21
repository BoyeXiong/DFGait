import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class static_module(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # [temporal kernel, sptial kernel] 
                 num_nodes=16
                 ):
        super().__init__()
        self.p = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size[1], 1, 0, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels)
        )
        self.t = nn.Sequential(
            nn.Conv1d(in_channels*num_nodes, out_channels*num_nodes, kernel_size[0], 1, 1, groups=num_nodes, bias=False),
            # nn.ReLU(inplace=True)
            nn.BatchNorm1d(out_channels * num_nodes)
        )
    def forward(self, x):
           p, n, c, s = x.size()
           x = self.p(x.permute(1,3,2,0).contiguous().view(n*s, c, p)).view(n, s, c, p)
           x = self.t(x.permute(0,3,2,1).contiguous().view(n, p*c, s)).view(n, p, c, s)
           x = x.permute(1,0,2,3).contiguous()
           return x    
        
class stDecoupe(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # [temporal kernel, sptial kernel]
                 num_nodes=16
                 ):
        super().__init__()
        self.stCommon = static_module(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                num_nodes=num_nodes)
        self.st1 = static_module(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                num_nodes=num_nodes)
        self.st2 = static_module(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                num_nodes=num_nodes)
           
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x): #(n, c, s, p)
        x = x.permute(3, 0, 1, 2).contiguous() # (p n c s)
        xc = self.stCommon(x)
        s1 = self.st1(xc)
        s2 = self.st2(xc)
        s1 = torch.max(s1.permute(1, 2, 3, 0).contiguous(), 2)[0]
        s2 = torch.max(s2.permute(1, 2, 3, 0).contiguous(), 2)[0]
        return s1, s2 

class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, spatial_part, modal_class):
        super(modal_Classifier, self).__init__()
        self.weighted_mean = torch.nn.Conv1d(in_channels=spatial_part, out_channels=1, kernel_size=1)
        self.Liner1 = nn.Linear(embed_dim, embed_dim // 2)
        self.Liner2 = nn.Linear(embed_dim // 2, modal_class)

    def forward(self, x):
        x = x.permute(1, 0, 2).contiguous()
        x = self.weighted_mean(x)
        x = x.squeeze(1)
        x = self.Liner1(x)
        x = torch.tanh(x)
        modal_cls = self.Liner2(x)
        if self.training:
            return modal_cls  # [batch,3]