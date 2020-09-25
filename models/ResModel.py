import torch
import torchvision
import torch.nn as nn


class ResModel(nn.Module):
    def __init__(self, config):
        super(ResModel,self).__init__()
        res = torchvision.models.resnet18(pretrained=config.MODEL.PRETRAINED)
        num_infeatures = res.fc.in_features
        res.fc =nn.Linear(num_infeatures,40*12)
        self.conv = res
        self.fc = nn.Sequential(
            nn.Linear(40,1),
        )
    def au_out(self, x):
        out = torch.argmax(self.fc(x),dim=1).type(torch.float32)
        return out.view([-1,1])

    def forward(self, x):
        x = self.conv(x)
        x_split = torch.split(x, 40, dim=1)

        x = self.fc(x_split[0])
        for s in x_split[1:]:
            v = self.fc(s)
            x = torch.cat([x, v],dim=1)

        return x
