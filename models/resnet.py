import torch
import torch.nn as nn
import torchvision
from torchvision import models

from munch import Munch
from drloc import DenseRelativeLoc

class ResNet50(nn.Module):
    def __init__(
        self, 
        num_classes,
        use_drloc=False,     # relative distance prediction
        drloc_mode="l1",
        sample_size=32,
        use_abs=False,
    ):
        super().__init__()
        self.use_drloc = use_drloc

        # don't use the pretrained model
        model = models.resnet50(pretrained=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        layers = [v for v in model.children()]
        self.model = nn.Sequential(*layers[:-2])
        self.pool = layers[-2]
        self.fc   = layers[-1]

        if self.use_drloc:
            self.drloc = nn.ModuleList()
            self.drloc.append(DenseRelativeLoc(
                in_dim=num_ftrs, 
                out_dim=2 if drloc_mode=="l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=use_abs))

    def forward(self,x): 
        x = self.model(x) # [B, C, H, W]
        outs = Munch()

        # SSUP
        B, C, H, W = x.size()
        if self.use_drloc:
            outs.drloc = []
            outs.deltaxy = []
            outs.plz = []

            for idx, x_cur in enumerate([x]):
                drloc_feats, deltaxy = self.drloc[idx](x_cur)
                outs.drloc.append(drloc_feats)
                outs.deltaxy.append(deltaxy)
                outs.plz.append(H) # plane size 

        x = self.fc(torch.flatten(self.pool(x), 1))
        outs.sup = x
        return outs
