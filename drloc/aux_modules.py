import torch
import torch.nn as nn
import torch.nn.functional as F

from munch import Munch
torch.manual_seed(12345)

def randn_sampling(maxint, sample_size, batch_size):
    return torch.randint(maxint, size=(batch_size, sample_size, 2))

def collect_samples(feats, pxy, batch_size):
    return torch.stack([feats[i, :, pxy[i][:,0], pxy[i][:,1]] for i in range(batch_size)], dim=0)

def collect_samples_faster(feats, pxy, batch_size):
    n,c,h,w = feats.size()
    feats = feats.view(n, c, -1).permute(1,0,2).reshape(c, -1)  # [n, c, h, w] -> [n, c, hw] -> [c, nhw]
    pxy = ((torch.arange(n).long().to(pxy.device) * h * w).view(n, 1) + pxy[:,:,0]*h + pxy[:,:,1]).view(-1)  # [n, m, 2] -> [nm]
    return (feats[:,pxy]).view(c, n, -1).permute(1,0,2)

def collect_positions(batch_size, N):
    all_positions = [[i,j]  for i in range(N) for j in range(N)]
    pts = torch.tensor(all_positions) # [N*N, 2]
    pts_norm = pts.repeat(batch_size,1,1)  # [B, N*N, 2]
    rnd = torch.stack([torch.randperm(N*N) for _ in range(batch_size)], dim=0) # [B, N*N]
    pts_rnd = torch.stack([pts_norm[idx,r] for idx, r in enumerate(rnd)],dim=0) # [B, N*N, 2]
    return pts_norm, pts_rnd

class DenseRelativeLoc(nn.Module):
    def __init__(self, in_dim, out_dim=2, sample_size=32, drloc_mode="l1", use_abs=False):
        super(DenseRelativeLoc, self).__init__()
        self.sample_size = sample_size
        self.in_dim  = in_dim
        self.drloc_mode = drloc_mode
        self.use_abs = use_abs

        if self.drloc_mode == "l1":
            self.out_dim = out_dim
            self.layers = nn.Sequential(
                nn.Linear(in_dim*2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
        elif self.drloc_mode in ["ce", "cbr"]:
            self.out_dim = out_dim if self.use_abs else out_dim*2 - 1
            self.layers  = nn.Sequential(
                nn.Linear(in_dim*2, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )
            self.unshared = nn.ModuleList()
            for _ in range(2):
                self.unshared.append(nn.Linear(512, self.out_dim))
        else:
            raise NotImplementedError("We only support l1, ce and cbr now.")

    def forward_features(self, x, mode="part"):
        # x, feature map with shape: [B, C, H, W]
        B, C, H, W = x.size()

        if mode == "part":
            pxs = randn_sampling(H, self.sample_size, B).detach()
            pys = randn_sampling(H, self.sample_size, B).detach()
            
            deltaxy = (pxs-pys).float().to(x.device) # [B, sample_size, 2]

            ptsx = collect_samples_faster(x, pxs, B).transpose(1,2).contiguous() # [B, sample_size, C]
            ptsy = collect_samples_faster(x, pys, B).transpose(1,2).contiguous() # [B, sample_size, C]
        else:
            pts_norm, pts_rnd = collect_positions(B, H)
            ptsx = x.view(B,C,-1).transpose(1,2).contiguous() # [B, H*W, C]
            ptsy = collect_samples(x, pts_rnd, B).transpose(1,2).contiguous() # [B, H*W, C]

            deltaxy = (pts_norm - pts_rnd).float().to(x.device) # [B, H*W, 2]

        pred_feats = self.layers(torch.cat([ptsx, ptsy], dim=2))
        return pred_feats, deltaxy, H

    def forward(self, x, normalize=False):
        pred_feats, deltaxy, H = self.forward_features(x)
        deltaxy = deltaxy.view(-1, 2) # [B*sample_size, 2]

        if self.use_abs:
            deltaxy = torch.abs(deltaxy)
            if normalize:
                deltaxy /= float(H-1)
        else:
            deltaxy += (H-1)
            if normalize:
                deltaxy /= float(2*(H - 1))
        
        if self.drloc_mode == "l1":
            predxy = pred_feats.view(-1, self.out_dim) # [B*sample_size, Output_size]
        else: 
            predx, predy = self.unshared[0](pred_feats), self.unshared[1](pred_feats)
            predx = predx.view(-1, self.out_dim) # [B*sample_size, Output_size]
            predy = predy.view(-1, self.out_dim) # [B*sample_size, Output_size]
            predxy = torch.stack([predx, predy], dim=2) # [B*sample_size, Output_size, 2]   
        return predxy, deltaxy

    def flops(self):
        fps =  self.in_dim * 2 * 512 * self.sample_size
        fps += 512 * 512 * self.sample_size
        fps += 512 * self.out_dim * self.sample_size
        if self.drloc_mode in ["ce", "cbr"]:
            fps += 512 * 512 * self.sample_size
            fps += 512 * self.out_dim * self.sample_size
        return fps
