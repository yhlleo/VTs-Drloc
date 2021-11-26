from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON=1e-8

def relative_constraint_l1(deltaxy, predxy):
    return F.l1_loss(deltaxy, predxy)

def relative_constraint_ce(deltaxy, predxy):
    #predx, predy = torch.chunk(predxy, chunks=2, dim=1)
    predx, predy = predxy[:,:,0], predxy[:,:,1]
    targetx, targety = deltaxy[:,0].long(), deltaxy[:,1].long()
    return F.cross_entropy(predx, targetx) + F.cross_entropy(predy, targety)

def variance_aware_regression(pred, beta, target, labels, lambda_var=0.001):
    # Variance aware regression.
    pred_titled = pred.unsqueeze(0).t().repeat(1, labels.size(1))
    pred_var = torch.sum((labels-pred_titled)**2*beta, dim=1) + EPSILON
    pred_log_var = torch.log(pred_var) 
    squared_error = (pred - target)**2
    return  torch.mean(torch.exp(-pred_log_var) * squared_error + lambda_var * pred_log_var)

# based on the codes: https://github.com/google-research/google-research/blob/master/tcc/tcc/losses.py
def relative_constraint_cbr(deltaxy, predxy, loss_type="regression_mse_var"):
    predx, predy = predxy[:,:,0], predxy[:,:,1]
    num_classes  = predx.size(1)
    targetx, targety = deltaxy[:,0].long(), deltaxy[:,1].long()   # [N, ], [N, ]
    betax, betay = F.softmax(predx,dim=1), F.softmax(predy,dim=1) # [N, C], [N, C]
    labels = torch.arange(num_classes).unsqueeze(0).to(predxy.device)  # [1, C]
    true_idx = targetx #torch.sum(targetx*labels, dim=1)      # [N, ]
    true_idy = targety #torch.sum(targety*labels, dim=1)      # [N, ]

    pred_idx = torch.sum(betax*labels, dim=1)        # [N, ]
    pred_idy = torch.sum(betay*labels, dim=1)        # [N, ]

    if loss_type in ["regression_mse", "regression_mse_var"]:
        if "var" in loss_type:
            # Variance aware regression.
            lossx = variance_aware_regression(pred_idx, betax, true_idx, labels)
            lossy = variance_aware_regression(pred_idy, betay, true_idy, labels)
        else:
            lossx = torch.mean((pred_idx - true_idx)**2)
            lossy = torch.mean((pred_idy - true_idy)**2)
        loss = lossx + lossy
        return loss
    else:
        raise NotImplementedError("We only support regression_mse and regression_mse_var now.")

def cal_selfsupervised_loss(outs, args, lambda_drloc=0.0):
    loss, all_losses = 0.0, Munch()
    if args.TRAIN.USE_DRLOC:
        if args.TRAIN.DRLOC_MODE == "l1": # l1 regression constraint
            reld_criterion = relative_constraint_l1 
        elif args.TRAIN.DRLOC_MODE == "ce": # cross entropy constraint
            reld_criterion = relative_constraint_ce
        elif args.TRAIN.DRLOC_MODE == "cbr": # cycle-back regression constaint: https://arxiv.org/pdf/1904.07846.pdf
            reld_criterion = relative_constraint_cbr
        else:
            raise NotImplementedError("We only support l1, ce and cbr now.")

        loss_drloc = 0.0
        for deltaxy, drloc, plane_size in zip(outs.deltaxy, outs.drloc, outs.plz):
            loss_drloc += reld_criterion(deltaxy, drloc) * lambda_drloc
        all_losses.drloc = loss_drloc.item()
        loss += loss_drloc

    return loss, all_losses

