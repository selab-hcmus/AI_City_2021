import os, json, cv2, time, copy, pickle
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

class BceDiceLoss(nn.Module):
    def __init__(self, weight_bce: float=0.0, weight_dice: float=1.0):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        pass 
    
    def forward(self, inp, target):
        batch_size = inp.shape[0]
        inp = inp.view(batch_size, -1)
        target = target.view(batch_size, -1)
        bce_loss = nn.BCELoss(reduction='none')(inp, target).mean(dim=-1).double() #[B]
        dice_coef = (2.0*(inp*target).sum(dim=-1).double() + 1)/(
            inp.sum(dim=-1).double() + target.sum(dim=-1).double() + 1
        )
        dice_loss = 1-dice_coef
        total_loss = torch.mean(self.weight_bce*bce_loss + self.weight_dice*dice_loss)
        return total_loss
    pass

class TSA_BceDiceLoss(nn.Module):
    def __init__(self, weight_bce:float=0.0, weight_dice:float=1.0, num_steps:int=2000, alpha:float=5.0, num_classes:int=8):
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.num_steps = num_steps 
        self.alpha = alpha
        self.current_step = 0
        self.num_classes = num_classes
        self.thres_history = []
        pass 
    
    def step(self):
        self.current_step += 1
        pass

    def threshold(self):
        # alpha_3: a = exp(5*(t/T-1)) 
        alpha_3 = torch.exp(torch.tensor(self.alpha*(self.current_step/self.num_steps - 1)))
        # alpha_1: a = 1 - exp(5* -t/T)
        alpha_1 = 1 - torch.exp(torch.tensor(-self.alpha*(self.current_step/self.num_steps)))
        
        thres = alpha_1*(1-1/self.num_classes) + 1/self.num_classes
        self.thres_history.append(thres)
        return thres

    def forward(self, inp, target):
        batch_size = inp.shape[0]
        inp = inp.view(batch_size, -1)
        target = target.view(batch_size, -1)
        bce_loss = nn.BCELoss(reduction='none')(inp, target).mean(dim=-1).double() #[B]
        dice_coef = (2.0*(inp*target).sum(dim=-1).double() + 1)/(
            inp.sum(dim=-1).double() + target.sum(dim=-1).double() + 1
        )

        mask = (dice_coef < self.threshold()).detach().double()
        dice_loss = 1-dice_coef
        total_loss = torch.mean((self.weight_bce*bce_loss + self.weight_dice*dice_loss)*mask)
        
        self.step()
        
        return total_loss

def l2_loss(reduction: str='mean'):
    criterion = nn.MSELoss(reduction=reduction)
    return criterion


