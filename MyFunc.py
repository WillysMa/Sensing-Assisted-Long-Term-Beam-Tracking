#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: MyFunc.py
@time: 2025/5/26 16:09
"""
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transf
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
from torch.optim.lr_scheduler import _LRScheduler
from collections.abc import Iterable
import torch.nn as nn
import torch.nn.functional as F
from math import log, cos, pi, floor
import random
from Radar_KPI import *
from thop import profile as thop_profile

def set_seed(seed=42):
    """Set all random seeds for reproducible training"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_best_gpu():
    if not torch.cuda.is_available():
        return ""
    
    best_gpu = 0
    min_memory_used = float('inf')
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        memory_used = torch.cuda.memory_allocated(i)
        if memory_used < min_memory_used:
            min_memory_used = memory_used
            best_gpu = i
    
    return str(best_gpu)


def compute_flops(model, inputs, name, batch_size=1):
    if thop_profile is None:
        print(f"[FLOPs] thop not available; skip {name} FLOPs.")
        return None, None
    # thop.profile() calls model(*inputs); a single tensor must be passed as (tensor,) so the model gets one argument
    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)
    model.eval()
    with torch.no_grad():
        flops, params = thop_profile(model, inputs=inputs, verbose=False)
    print(f"[FLOPs] {name}: {flops/batch_size/1e6:.3f} M FLOPs, {params/1e6:.3f} M params")
    return flops, params


    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_topk_accuracy(outputs, labels, k_values=[1, 2, 3, 5, 10]):
    """Calculate top-k accuracy for given k values"""
    num_pred = labels.shape[1]
    topk_correct = {k: np.zeros((num_pred,)) for k in k_values}
    total = torch.sum(labels != -100, dim=0).cpu().numpy()
    
    _, idx = torch.topk(outputs, max(k_values), dim=-1)
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for i in range(labels.shape[1]):  # for each time step
        for j in range(labels.shape[0]):  # examine all samples
            for k in k_values:
                topk_correct[k][i] += np.isin(labels[j, i], idx[j, i, :k])
    
    # Calculate accuracy
    topk_acc = {}
    for k in k_values:
        topk_acc[k] = topk_correct[k] / (total + 1e-8)  # Add small epsilon to avoid division by zero
    
    return topk_acc, total

def calculate_dba_score(outputs, labels, delta=5):
    """Calculate DBA (Distance-Based Accuracy) score"""
    num_pred = labels.shape[1]
    dba_score = np.zeros((num_pred,))
    valid_count = np.zeros((num_pred,))
    
    _, idx = torch.topk(outputs, 3, dim=-1)  # top-3 predictions for DBA
    idx = idx.cpu().numpy()
    labels = labels.cpu().numpy()
    
    for t in range(labels.shape[1]):
        for b in range(labels.shape[0]):
            gt = labels[b, t]
            if gt == -100:
                continue  # skip invalid label
            
            preds = idx[b, t, :3]  # top-3 predictions
            norm_dists = np.minimum(np.abs(preds - gt) / delta, 1.0)
            min_norm_dist = np.min(norm_dists)
            
            dba_score[t] += min_norm_dist
            valid_count[t] += 1
    
    # Avoid division by zero
    valid_count[valid_count == 0] = 1
    dba_score = 1 - (dba_score / valid_count)
    
    return dba_score

def save_checkpoint(state, save_path, filename='checkpoint.pth'):
    """Save training checkpoint"""
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(save_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint_path = os.path.join(save_path, 'Final_model.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return start_epoch, checkpoint.get('test_loss', 0.0)
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, 0.0
    

    

def plot_training_curves(train_acc_hist, train_loss_hist, test_acc_hist, test_loss_hist, lrs, save_path, 
            train_task_loss_hist=None, train_distill_loss_hist=None, div_history=None):
    """Plot and save training curves including knowledge distillation losses"""
    epochs = len(train_acc_hist)
    
    # Learning rate schedule
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.grid(True)
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(save_path, 'LR_schedule.png'))
    plt.close()
    
    # Accuracy curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_acc_hist, '-o', label='Train')
    plt.plot(np.arange(1, epochs + 1), test_acc_hist, '-o', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Accuracy_curves.png'))
    plt.close()
    
    # Loss curves
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_loss_hist, '-o', label='Train Total')
    plt.plot(np.arange(1, epochs + 1), test_loss_hist, '-o', label='Test')
    
    # Add knowledge distillation loss components if available
    if train_task_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_task_loss_hist, '--', label='Train Task Loss')
    if train_distill_loss_hist is not None:
        plt.plot(np.arange(1, epochs + 1), train_distill_loss_hist, ':', label='Train Distillation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Components (Knowledge Distillation)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Loss_curves_KD.png'))
    plt.close()

    # KL divergence curves for each time slot
    if div_history is not None:
        div_history = np.array(div_history)
        
        # Check if div_history has meaningful data (not empty and not all zeros)
        if div_history.size > 0 and not np.all(div_history == 0):
            plt.figure(figsize=(12, 8))
            
            # If div_history is 2D (epochs x time_slots), plot each time slot
            if div_history.ndim == 2:
                num_time_slots = div_history.shape[1]
                epochs_range = np.arange(1, epochs + 1)
                
                for slot in range(num_time_slots):
                    plt.plot(epochs_range, div_history[:, slot], '-o', 
                            label=f'Time Slot {slot}', markersize=3)
            else:
                # If 1D, assume it's for a single time slot or average
                plt.plot(np.arange(1, epochs + 1), div_history, '-o', 
                        label='KL Divergence', markersize=3)
            
            plt.xlabel('Epoch')
            plt.ylabel('KL Divergence')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('KL Divergence per Time Slot vs Epochs')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'KL_divergence_vs_epochs.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Also create a heatmap showing KL divergence evolution
            if div_history.ndim == 2:
                plt.figure(figsize=(12, 6))
                im = plt.imshow(div_history.T, aspect='auto', cmap='viridis', 
                               interpolation='nearest')
                plt.colorbar(im, label='KL Divergence')
                plt.xlabel('Epoch')
                plt.ylabel('Time Slot')
                plt.title('KL Divergence Heatmap: Time Slots vs Epochs')
                plt.xticks(np.arange(0, epochs, max(1, epochs//10)), 
                          np.arange(1, epochs+1, max(1, epochs//10)))
                plt.yticks(np.arange(num_time_slots), 
                          [f'Slot {i}' for i in range(num_time_slots)])
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'KL_divergence_heatmap.png'), 
                           bbox_inches='tight', dpi=300)
                plt.close()
        else:
            print("Skipping KL divergence plots: No meaningful KL divergence data available (KD mode may not be 1 or teacher model not used)")



