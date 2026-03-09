#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: test_model_both.py
@time: 2025/12/12 17:50
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import shutil
import time 
import subprocess
import json
import argparse
from torch.utils.data import DataLoader
from pytorch_model_summary import summary
from tqdm import tqdm
import sys
import datetime
import torchvision.transforms as transf
import matplotlib.pyplot as plt
from thop import profile as thop_profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DataFeed import DataFeed
from MyFunc import *
from model_image import *


# Automatically select least used GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = select_best_gpu()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Model Testing')
    
    # testing parameters
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--test_csv_name', type=str, default='test_seqs_RA.csv', help='Test csv name')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['crossentropy', 'focal'], help='Loss function type')
    parser.add_argument('--model_arch', type=str, default='Teacher', choices=['Teacher', 'Student'], help='Model architecture')
    parser.add_argument('--kd_mode', type=int, default=0, choices=[0, 1], help='KD mode; 0: no KD, 1: KD')
  
    # Model parameters
    parser.add_argument('--feature_size', type=int, default=64, help='Feature size')
    parser.add_argument('--gru_hidden_size', type=int, default=64, help='GRU hidden size')
    parser.add_argument('--gru_num_layers_teacher', type=int, default=2, help='Number of GRU layers for teacher')
    parser.add_argument('--gru_num_layers_student', type=int, default=1, help='Number of GRU layers for student')
    parser.add_argument('--num_classes', type=int, default=64, help='Number of classes')
    parser.add_argument('--seq_length_teacher', type=int, default=8, help='Sequence length for teacher model')
    parser.add_argument('--seq_length_student', type=int, default=8, choices=[8, 5, 3], help='Sequence length for student model')
    parser.add_argument('--num_pred', type=int, default=6, help='Number of predictions')
    parser.add_argument('--downsample_ratio', type=int, default=1, help='Downsample ratio')
    parser.add_argument('--attention', type=bool, default=True, help='Use attention for teacher model')
    # Testing setting
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available')
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode (saves to saved_folder_debug)')
    parser.add_argument('--dataset_pct', type=float, default=1.0, help='Dataset percentage to use')
    parser.add_argument('--save_dir', type=str, default='saved_folder_test', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')


    # Latency parameters
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='belong to[-1, 0, 1]; -1  auto-select least-busy GPU, 0: use GPU 0, 1: use GPU 1')
    parser.add_argument('--gpu_power_limit', type=float, default=None, metavar='WATTS',
                        help='Set GPU power limit in Watts for auto-selected GPU (e.g. 200). Ignored if --gpu_id >= 0')
    parser.add_argument('--latency_samples_amount', type=int, default=10, help='Number of samples to average for latency')
    parser.add_argument('--latency_runs', type=int, default=30, help='Number of runs per sample for latency')
    parser.add_argument('--latency_warmup', type=int, default=5, help='Number of warmup runs per batch for latency')
    parser.add_argument('--record_latency', action='store_true', default=False, help='Record latency')

    return parser.parse_args()

def get_free_gpu():
    import subprocess
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits"
        ]
    )

    lines = result.decode().strip().split("\n")

    best_gpu = min(lines, key=lambda x: int(x.split(",")[1]))
    return best_gpu.split(",")[0]


def measure_latency(model, x, runs=100, warmup=20, device="cuda"):
    """
    Measure per-batch latency in milliseconds.
    Uses CUDA Events when running on GPU for stable timing.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    model.eval()
    model = model.to(device)
    if isinstance(x, (tuple, list)):
        x = tuple(item.to(device, non_blocking=True).contiguous() for item in x)
    else:
        x = x.to(device, non_blocking=True).contiguous()

    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(x, (tuple, list)):
                _ = model(*x)
            else:
                _ = model(x)

        if device == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(runs):
                if isinstance(x, (tuple, list)):
                    _ = model(*x)
                else:
                    _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            elapsed_ms = starter.elapsed_time(ender) / runs
        else:
            t0 = time.time()
            for _ in range(runs):
                if isinstance(x, (tuple, list)):
                    _ = model(*x)
                else:
                    _ = model(x)
            t1 = time.time()
            elapsed_ms = (t1 - t0) / runs * 1000.0

    return elapsed_ms


def test_model(model, dataloader, args, device):
    """Test function with comprehensive evaluation"""
    model.eval()
    
    # Initialize loss function
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss()
    
    val_loss = 0
    all_outputs = []
    all_labels = []
    
    with tqdm(dataloader, unit="batch", file=sys.stdout) as tepoch:
        for i, (img, _, _, beam, label) in enumerate(tepoch, 0):
            tepoch.set_description(f"Testing batch {i}")

            # Prepare data
            beam_downsampled = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
            label_downsampled = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)

            img = img.unsqueeze(2)
            d1,d2,d3,d4,d5 = img.shape
            image_batch = torch.cat([img[:,1-args.seq_length_student:, ...], torch.zeros(d1, args.num_pred, d3, d4, d5)], dim=1).to(device)
            
            # Adjust label based on test_num_pred - only take as many predictions as we're testing
            if args.num_pred <= label_downsampled.shape[1]:
                test_label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:, :args.num_pred]], dim=-1).to(device)
            else:
                print('Error: More predictions than available lables')
                exit()
            beam_opt = beam_downsampled[:, 0].type(torch.LongTensor).to(device)
            
            # Forward pass
            with torch.no_grad():
                student_outputs, _, _ = model(image_batch)
            
            student_outputs = student_outputs[:, -(args.num_pred + 1):, :]

            val_loss += criterion(student_outputs.reshape(-1, args.num_classes), test_label.flatten()).item()
            
            all_outputs.append(student_outputs)
            all_labels.append(test_label)
        
    # Concatenate all outputs and labels
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics
    topk_acc, total = calculate_topk_accuracy(all_outputs, all_labels)
    dba_score = calculate_dba_score(all_outputs, all_labels)
    
    val_loss /= len(dataloader)
    
    print(f'Test Loss: {val_loss:.4f}', flush=True)
    print("DBA-Score (Top-3):", dba_score)
    print('Top-K Accuracy:', flush=True)
    for k, acc in topk_acc.items():
        print(f'Top-{k}: {acc}', flush=True)


    return val_loss, topk_acc, dba_score


def main():
    """Main function for testing only"""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # Setup data
    current_dir = os.path.dirname(__file__)
    # parent_dir = os.path.abspath(os.path.join(current_dir, "..","..",".."))

    data_root = current_dir + '/dataset/scenario9'
    test_dir = os.path.join(data_root, args.test_csv_name)

    # Data preprocessing
    img_resize = transf.Resize((224, 224))
    proc_pipe = transf.Compose([transf.ToPILImage(), img_resize])

    test_dataset = DataFeed(
        data_root, test_dir, transform=proc_pipe, portion=args.dataset_pct
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers
    )
    latency_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    print(f'TestDataSize: {len(test_loader.dataset)}')

    # Setup device
    if args.use_gpu and torch.cuda.is_available():
        if args.gpu_id >= 0:
            torch.cuda.set_device(args.gpu_id)
            device = torch.device("cuda")
            print(f"Using specified GPU: {args.gpu_id}")
        else:
            best_gpu = int(get_free_gpu())
            torch.cuda.set_device(best_gpu)
            device = torch.device("cuda")
            print(f"Using least-busy GPU: {best_gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Define GRU parameters and load model
    GRU_PARAMS_teacher = (args.feature_size, args.gru_hidden_size, args.gru_num_layers_teacher)
    GRU_PARAMS_student = (args.feature_size, args.gru_hidden_size, args.gru_num_layers_student)
    print(f"=====Model architecture: {args.model_arch}=====")
    print(f"=====Sequence length: {args.seq_length_student}=====")
    print(f"=====KD mode: {args.kd_mode}=====")
    if args.model_arch == 'Teacher':
        model = ImageModalityNet_MHA(args.feature_size, args.num_classes, GRU_PARAMS_teacher, attention=args.attention)
        if args.attention:
            if args.kd_mode == 0:
                args.model_name = 'Teacher_withAtten.pth'
            else:
                args.model_name = 'Teacher_selfKD.pth'
        else:
            args.model_name = 'Teacher_noAtten.pth'

    elif args.model_arch == 'Student':

        model = StudentModalityNet(args.feature_size, args.num_classes, GRU_PARAMS_student)
        if args.seq_length_student == 8:
            if args.kd_mode == 0:
                args.model_name = 'StudentL8_noKD.pth'
            else:
                args.model_name = 'StudentL8_KD.pth'
        if args.seq_length_student == 5:
            if args.kd_mode == 0:
                args.model_name = 'StudentL5_noKD.pth'
            else:
                args.model_name = 'StudentL5_KD.pth'
        if args.seq_length_student == 3:
            if args.kd_mode == 0:
                args.model_name = 'StudentL3_noKD.pth'
            else:
                args.model_name = 'StudentL3_KD.pth'

    else:
        raise ValueError(f"Invalid model architecture: {args.model_arch}")


    model_path = os.path.join(current_dir, 'All_models',args.model_name)
    state_dict = torch.load(model_path, map_location=device)
    # if isinstance(state_dict, dict) and "state_dict" in state_dict:
    #     state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"=====Model loaded from: {model_path}=====")
    # Compute FLOPs and params once at model load
    flops = params = None
    try:
        if args.model_arch == 'Teacher':
            image_input = torch.randn(1, args.seq_length_teacher, 1, 224, 224).to(device)
        elif args.model_arch == 'Student':
            image_input = torch.randn(1, args.seq_length_student, 1, 224, 224).to(device)
        else:
            raise ValueError(f"Invalid model architecture: {args.model_arch}")
        flops, params = compute_flops(model, image_input, "Model")
 
    except Exception as exc:
        print(f"Warning: FLOPs/params computation failed: {exc}", flush=True)


    # Calculate parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"=====Model parameters: {total_params:,}\n")
    print(f"=====Trainable parameters: {trainable_params:,}\n")
    print(f"=====Non-trainable parameters: {total_params - trainable_params:,}\n")

    # Test model
    print('\nStart testing model...\n', flush=True)
    test_model(model, test_loader, args, device)

    # Measure inference latency at the end (per-batch latency in ms)
    try:
        latency_samples = []
        total_samples = len(latency_loader)
        sample_num = max(0, args.latency_samples_amount)
        if total_samples == 0 or sample_num == 0:
            raise StopIteration

        data_iter = iter(latency_loader)
        for _ in tqdm(range(sample_num), desc="Latency sampling", unit="sample", file=sys.stdout):
            try:
                img, _, _, beam, label = next(data_iter)
            except StopIteration:
                data_iter = iter(latency_loader)
                img, _, _, beam, label = next(data_iter)

            beam_downsampled = torch.floor(beam.float() / args.downsample_ratio).to(torch.int64)
            label_downsampled = torch.floor(label.float() / args.downsample_ratio).to(torch.int64)

            img = img.unsqueeze(2)
            image_batch = torch.cat([img, torch.zeros_like(img[:,:args.num_pred, ...])], dim=1).to(device)
            label = torch.cat([beam_downsampled[..., -1:], label_downsampled[:,:args.num_pred]], dim=-1).to(device)

            latency_ms = measure_latency(
                model,
                image_batch,
                runs=args.latency_runs,
                warmup=args.latency_warmup,
                device=device.type
            )
            latency_samples.append(latency_ms)

        if latency_samples:
            mean_latency = float(np.mean(latency_samples))
            std_latency = float(np.std(latency_samples))
            print(
                f"Inference latency (per sample): {mean_latency:.3f} ± {std_latency:.3f} ms "
                f"over {len(latency_samples)} samples",
                flush=True
            )
            # Record latency evaluation details
            if args.record_latency:
                latency_path = os.path.join(current_dir, "latency.txt")
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                try:
                    gpu_status = subprocess.check_output(["nvidia-smi"], text=True).strip()
                except Exception as exc:
                    gpu_status = f"nvidia-smi unavailable: {exc}"
                with open(latency_path, "a") as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"Time: {timestamp}\n")
                    f.write(f"Latency: {mean_latency:.3f} ± {std_latency:.3f} ms\n")
                    f.write("GPU status:\n")
                    f.write(gpu_status + "\n")
        else:
            print("Warning: empty dataloader, latency not measured.", flush=True)
    except StopIteration:
        print("Warning: empty dataloader, latency not measured.", flush=True)
    



if __name__ == "__main__":
    main()


