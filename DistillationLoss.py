#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact: mamengyuan410@gmail.com
@file: DistillationLoss.py
@time: 2025/12/22 12:28
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Enhanced Knowledge Distillation Loss supporting both logits and feature-based distillation, including relational KD
    """
    def __init__(self, task_criterion, args):
        super(DistillationLoss, self).__init__()
        self.task_criterion = task_criterion
        self.args = args
        self.kl_div = nn.KLDivLoss(reduction='none')  # Changed to 'none' to handle per-sample weights
        self.mse_loss = nn.MSELoss()



    def feature_distillation_loss(self, student_features, teacher_features):
        """
        Calculate feature distillation loss using Euclidean distance (MSE)
        """
        # Ensure features have the same shape
        if student_features.shape == teacher_features.shape:     
            feature_loss = self.mse_loss(student_features, teacher_features)
        else:
            print(f"Student features shape: {student_features.shape}, Teacher features shape: {teacher_features.shape}")
            raise ValueError("Student and teacher features must have the same shape")
        return feature_loss
    
    def select_pairs(self, batch_size, k):
        """
        Select k pairs for each anchor sample in the batch
        Args:
            batch_size: size of the batch
            k: number of pairs per anchor
        Returns:
            pairs: tensor of shape [batch_size, k, 2] containing anchor-positive pairs
        """
        pairs = []
        for i in range(batch_size):
            # Get all possible positive indices (excluding the anchor itself)
            positive_indices = list(range(batch_size))
            positive_indices.remove(i)
            
            # Randomly select k pairs (or all available if k > available)
            k_actual = min(k, len(positive_indices))
            if k_actual > 0:
                selected_positives = torch.randperm(len(positive_indices))[:k_actual]
                selected_indices = [positive_indices[idx] for idx in selected_positives]
                
                # Create pairs [anchor, positive]
                anchor_pairs = [[i, j] for j in selected_indices]
                pairs.extend(anchor_pairs)
        
        return torch.tensor(pairs) if pairs else torch.empty(0, 2, dtype=torch.long)
    
    def compute_euclidean_distance(self, features, pairs):
        """
        Compute normalized Euclidean distances for given pairs
        Args:
            features: tensor of shape [batch_size, seq_len, feature_dim]
            pairs: tensor of shape [num_pairs, 2] containing indices
        Returns:
            distances: tensor of shape [num_pairs] containing normalized distances
        """
        if pairs.numel() == 0:
            return torch.empty(0, device=features.device)
            
        # Flatten features for distance computation: [batch_size, seq_len * feature_dim]
        features_flat = features.reshape(features.size(0), -1)
        
        # Get feature vectors for pairs
        anchor_features = features_flat[pairs[:, 0]]  # [num_pairs, seq_len * feature_dim]
        positive_features = features_flat[pairs[:, 1]]  # [num_pairs, seq_len * feature_dim]
        
        # Compute Euclidean distances
        distances = torch.norm(anchor_features - positive_features, p=2, dim=1)
        
        # Normalize distances by mean of all d_ij in batch
        mean_distance = distances.mean() if distances.numel() > 0 else torch.tensor(1.0, device=features.device)
        if mean_distance > 0:
            distances = distances / mean_distance
        
        return distances
    
    def compute_cosine_distance(self, features, pairs):
        """
        Compute cosine distances (1 - cosine_similarity) for given pairs
        Args:
            features: tensor of shape [batch_size, seq_len, feature_dim]
            pairs: tensor of shape [num_pairs, 2] containing indices
        Returns:
            distances: tensor of shape [num_pairs] containing cosine distances
        """
        if pairs.numel() == 0:
            return torch.empty(0, device=features.device)
            
        # Flatten features for distance computation: [batch_size, seq_len * feature_dim]
        features_flat = features.reshape(features.size(0), -1)
        
        # Get feature vectors for pairs
        anchor_features = features_flat[pairs[:, 0]]  # [num_pairs, seq_len * feature_dim]
        positive_features = features_flat[pairs[:, 1]]  # [num_pairs, seq_len * feature_dim]
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(anchor_features, positive_features, dim=1)
        
        # Convert to cosine distance (1 - cosine_similarity)
        cosine_distances = 1 - cos_sim
        
        return cosine_distances
    
    def relational_knowledge_distillation_loss(self, student_features, teacher_features):
        """
        Compute Relational Knowledge Distillation loss
        Args:
            student_features: tensor of shape [batch_size, seq_len, feature_dim]
            teacher_features: tensor of shape [batch_size, seq_len, feature_dim]
        Returns:
            rkd_loss: scalar tensor representing the RKD loss
        """
        batch_size = student_features.size(0)
        
        # Select pairs for relational learning
        pairs = self.select_pairs(batch_size, self.args.rkd_pairs_per_anchor)
        
        if pairs.numel() == 0:
            return torch.tensor(0.0, device=student_features.device)
        
        pairs = pairs.to(student_features.device)
        
        # Compute distances for student features
        student_euclidean = self.compute_euclidean_distance(student_features, pairs)
        student_cosine = self.compute_cosine_distance(student_features, pairs)
        
        # Compute distances for teacher features
        teacher_euclidean = self.compute_euclidean_distance(teacher_features, pairs)
        teacher_cosine = self.compute_cosine_distance(teacher_features, pairs)
        
        # Compute distance-based loss (MSE between normalized distances)
        distance_loss = self.mse_loss(student_euclidean, teacher_euclidean)
        
        # Compute angle-based loss (MSE between cosine distances)
        angle_loss = self.mse_loss(student_cosine, teacher_cosine)
        
        # Combine losses with weights
        rkd_loss = self.args.rkd_distance_weight * distance_loss + self.args.rkd_angle_weight * angle_loss
        
        return rkd_loss
    
    def forward(self, student_logits, teacher_logits, targets, 
                student_input_features=None, teacher_input_features=None,
                student_output_features=None, teacher_output_features=None,
                 current_alpha=None):
        """
        Args:
            student_logits: logits from student model [batch_size * seq_len, num_classes]
            teacher_logits: logits from teacher model [batch_size * seq_len, num_classes]
            targets: ground truth labels [batch_size * seq_len]
            student_input_features: input features from student model [batch_size, seq_len, feature_size]
            teacher_input_features: input features from teacher model [batch_size, seq_len, feature_size]
            student_output_features: output features from student model [batch_size, seq_len, hidden_size]
            teacher_output_features: output features from teacher model [batch_size, seq_len, hidden_size]
            input_shape_mapping: shape mapping network for input features
        """
        # Task loss (standard cross-entropy or focal loss)
        task_loss = self.task_criterion(student_logits, targets)
        
        # Initialize distillation loss
        distillation_loss = torch.tensor(0.0, device=student_logits.device)
        
        if self.args.kd_mode == 0:
            # No knowledge distillation, only task loss
            total_loss = task_loss
            return total_loss, task_loss, distillation_loss
        
        elif self.args.kd_mode == 1:
            # Standard logits-based knowledge distillation
            student_soft = F.log_softmax(student_logits / self.args.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.args.temperature, dim=1)
            
            # Calculate KL divergence
            kl_loss = self.kl_div(student_soft, teacher_soft)  # [batch_size * seq_len, num_classes]
            distillation_loss = kl_loss.sum(dim=1).mean()  # sum over classes, mean over batch
            
            distillation_loss = distillation_loss * (self.args.temperature ** 2)
        
        elif self.args.kd_mode == 2:
            # Relational Knowledge Distillation using output features
            if student_output_features is None or teacher_output_features is None:
                raise ValueError("Output features required for relational KD mode")

            distillation_loss = self.relational_knowledge_distillation_loss(student_output_features, teacher_output_features)
            # distillation_loss *= self.args.feature_loss_weight
            
        # Use current_alpha if provided, otherwise use the default alpha from args
        alpha = current_alpha if current_alpha is not None else self.args.alpha
        total_loss = (1 - alpha) * task_loss + alpha * distillation_loss   
        return total_loss, task_loss, distillation_loss

