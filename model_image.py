#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Mengyuan Ma
@contact:mamengyuan410@gmail.com
@file: MyFuncs.py
@time: 2025/12/12 17:47
"""
from pytorch_model_summary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch

  
class ImageFeatureExtractor(nn.Module):
    def __init__(self, n_feature, in_channel=1):
        super(ImageFeatureExtractor, self).__init__()


        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=4, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))

        )

        # 全局平均池化层

        self.flatten = nn.Flatten()

        # 全连接层用于减少特征维度
        self.fc_layer = nn.Sequential(
             nn.Linear(64 * 7 * 7, 512),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(512, 128),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(128, 64),
             nn.ReLU(),
             nn.Dropout(0.2),
             nn.Linear(64, n_feature)
        )




    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()

        # 合并 batch 和时间维度以向量化处理
        frames = x.reshape(batch_size * seq_length, channels, height, width)

        # Apply CNN layers
        frame_features = self.cnn_layers(frames)

        # Flatten and process through FC layers
        frame_features = self.flatten(frame_features)
        frame_features = self.fc_layer(frame_features)

        # 还原为 (batch_size, seq_length, n_feature)
        spatial_features = frame_features.view(batch_size, seq_length, -1)
        return spatial_features



class ImageModalityNet_MHA(nn.Module):
    def __init__(self, feature_size, num_classes, gru_params, attention=True,num_heads=8):
        super(ImageModalityNet_MHA, self).__init__()
        '''
        This model uses only image as input for learning.
        '''
        self.name = 'ImageModalityNet_MHA'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"

        self.feature_extraction = ImageFeatureExtractor(feature_size) # image input only

  
        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.8, batch_first=True)

        self.attention = attention

        # Multi-head attention module
        self.num_heads = num_heads
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )


        # Add LayerNorm before GRU input
        self.layer_norm = nn.LayerNorm(gru_input_size)
        # Classifier
        self.classifier = nn.Sequential(
             nn.Linear(gru_hidden_size, 64),
             nn.ReLU(),
             nn.Dropout(0.5),
             nn.Linear(64, 64),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(64, num_classes)
        )



    def forward(self, image_batch):
        batch_size, seq_len, _, _, _ = image_batch.size()
        # Extract features using the feature extraction network

        features = self.feature_extraction(image_batch)

        # Apply LayerNorm to the features
        features = self.layer_norm(features)
        Seq_out, _ = self.GRU(features)

        if self.attention:
        # Apply multi-head attention to GRU output
            # Use self-attention where query, key, and value are all the GRU output
            attn_output, attn_weights = self.multihead_attention(
                query=Seq_out,
                key=Seq_out, 
                value=Seq_out
            )
            enhanced_seq_out = attn_output + Seq_out
        else:
            enhanced_seq_out = Seq_out

        Pred = self.classifier(enhanced_seq_out) # Final classification layer

        return Pred, features, enhanced_seq_out
    



class StudentModalityNet(nn.Module):
    """
    Student for image-only learning. Uses a lighter CNN + GRU.
    """
    def __init__(self, feature_size, num_classes, gru_params, width_multiplier=1.5, image_channels=1, num_heads=8,
        use_MHA=True,  use_channel_attention=True, use_spatial_attention=True):
        super(StudentModalityNet, self).__init__()
        self.use_channel_attention = use_channel_attention
        self.use_spatial_attention = use_spatial_attention
        self.use_MHA = use_MHA
        '''
        This model uses only image as input for learning.
        Minimal CNN architecture with GlobalAveragePool and GlobalMaxPool.
        '''
        self.name = 'StudentModalityNet'
        gru_input_size, gru_hidden_size, gru_num_layers = gru_params
        assert gru_input_size == feature_size, f"Error: gru_input_size ({gru_input_size}) must be equal to feature_size ({feature_size})"

        # Depthwise-separable conv block to reduce FLOPs
        def ds_conv_block(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # Lightweight CNN layers (moderate downsample + mid channels)
        # Targets ~5x-10x FLOPs reduction while preserving features.
        c1 = int(12*width_multiplier)
        c2 = int(24*width_multiplier)
        c3 = int(48*width_multiplier)
        c4 = int(96*width_multiplier)
        self.image_cnn_layers = nn.Sequential(
            nn.Conv2d(image_channels, c1, 3, stride=2, padding=1, bias=False),  # 224->112
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c1, c2, stride=1),   # 112->56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c2, c3, stride=1),   # 56->28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ds_conv_block(c3, c4, stride=1),   # 28->14
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Channel and spatial attention on CNN features (like teacher's ImageFeatureExtractor)
        reduction = max(c4 // 2, 1)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c4, reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, c4, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c4, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        # Global pooling layers
        # self.image_global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.image_global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Feature fusion and projection
        self.fusion_layer = nn.Sequential(
            nn.Linear(c4 , 64),  
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(64, feature_size)
        )

        self.GRU = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_size, num_layers=gru_num_layers,
                          dropout=0.8, batch_first=True)

        # Multi-head attention module

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=gru_hidden_size, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )

        # Add LayerNorm before GRU input
        self.layer_norm = nn.LayerNorm(gru_input_size)
        
        # Classifier
        self.classifier = nn.Sequential(
             nn.Linear(gru_hidden_size, 64),
             nn.ReLU(),
             nn.Dropout(0.3),
             nn.Linear(64, num_classes)
        )

    def forward(self, image_batch):
        B, T, C, H, W = image_batch.shape

        # Flatten time into batch
        img = image_batch.view(B*T, C, H, W)          # (B*T, 1, 224, 224)

        # CNN forward
        img_feat = self.image_cnn_layers(img)

        # Channel and spatial attention (same structure as teacher)
        if self.use_channel_attention:
            channel_att = self.channel_attention(img_feat)
            img_feat = img_feat * channel_att
        if self.use_spatial_attention:
            spatial_att = self.spatial_attention(img_feat)
            img_feat = img_feat * spatial_att

        # Global pool then flatten
        img_pooled = self.image_global_max_pool(img_feat).flatten(1)  # (B*T, c4)

        # Projection to feature_size, then reshape back to sequence
        fused_features = self.fusion_layer(img_pooled).view(B, T, -1)   # (B, T, feature_size)

        # LayerNorm + GRU
        features = self.layer_norm(fused_features)
        Seq_out, _ = self.GRU(features)

        
        
        if self.use_MHA:
        
            attn_output, attn_weights = self.multihead_attention(
                query=Seq_out,
                key=Seq_out, 
                value=Seq_out
            )

            enhanced_seq_out = attn_output + Seq_out
        else:
            enhanced_seq_out = Seq_out

        Pred = self.classifier(enhanced_seq_out)
        return Pred, features, enhanced_seq_out
    
