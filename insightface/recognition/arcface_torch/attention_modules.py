
"""
Custom Attention Modules for Ablation Studies
==============================================
Supports the following configurations:
- baseline: No attention (standard ArcFace)
- dual_attention: Original AttentionBlock
- uncertainty: UncertaintyAttention only
- channel:  ChannelAttentionBlock only  
- hybrid: HybridAttentionBlock (Channel + Uncertainty)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Baseline Dual Attention Block
    Input: 512 embedding
    Output:  Scalar score between 0.5 and .5
    """
    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        raw_score = self.sigmoid(x)
        attention_score = 0.5 + raw_score
        return attention_score


class ChannelAttentionBlock(nn.Module):
    """
    Channel Attention using Squeeze-and-Excitation
    """
    def __init__(self, in_features, reduction=64):
        super(ChannelAttentionBlock, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )
        self.final_score = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        importance = self.se(x)
        x_refined = x * importance
        out = self.final_score(x_refined)
        raw_score = self.sigmoid(out)
        return 0.5 + raw_score


class UncertaintyAttention(nn.Module):
    """
    Uncertainty-based Attention using Log-Variance prediction
    """
    def __init__(self, in_features):
        super(UncertaintyAttention, self).__init__()
        self.fc_log_var = nn.Linear(in_features, 1)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        log_var = self.fc_log_var(x)
        log_var = self.bn(log_var)
        sigma = torch.exp(0.5 * log_var)
        atten_score = .0 / (sigma + 0.1)
        atten_score = torch.clamp(atten_score, 0.5, .5)
        return atten_score


class HybridAttentionBlock(nn.Module):
    """
    Hybrid Attention:  Channel Attention + Uncertainty Estimation
    Step 1: Channel Attention (Filter noise from features)
    Step 2: Uncertainty Estimation (Calculate confidence based on clean features)
    """
    def __init__(self, in_features, reduction=64):
        super(HybridAttentionBlock, self).__init__()

        # Channel Attention (The Filter)
        self.channel_fc1 = nn.Linear(in_features, in_features // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.channel_fc2 = nn.Linear(in_features // reduction, in_features, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Uncertainty Estimation (The Judge)
        self.uncertainty_fc = nn.Linear(in_features, 1)
        self.uncertainty_bn = nn.BatchNorm1d(1)

    def forward(self, x):
        # .Channel Attention - Clean the features
        weights = self.channel_fc1(x)
        weights = self.relu(weights)
        weights = self.channel_fc2(weights)
        weights = self.sigmoid(weights)
        x_refined = x * weights

        # 2.Uncertainty Estimation
        log_var = self.uncertainty_fc(x_refined)
        log_var = self.uncertainty_bn(log_var)
        sigma = torch.exp(0.5 * log_var)

        # 3.Convert to attention score [0.5, .5]
        raw_score = .0 / (.0 + sigma)
        attention_score = 0.5 + raw_score

        return attention_score


class NoAttention(nn.Module):
    """
    Baseline: No attention, returns constant score of .0
    """
    def __init__(self, in_features):
        super(NoAttention, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        batch_size = x.size(0)
        return torch.ones(batch_size, 1, device=x.device)


def get_attention_module(attention_type, in_features, reduction=32):
    """
    Factory function to get attention module based on type

    Args:
        attention_type: One of ['baseline', 'dual_attention', 'uncertainty', 'channel', 'hybrid']
        in_features: Input feature dimension (usually 512)
        reduction: Reduction ratio for channel attention

    Returns: 
        Attention module
    """
    attention_modules = {
        'baseline': NoAttention,
        'dual_attention':  AttentionBlock,
        'uncertainty':  UncertaintyAttention,
        'channel': ChannelAttentionBlock,
        'hybrid':  HybridAttentionBlock,
    }

    if attention_type not in attention_modules:
        raise ValueError(f"Unknown attention type: {attention_type}."
                        f"Choose from {list(attention_modules.keys())}")

    module_class = attention_modules[attention_type]

    if attention_type in ['channel', 'hybrid']: 
        return module_class(in_features, reduction=reduction)
    else: 
        return module_class(in_features)
