
"""
Attention-based Dual Margin Loss
================================
Integrates with InsightFace's PartialFC framework
Supports ablation studies with different attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from attention_modules import get_attention_module


class AttenDualMarginLoss(nn.Module):
    """
    Attention-based Dual Margin Loss

    Combines:
    - Adaptive positive margin (based on attention score)
    - Fixed negative margin
    - Support for different attention mechanisms
    """
    def __init__(
        self,
        scale=64.0,
        m1=0.5,           # Base positive margin
        m2=0.1,           # Negative margin
        attention_type='hybrid',  # baseline, dual_attention, uncertainty, channel, hybrid
        embedding_size=512,
        reduction=32,
    ):
        super(AttenDualMarginLoss, self).__init__()
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.attention_type = attention_type

        # Get attention module
        self.attention_net = get_attention_module(
            attention_type, embedding_size, reduction
        )

        print(f"[AttenDualMarginLoss] Attention Type: {attention_type}")
        print(f"[AttenDualMarginLoss] Scale: {scale}, M1: {m1}, M2: {m2}")

    def forward(self, logits, labels):
        """
        Args: 
            logits:  Cosine similarity scores (B, num_classes)
            labels: Ground truth labels (B,)

        Returns:
            Modified logits with dual margin applied
        """
        # This will be called from PartialFC_V2 with cosine similarities
        # We need to store embeddings separately for attention calculation
        raise NotImplementedError("Use AttenDualPartialFC instead")


class AttenDualPartialFC(nn.Module):
    """
    Partial FC with Attention-based Dual Margin
    Drop-in replacement for PartialFC_V2
    """
    def __init__(
        self,
        embedding_size=512,
        num_classes=85742,
        sample_rate=.0,
        scale=64.0,
        m1=0.5,
        m2=0.1,
        attention_type='hybrid',
        reduction=32,
    ):
        super(AttenDualPartialFC, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.scale = scale
        self.m1 = m1
        self.m2 = m2
        self.attention_type = attention_type

        # Class weights
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # Attention module
        self.attention_net = get_attention_module(
            attention_type, embedding_size, reduction
        )

        # For logging
        self.last_attention_score = None

        print(f"[AttenDualPartialFC] Initialized")
        print(f"  - Attention:  {attention_type}")
        print(f"  - Classes: {num_classes}")
        print(f"  - Scale: {scale}, M1: {m1}, M2: {m2}")

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Feature embeddings (B, embedding_size)
            labels: Ground truth labels (B,)

        Returns:
            loss: Cross entropy loss
        """
        # .Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)

        # 2.Compute cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)

        # 3.Get attention scores
        atten_score = self.attention_net(embeddings)  # (B, 1)
        self.last_attention_score = atten_score.mean().item()

        # 4.Clamp cosine for numerical stability
        cosine = torch.clamp(cosine, -.0 + 1e-7, .0 - 1e-7)

        # 5.Convert to angles
        theta = torch.acos(cosine)

        # 6.Calculate adaptive positive margin
        m1_dynamic = self.m1 * atten_score  # (B, 1)

        # 7.Create one-hot mask
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), .0)

        # 8.Apply dual margins
        # Target class: add positive margin
        target_angle = theta + m1_dynamic
        target_cosine = torch.cos(target_angle)

        # Non-target classes: subtract negative margin
        non_target_angle = theta - self.m2
        non_target_cosine = torch.cos(non_target_angle)

        # 9.Combine
        output = one_hot * target_cosine + (.0 - one_hot) * non_target_cosine

        # 10.Scale
        output = output * self.scale

        # 1.Compute loss
        loss = F.cross_entropy(output, labels)

        return loss

    def get_attention_score(self):
        """Return last computed attention score for logging"""
        return self.last_attention_score if self.last_attention_score else 0.0
