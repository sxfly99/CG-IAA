import os
import requests

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

EPISILON=1e-10

class JSDLoss(torch.nn.Module):

  def __init__(self, weight=1.0, softmax_sign=False):
    super(JSDLoss, self).__init__()
    self.weight = weight
    self.softmax_sign = softmax_sign

  def forward(self, p, q):
    if self.softmax_sign is False:
      p = F.softmax(p, dim=1)
      q = F.softmax(q, dim=1)

    loss1 = p * torch.log(p / q + EPISILON)
    loss1 = loss1.sum(1)

    loss2 = q * torch.log(q / p + EPISILON)
    loss2 = loss2.sum(1)

    loss = loss1 + loss2
    loss = loss.mean()*self.weight
    return loss


class SoftLabelCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean", normalize_targets=True):
        assert reduction in (
            "mean",
            "sum",
        ), "Argument `reduction` only supports `mean` and `sum`"

        super().__init__()

        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalize_targets = normalize_targets
        self.eps = torch.finfo(torch.float32).eps

    @staticmethod
    def convert_to_one_hot(targets, n_classes):
        one_hot_targets = torch.zeros(
            (targets.size(0), n_classes), dtype=torch.long, device=targets.device
        )
        one_hot_targets.scatter_(1, targets.long().view(-1, 1), 1)
        return one_hot_targets

    def forward(self, targets, scores):
        """for N examples and C classes
        - scores: N x C these are raw outputs (without softmax/sigmoid)
        - targets: N x C or N corresponding targets

        Target elements set to ignore_index contribute 0 loss.

        Samples where all entries are ignore_index do not contribute to the loss
        reduction.
        """

        assert targets.size(0) == scores.size(
            0
        ), "`targets` and `scores` should have the same batch size"

        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
            mask = targets.ne(self.ignore_index).float()  # mask out `ignore_index`
        else:
            mask = targets.sum(-1, keepdim=True).ne(0).float()  # mask out zero rows
            # print(mask)

        if targets.size(1) == 1:
            targets = self.convert_to_one_hot(targets, scores.size(1))
        targets = targets.float() * mask

        if self.normalize_targets:
            targets /= self.eps + targets.sum(dim=1, keepdim=True)

        per_sample_per_target_loss = -targets * F.log_softmax(scores, dim=-1)
        per_sample_loss = torch.sum(per_sample_per_target_loss, -1)
        loss = per_sample_loss.sum()
        # perform reduction
        if self.reduction == "mean":
            # normalize based on the number of samples with > 0 non-ignored targets
            loss /= torch.sum(torch.sum(mask, -1) > 0).clamp(min=1)
        return loss



def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]