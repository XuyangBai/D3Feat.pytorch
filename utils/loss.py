import numpy as np
import torch
import numbers
import torch.nn as nn
import torch.nn.functional as F


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.

    Args:
        a (2D tensor): The left-hand side, shaped (B1, F).
        b (2D tensor): The right-hand side, shaped (B2, F).
        metric (string): Which distance metric to use, see notes.

    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    if metric == 'cosine':
        return torch.sqrt(2 - 2 * torch.matmul(a, b.T))
    elif metric == 'arccosine':
        return torch.acos(torch.matmul(a, b.T))
    else:
        diffs = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)
        if metric == 'sqeuclidean':
            return torch.sum(diffs ** 2, dim=-1)
        elif metric == 'euclidean':
            return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
        elif metric == 'cityblock':
            return torch.sum(torch.abs(diffs), dim=-1)
        else:
            raise NotImplementedError(
                'The following metric is not implemented by `cdist` yet: {}'.format(metric))


class ContrastiveLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.4, metric='euclidean', safe_radius=0.25):
        super(ContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.metric = metric
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts):
        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        dist = cdist(anchor, positive, metric=self.metric)
        dist_keypts = np.eye(dist_keypts.shape[0]) * 10 + dist_keypts.detach().cpu().numpy()
        add_matrix = torch.zeros_like(dist)
        add_matrix[np.where(dist_keypts < self.safe_radius)] += 10
        dist = dist + add_matrix
        return self.calculate_loss(dist, pids)


    def calculate_loss(self, dists, pids):
        """Computes the batch-hard loss from arxiv.org/abs/1703.07737.

        Args:
            dists (2D tensor): A square all-to-all distance matrix as given by cdist.
            pids (1D tensor): The identities of the entries in `batch`, shape (B,).
                This can be of any type that can be compared, thus also a string.
            margin: The value of the margin if a number, alternatively the string
                'soft' for using the soft-margin formulation, or `None` for not
                using a margin at all.

        Returns:
            A 1D tensor of shape (B,) containing the loss value for each sample.
        """
        # generate the mask that mask[i, j] reprensent whether i th and j th are from the same identity.
        # torch.equal is to check whether two tensors have the same size and elements
        # torch.eq is to computes element-wise equality
        same_identity_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        # negative_mask = np.logical_not(same_identity_mask)

        # dists * same_identity_mask get the distance of each valid anchor-positive pair.
        furthest_positive, _ = torch.max(dists * same_identity_mask.float(), dim=1)
        # here we use "dists +  10000*same_identity_mask" to avoid the anchor-positive pair been selected.
        closest_negative, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=1)
        # closest_negative_row, _ = torch.min(dists + 1e5 * same_identity_mask.float(), dim=0)
        # closest_negative = torch.min(closest_negative_col, closest_negative_row)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]
        loss = torch.max(furthest_positive - self.pos_margin, torch.zeros_like(diff)) + torch.max(self.neg_margin - closest_negative, torch.zeros_like(diff))

        average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)
        
        return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), 0, dists


class CircleLoss(nn.Module):
    def __init__(self, dist_type='cosine', log_scale=10, safe_radius=0.10, pos_margin=0.1, neg_margin=1.4):
        super(CircleLoss, self).__init__()
        self.log_scale = log_scale
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_margin
        self.neg_optimal = neg_margin
        self.dist_type = dist_type
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts):
        dists = cdist(anchor, positive, metric=self.dist_type)

        pids = torch.FloatTensor(np.arange(len(anchor))).to(anchor.device)
        pos_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        neg_mask = dist_keypts > self.safe_radius


        furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
        closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)
        average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)
        diff = furthest_positive - closest_negative
        accuracy = (diff < 0).sum() * 100.0 / diff.shape[0]

        pos = dists - 1e5 * neg_mask.float()
        pos_weight = (pos - self.pos_optimal).detach()
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight)
        lse_positive_row = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-1)
        lse_positive_col = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight, dim=-2)

        neg = dists + 1e5 * (~neg_mask).float()
        neg_weight =  (self.neg_optimal - neg).detach()
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight)
        lse_negative_row = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-1)
        lse_negative_col = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight, dim=-2)

        loss_col = F.softplus(lse_positive_row + lse_negative_row) / self.log_scale
        loss_row = F.softplus(lse_positive_col + lse_negative_col) / self.log_scale
        loss = loss_col + loss_row

        return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), 0, dists


class DetLoss(nn.Module):
    def __init__(self, metric='euclidean'):
        super(DetLoss, self).__init__()
        self.metric = metric
    
    def forward(self, dists, anc_score, pos_score):
        pids = torch.FloatTensor(np.arange(len(anc_score))).to(anc_score.device)
        pos_mask = torch.eq(torch.unsqueeze(pids, dim=1), torch.unsqueeze(pids, dim=0))
        neg_mask = torch.logical_not(pos_mask)

        furthest_positive, _ = torch.max(dists * pos_mask.float(), dim=1)
        closest_negative, _ = torch.min(dists + 1e5 * pos_mask.float(), dim=1)

        loss = (furthest_positive - closest_negative) * (anc_score + pos_score).squeeze(-1)
        return torch.mean(loss)