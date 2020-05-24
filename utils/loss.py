import numpy as np
import torch
import numbers
import torch.nn as nn


def all_diffs(a, b):
    """ Returns a tensor of all combinations of a - b.

    Args:
        a (2D tensor): A batch of vectors shaped (B1, F).
        b (2D tensor): A batch of vectors shaped (B2, F).

    Returns:
        The matrix of all pairwise differences between all vectors in `a` and in
        `b`, will be of shape (B1, B2).

    """
    return torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)


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

    diffs = all_diffs(a, b)
    if metric == 'sqeuclidean':
        return torch.sum(diffs ** 2, dim=-1)
    elif metric == 'euclidean':
        return torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-12)
    elif metric == 'cityblock':
        return torch.sum(torch.abs(diffs), dim=-1)
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))


class BatchHardLoss(nn.Module):
    def __init__(self, margin, metric, safe_radius=0.25):
        super(BatchHardLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        self.safe_radius = safe_radius

    def forward(self, anchor, positive, dist_keypts):
        pids = torch.FloatTensor(np.arange(len(anchor)))
        if torch.cuda.is_available():
            pids = pids.cuda()
        # if self.metric == 'euclidean':
            # distance = torch.sqrt(2 - 2 * torch.matmul(anchor, positive.transpose(0, 1)))
            # return batch_hard(distance, pids, margin=self.margin)
        dist = cdist(anchor, positive, metric=self.metric)
        dist_keypts = np.eye(dist_keypts.shape[0]) * 10 + dist_keypts.detach().cpu().numpy()
        add_matrix = torch.zeros_like(dist)
        add_matrix[np.where(dist_keypts < self.safe_radius)] += 10
        dist = dist + add_matrix
        return batch_hard(dist, pids, margin=self.margin)


def batch_hard(dists, pids, margin=1, batch_precision_at_k=None):
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
    if isinstance(margin, numbers.Real):
        loss = torch.max(furthest_positive - 0.1, torch.zeros_like(diff)) + torch.max(1.4 - closest_negative, torch.zeros_like(diff))
    elif margin == 'soft':
        loss = torch.nn.Softplus()(diff)
    elif margin == 'dynamic':
        margin = float(torch.diag(dists).mean())
        loss = torch.max(diff + margin, torch.zeros_like(diff))
    else:
        raise NotImplementedError('The margin {} is not implemented in batch_hard'.format(margin))

    average_negative = (torch.sum(dists, dim=-1) - furthest_positive) / (dists.shape[0] - 1)
    
    if batch_precision_at_k is None:
        # return torch.mean(loss), accuracy, diff, 0, dists
        return torch.mean(loss), accuracy, furthest_positive.tolist(), average_negative.tolist(), 0, dists
