# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_coref: float = 1, num_queries=100):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_coref = cost_coref
        self.num_queries = num_queries

    @torch.no_grad()
    def forward(self, coref_logits, res, gold_matrix):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "coref_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "cluster_logits": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if gold_matrix.shape[1] == 0 or torch.sum(gold_matrix) == 0:
            return False, False

        coref_logits = coref_logits[:self.num_queries] # [num_queries, tokens_without_dummy]

        real_cluster_target_rows = torch.sum(gold_matrix, -1) > 0
        real_cluster_target = gold_matrix[real_cluster_target_rows]

        cluster_repeated = real_cluster_target[:,:-1].unsqueeze(1).repeat(1, coref_logits.shape[0], 1)
        coref_logits_repeated = coref_logits.unsqueeze(0).repeat(real_cluster_target.shape[0], 1, 1)
        clamped_logits1 = coref_logits_repeated[:,:,:-1].clamp(max=1.0)
        cost_coref = torch.mean(F.binary_cross_entropy(clamped_logits1, \
            cluster_repeated, reduction='none'), -1) + \
                coref_logits_repeated[:,:, -1]
        cost_coref = cost_coref.transpose(0,1)

        total_cost = self.cost_coref * cost_coref
        # total_cost = self.cost_coref * cost_coref
        total_cost = total_cost.cpu()
        indices = linear_sum_assignment(total_cost)
        ind1, ind2 = indices

        return torch.as_tensor(ind1, dtype=torch.int64), torch.as_tensor(ind2, dtype=torch.int64)


def build_matcher(args=None, num_queries=-1):
    if args is None:
        return HungarianMatcher(num_queries=num_queries)
    else:    
        return HungarianMatcher(cost_is_cluster=args.cost_is_cluster, cost_coref=args.cost_coref, num_queries=args.num_queries)
