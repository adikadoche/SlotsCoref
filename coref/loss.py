""" Describes the loss function used to train the model, which is a weighted
sum of NLML and BCE losses. """

import torch
import torch.nn.functional as F
import logging

from coref.matcher import build_matcher

logger = logging.getLogger(__name__)

class MatchingLoss(torch.nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_queries, eos_coef=0.1, cost_is_cluster=1, cost_coref=5, cost_is_mention=1):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = build_matcher(num_queries=num_queries)
        self.cost_is_cluster = cost_is_cluster
        self.cost_coref = cost_coref
        self.cost_is_mention = cost_is_mention
        self.num_queries = num_queries
        self.eos_coef = eos_coef
        # self._bce_module = torch.nn.BCEWithLogitsLoss()

    def forward(self, res, doc):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        gold_matrix, coref_logits, dist_matrix, goldgold_dist_mask, junkgold_dist_mask = \
            self._create_gold_matrix(res, doc)
        # Retrieve the matching between the outputs of the last layer and the targets
        matched_predicted_cluster_id, matched_gold_cluster_id = \
            self.matcher(coref_logits, res, gold_matrix)

        costs_parts = {}
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        cluster_logits = res.cluster_logits.squeeze() # [num_queries+num_junk_queries]
        #TODO: normalize according to number of clusters? (identical to DETR)

        gold_is_cluster = torch.zeros_like(cluster_logits)
        gold_is_cluster[:self.num_queries] = 1
        weight_cluster = self.eos_coef * torch.ones_like(cluster_logits)
        weight_cluster[:self.num_queries] = 1
        cost_is_cluster = F.binary_cross_entropy(cluster_logits, gold_is_cluster, weight=weight_cluster)

        # gold_input = input_[res.coref_y == 1]
        # junk_input = input_[res.coref_y == 0]
        # cost_is_mention =  self._bce_module(torch.clamp(gold_input, min=-50, max=50), torch.ones_like(gold_input)) + \
        #     self._bce_module(torch.clamp(junk_input, min=-50, max=50), torch.ones_like(junk_input))
        cost_is_mention = res.cost_is_mention

        cost_coref = torch.tensor(0., device=coref_logits.device)
        cost_junk = torch.tensor(0., device=coref_logits.device)
        if matched_predicted_cluster_id is not False:  #TODO: add zero rows?
            permuted_coref_logits = coref_logits[matched_predicted_cluster_id.numpy()]
            junk_coref_logits = coref_logits[[x for x in range(coref_logits.shape[0]) if x not in matched_predicted_cluster_id.numpy()]]
            permuted_gold = gold_matrix[matched_gold_cluster_id.numpy()]
            permuted_gold = permuted_gold[:, :-1]
            junk_gold = torch.zeros_like(junk_coref_logits[:, :-1])
            premuted_cluster_logits = cluster_logits[matched_predicted_cluster_id.numpy()]
            junk_cluster_logits = cluster_logits[[x for x in range(coref_logits.shape[0]) if x not in matched_predicted_cluster_id.numpy()]]
            clamped_logits = (premuted_cluster_logits.unsqueeze(1) * permuted_coref_logits[:, :-1]).clamp(max=1.0)
            cost_coref = F.binary_cross_entropy(clamped_logits, permuted_gold, reduction='mean') + \
                          torch.mean(permuted_coref_logits[:, -1] * premuted_cluster_logits)
            clamped_junk_logits = (junk_cluster_logits.unsqueeze(1) * junk_coref_logits[:, :-1]).clamp(max=1.0)
            cost_junk = F.binary_cross_entropy(clamped_junk_logits, junk_gold, reduction='mean') + \
                                                    torch.mean(junk_coref_logits[:, -1] * junk_cluster_logits)
        elif coref_logits.shape[1] > 0:
            clamped_logits = coref_logits.clamp(max=1.0)
            cost_coref = F.binary_cross_entropy(clamped_logits, torch.zeros_like(coref_logits), reduction='mean')

        dist_matrix = dist_matrix.clamp(min=0.0, max=1.0)
        goldgold_denom = torch.sum(goldgold_dist_mask)
        goldgold_denom = torch.maximum(torch.ones_like(goldgold_denom), goldgold_denom)
        log_incluster_dists = dist_matrix * goldgold_dist_mask
        log_outcluster_dists = (1-dist_matrix) * junkgold_dist_mask
        cost_coref += F.binary_cross_entropy(log_incluster_dists, torch.zeros_like(log_incluster_dists), \
            reduction='sum') / goldgold_denom
        junkgold_denom = torch.sum(junkgold_dist_mask)
        junkgold_denom = torch.maximum(torch.ones_like(junkgold_denom), junkgold_denom)
        cost_junk += F.binary_cross_entropy(log_outcluster_dists, torch.zeros_like(log_outcluster_dists), \
            reduction='sum') / junkgold_denom

        costs_parts['loss_is_cluster']= self.cost_is_cluster * cost_is_cluster.detach().cpu()
        costs_parts['loss_is_mention']= self.cost_is_mention * cost_is_mention.detach().cpu()
        costs_parts['loss_coref'] = self.cost_coref * cost_coref.detach().cpu()
        costs_parts['loss_junk'] = self.cost_coref * cost_junk.detach().cpu()
        total_cost = self.cost_coref * cost_coref + self.cost_is_cluster * cost_is_cluster + \
            self.cost_coref * cost_junk + self.cost_is_mention * cost_is_mention
        return total_cost, costs_parts

    def _create_gold_matrix(self, res, doc):
        gold_words = [gw for gc in doc['word_clusters'] for gw in gc]
        gold_matrix = torch.zeros(self.num_queries, len(gold_words)+1, device=res.input_emb.device)
        if self.num_queries < len(doc['word_clusters']):
            logger.info("in utils, exceeds num_queries with length {}".format(len(doc['word_clusters'])))
        for cluster_id, cluster in enumerate(doc['word_clusters']):
            if cluster_id >= self.num_queries:
                continue
            for word in cluster:
                word_index = gold_words.index(word)
                assert word_index >= 0
                gold_matrix[cluster_id, word_index] = 1

        mentions_list = res.coref_indices
        junk_mentions_indices = torch.tensor([i for i, m in enumerate(mentions_list) if m not in gold_words], \
            dtype=torch.long, device=gold_matrix.device)
        common_mentions = [m.item() for m in mentions_list if m in gold_words]

        common_predict_ind = torch.zeros(len(common_mentions), dtype=torch.long, device=gold_matrix.device)
        common_gold_ind = torch.zeros(len(gold_words)+1, device=gold_matrix.device)

        ind = 0
        for i in range(len(gold_words)):
            if gold_words[i] in common_mentions:
                for j in range(len(mentions_list)):
                    if mentions_list[j] == gold_words[i]:
                        common_predict_ind[ind] = j
                        common_gold_ind[i] = 1
                        ind += 1

        new_coref_logits = torch.zeros(gold_matrix.shape[1], res.coref_logits.shape[0], device=gold_matrix.device)
        new_coref_logits[common_gold_ind == 1] = torch.index_select(res.coref_logits.transpose(0,1), 0, common_predict_ind)         
        new_coref_logits[-1] = torch.sum(res.coref_logits[:, junk_mentions_indices], 1)
        new_coref_logits = new_coref_logits.transpose(0,1)

        ordered_inputs = res.input_emb[torch.cat([common_predict_ind, junk_mentions_indices],0)]
        common_gold_matrix = gold_matrix[:,common_gold_ind==1]

        indices = torch.arange(0,len(common_mentions),device=gold_matrix.device).reshape(1,-1).repeat(torch.sum(torch.sum(gold_matrix,-1)>0),1)
        indices = torch.cat([torch.arange(0,indices.shape[0],device=gold_matrix.device).reshape(-1,1,1).repeat(1,indices.shape[1],1), indices.unsqueeze(-1)],-1)
        x=torch.masked_select(indices[:,:,0], common_gold_matrix[:torch.sum(torch.sum(gold_matrix,-1)>0)]==1)
        y=torch.masked_select(indices[:,:,1], common_gold_matrix[:torch.sum(torch.sum(gold_matrix,-1)>0)]==1)

        ordered_inputs[:len(y)] = ordered_inputs[y]
        ordered_inputs = F.normalize(ordered_inputs, dim=-1)
        dist_matrix = torch.matmul(ordered_inputs, ordered_inputs.transpose(0,1))
        dist_matrix = 1 - (dist_matrix + 1) / 2

        goldgold_dist_mask = torch.zeros_like(dist_matrix)
        goldgold_dist_mask[:len(x),:len(y)] = common_gold_matrix[x][:,y]
        junkgold_dist_mask = 1 - goldgold_dist_mask.clone()
        is_junk_mention = torch.arange(0,dist_matrix.shape[0],device=res.input_emb.device) >= len(common_mentions)
        junkgold_dist_mask = junkgold_dist_mask * \
            ~(is_junk_mention.reshape(-1,1).repeat(1,dist_matrix.shape[1]) * \
                is_junk_mention.reshape(1,-1).repeat(dist_matrix.shape[0],1))
        return gold_matrix, new_coref_logits, \
            dist_matrix, goldgold_dist_mask, junkgold_dist_mask
