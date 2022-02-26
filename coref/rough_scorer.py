""" Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import List, Tuple

import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers.models.bert.modeling_bert import ACT2FN
import torch.nn.functional as F

from coref.config import Config


class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob=0.3):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """
    def __init__(self, features: int, config: Config, bert_config):
        super().__init__()
        self.ffnn_size = 3072
        self.mention_mlp = FullyConnectedLayer(bert_config, bert_config.hidden_size, self.ffnn_size)
        self.mention_classifier = Linear(self.ffnn_size, 1)
        self.k_menprop = config.topk_lambda

        # self.dropout = torch.nn.Dropout(config.dropout_rate)
        # self.bilinear = torch.nn.Linear(features, features)

        self.k = config.rough_k

    def forward(self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                mentions: torch.Tensor,
                word_clusters: List
                ):
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        mention_reps = self.mention_mlp(mentions)
        mention_logits = self.mention_classifier(mention_reps).squeeze(-1)
        mention_logits = mention_logits.sigmoid()
        top_scores, indices = torch.topk(mention_logits,
                                         k=int(self.k_menprop * len(mention_logits)),
                                         dim=0, sorted=False)
        indices = indices.sort()[0]
        top_indices = indices.reshape(1,-1).repeat(mentions.shape[0],1)


        antece_mask = torch.arange(mentions.shape[0])
        antece_mask = antece_mask.unsqueeze(1) - antece_mask.unsqueeze(0)
        antece_mask = antece_mask[:,indices]
        indices_vector = torch.tensor([i in indices for i in range(mentions.shape[0])]).to(torch.float)
        indices_mask = indices_vector.unsqueeze(1).repeat(1,indices.shape[0])
        pair_mask = (antece_mask > 0).to(torch.float) * (indices_mask > 0).to(torch.float)
        pair_mask = torch.log(pair_mask)
        pair_mask = pair_mask.to(mentions.device)

        # bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T) + \
        #     (mention_logits.unsqueeze(0).repeat(mentions.shape[0],1) * mention_logits.unsqueeze(1).repeat(1,mentions.shape[0]))

        # rough_scores = pair_mask + bilinear_scores

        gold_indices = [gw for gc in word_clusters for gw in gc]
        cost_is_mention = torch.tensor(0., device=mention_logits.device)
        if len(gold_indices) > 0:
            gold_probs = mention_logits[gold_indices]
            cost_is_mention = F.binary_cross_entropy(gold_probs, torch.ones_like(gold_probs))
        junk_probs = mention_logits[[i for i in range(len(mention_logits)) if i not in gold_indices]]
        cost_is_mention += F.binary_cross_entropy(junk_probs, torch.zeros_like(junk_probs))
        return pair_mask, top_indices, indices, cost_is_mention*.3
        # return *self._prune(rough_scores), indices, cost_is_mention*.3

        # pair_mask = torch.arange(mentions.shape[0])
        # pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        # pair_mask = torch.log((pair_mask > 0).to(torch.float))
        # pair_mask = pair_mask.to(mentions.device)

        # bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)

        # rough_scores = pair_mask + bilinear_scores

        # return self._prune(rough_scores)

    def _prune(self,
               rough_scores: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects top-k rough antecedent scores for each mention.

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        top_scores, indices = torch.topk(rough_scores,
                                         k=min(self.k, len(rough_scores)),
                                         dim=1, sorted=False)
        return top_scores, indices
