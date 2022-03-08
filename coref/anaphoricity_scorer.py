""" Describes AnaphicityScorer, a torch module that for a matrix of
mentions produces their anaphoricity scores.
"""
import torch
import math
import copy

from coref import utils
from coref.config import Config
from coref.multi_head_attention import MultiheadAttention

def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SelfAttention(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 dropout_rate):
        super().__init__()
        self.to_q = torch.nn.Linear(in_features, in_features)
        self.to_k = torch.nn.Linear(in_features, in_features)
        self.to_v = torch.nn.Linear(in_features, in_features)
        self.bsz = 1
        self.num_heads = 1

    def _scaled_dot_product_attention(self,
        src,
        attn_mask = None,
        dropout_p = 0.0,
    ):
        k,q,v = self.to_k(src), self.to_q(src), self.to_v(src)
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn += attn_mask
        # if dropout_p > 0.0:
        #     attn = dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn.softmax(dim=-1), v)
        return output, attn
    
    def forward(self, src, attn_mask):
        if attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        attn_output, attn_output_weights = self._scaled_dot_product_attention(src, attn_mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(src.shape[1], self.bsz, src.shape[-1])
        attn_output_weights = attn_output_weights.view(self.bsz, self.num_heads, src.shape[1], src.shape[1])
        src, attn_weights = attn_output.transpose(0,1), attn_output_weights.sum(dim=1) / self.num_heads

        return src, attn_weights#, attn_mask


class AnaphoricityScorer(torch.nn.Module):
    """ Calculates anaphoricity scores by passing the inputs into a FFNN """
    def __init__(self,
                 in_features: int,
                 config: Config, args):
        super().__init__()
        # self.not_cluster = torch.nn.Embedding(1, in_features)
        # self.is_choose = torch.nn.Embedding(1, in_features)
        self_attn_layer = SelfAttention(in_features, config.dropout_rate)
        self.layers = _get_clones(self_attn_layer, args.layernum)
        self.dropout = torch.nn.Dropout(config.dropout_rate/args.dropdiv)
        # self.relu = torch.nn.ReLU(inplace=False)
        # self.layers_weights = torch.nn.Linear(len(self.layers), 1)
        # self.is_choose_classifier = torch.nn.Linear(in_features*2, 1)

        # hidden_size = config.hidden_size
        # if not config.n_hidden_layers:
        #     hidden_size = in_features
        # layers = []
        # for i in range(config.n_hidden_layers):
        #     layers.extend([torch.nn.Linear(hidden_size if i else in_features,
        #                                    hidden_size),
        #                    torch.nn.LeakyReLU(),
        #                    torch.nn.Dropout(config.dropout_rate)])
        # self.hidden = torch.nn.Sequential(*layers)
        # self.out = torch.nn.Linear(hidden_size, out_features=1)


    def forward(self, *,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                all_mentions: torch.Tensor,
                cls, free_tokens
                # mentions_batch: torch.Tensor,
                # pw_batch: torch.Tensor,
                # top_indices_batch: torch.Tensor,
                # top_rough_scores_batch: torch.Tensor,
                ) -> torch.Tensor:
        """ Builds a pairwise matrix, scores the pairs and returns the scores.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb]
            mentions_batch (torch.Tensor): [batch_size, mention_emb]
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb]
            top_indices_batch (torch.Tensor): [batch_size, n_ants]
            top_rough_scores_batch (torch.Tensor): [batch_size, n_ants]

        Returns:
            torch.Tensor [batch_size, n_ants + 1]
                anaphoricity scores for the pairs + a dummy column
        """
        # [batch_size, n_ants, pair_emb]
        # mentions_with_start_tokens = torch.cat([self.is_choose.weight, self.not_cluster.weight, all_mentions], 0)
        mentions_with_start_tokens = torch.cat([cls, all_mentions, free_tokens], 0)
        # mentions_with_start_tokens = all_mentions
        src = mentions_with_start_tokens.unsqueeze(0)
        causal_mask = torch.triu(float("-inf")*torch.ones(mentions_with_start_tokens.shape[0], mentions_with_start_tokens.shape[0], device=all_mentions.device), diagonal=1)
        if free_tokens.shape[0] > 0:
            causal_mask[:,-free_tokens.shape[0]:] = causal_mask[0,0]
        # causal_mask[0,0] = causal_mask[1,0]
        # causal_mask[1,0] = causal_mask[1,1]
        # causal_mask[1,1] = causal_mask[0,0]
        attn_weights = [[]] * len(self.layers)
        # cls_scores = torch.zeros(1, device=src.device)
        # layers_weights = self.layers_weights.weight.softmax(1).transpose(0,1)
        final_mask = torch.triu(float("-inf")*torch.ones(mentions_with_start_tokens.shape[0], mentions_with_start_tokens.shape[0], device=all_mentions.device), diagonal=0)
        for i,layer in enumerate(self.layers):
            src, attn_weights[i] = layer(src,attn_mask=causal_mask)
            # attn_weights[i] = attn_weights[i] + final_mask
            # attn_weights[i] = attn_weights[i][:,1:].softmax(dim=-1)
            # is_choose = self.is_choose_classifier(torch.cat([src, out],-1)).sigmoid()
            # attn_weights[i] = attn_weights[i].clamp(max=1.0, min=0.0)
            # cls_score = attn_weights[i][:,1:,0]
            # cls_scores = cls_scores + cls_score
            # attn_weights[i] = attn_weights[i][:, 1:] + causal_mask[1:]
            # attn_weights[i] = attn_weights[i] + final_mask[1:]
            # attn_weights[i][:,0,0]=0
            # attn_weights[i] = attn_weights[i] * (1-cls_score.unsqueeze(-1))
            # out = (out+src) / 2
            # attn_weights[i] = self.relu(attn_weights[i]) + causal_mask[1:,:].unsqueeze(0)
        # for i in range(len(self.self_attn)):
        #     src, attn_weights = self.self_attn[i](src, src, src, need_weights=True, \
        #         attn_mask=causal_mask)
        attn_weights = torch.cat(attn_weights, 0)
        # is_choose = attn_weights[:,2:,0].sigmoid()
        # choose_attn_weights = is_choose.unsqueeze(-1) * attn_weights[:,2:,1:]
        attn_weights = torch.sum(attn_weights, dim=0)
        attn_weights[:,0] = attn_weights[:,0]/len(self.layers)
        # attn_weights[:,0] = attn_weights[:,0] / len(self.layers)
        # attn_weights = attn_weights.squeeze(0)
                            #   key_padding_mask=src_key_padding_mask)[0]
        # pair_matrix = self._get_pair_matrix(
        #     all_mentions, mentions_batch, pw_batch, top_indices_batch)

        # # [batch_size, n_ants]
        # scores = top_rough_scores_batch + self._ffnn(pair_matrix)
        # scores = utils.add_dummy(scores, eps=True)
        # attn_weights[torch.arange(0,attn_weights.shape[0]), torch.arange(0,attn_weights.shape[0])] = 0

        # return torch.cat([(cls_scores/len(self.layers)).transpose(0,1), attn_weights], dim=-1) + final_mask[1:,:]
        if free_tokens.shape[0] > 0:
            return self.dropout(attn_weights[1:-free_tokens.shape[0],:-free_tokens.shape[0]])# + final_mask[1:]
        else:
            return self.dropout(attn_weights[1:])# + final_mask[1:]

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.out(self.hidden(x))
        return x.squeeze(2)

    @staticmethod
    def _get_pair_matrix(all_mentions: torch.Tensor,
                         mentions_batch: torch.Tensor,
                         pw_batch: torch.Tensor,
                         top_indices_batch: torch.Tensor,
                         ) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb],
                all the valid mentions of the document,
                can be on a different device
            mentions_batch (torch.Tensor): [batch_size, mention_emb],
                the mentions of the current batch,
                is expected to be on the current device
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb],
                pairwise features of the current batch,
                is expected to be on the current device
            top_indices_batch (torch.Tensor): [batch_size, n_ants],
                indices of antecedents of each mention

        Returns:
            torch.Tensor: [batch_size, n_ants, pair_emb]
        """
        emb_size = mentions_batch.shape[1]
        n_ants = pw_batch.shape[1]

        a_mentions = mentions_batch.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = all_mentions[top_indices_batch]
        similarity = a_mentions * b_mentions

        out = torch.cat((a_mentions, b_mentions, similarity, pw_batch), dim=2)
        return out
