""" Describes SlorsScorer, a torch module that for a matrix of
mentions produces their slots attention scores.
"""
import torch
from torch import nn
from torch.nn import init

from coref import utils
from coref.config import Config


class SlotsScorer(torch.nn.Module):
    """ Calculates slots scores by passing the inputs into a slots attention """
    def __init__(self,
                 in_features: int,
                 config: Config,
                 num_queries: int,
                 random_queries: bool):
        super().__init__()
        self.num_queries = num_queries
        self.random_queries = random_queries
        hidden_size = config.hidden_size
        if not config.n_hidden_layers:
            hidden_size = in_features
        layers = []
        for i in range(config.n_hidden_layers):
            layers.extend([torch.nn.Linear(hidden_size if i else in_features,
                                           hidden_size),
                           torch.nn.LeakyReLU(),
                           torch.nn.Dropout(config.dropout_rate)])
        self.hidden = torch.nn.Sequential(*layers)
        self.slots_query_embed = nn.Embedding(num_queries, hidden_size)
        self.num_slots = self.num_queries
        self.slots_iters = 3
        self.slots_eps = 1e-8
        self.slots_scale = hidden_size ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, hidden_size))
        init.xavier_uniform_(self.slots_logsigma)

        self.slots_to_q = nn.Linear(hidden_size, hidden_size)
        self.slots_to_k = nn.Linear(hidden_size, hidden_size)
        self.slots_to_v = nn.Linear(hidden_size, hidden_size)

        self.slots_gru = nn.GRUCell(hidden_size, hidden_size)

        self.slots_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        self.slots_norm_input = nn.LayerNorm(hidden_size)
        self.norm_slots = nn.LayerNorm(hidden_size)
        self.slots_norm_pre_ff = nn.LayerNorm(hidden_size)

        self.slots_mlp_classifier = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_size / 2), 1),
            nn.Sigmoid()
        ) 

    def forward(self, *,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
                all_mentions: torch.Tensor,
                mentions_batch: torch.Tensor,
                pw_batch: torch.Tensor,
                top_indices_batch: torch.Tensor,
                top_rough_scores_batch: torch.Tensor,
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
        pair_matrix = self._get_pair_matrix(
            all_mentions, mentions_batch, pw_batch, top_indices_batch)

        # [batch_size, n_ants]
        inputs, cluster_logits, coref_logits = self._slot_attention(\
            self._ffnn(pair_matrix), top_rough_scores_batch > float('-inf'))
        scores = top_rough_scores_batch
        scores = utils.add_dummy(scores, eps=True)

        return scores

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.hidden(x)
        return x

    def _slot_attention(self, input_emb, mask):
        words, ants, emb, device = *input_emb.shape, input_emb.device

        input_emb = input_emb.reshape(1, -1, emb)
        mask = mask.type(torch.long).reshape(1, -1, 1)

        input_emb = mask * input_emb + (1-mask) * torch.zeros_like(input_emb)

        if self.random_queries:
            mu = self.slots_mu.expand(1, self.num_slots, -1)
            sigma = self.slots_logsigma.exp().expand(1, self.num_slots, -1)

            slots = mu + sigma * torch.randn(mu.shape, device=device)
        else:
            slots = self.slots_query_embed.weight.unsqueeze(0)

        inputs = self.slots_norm_input(input_emb)
        k, v = self.slots_to_k(inputs), self.slots_to_v(inputs)

        for _ in range(self.slots_iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.slots_to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.slots_scale
            attn = dots.softmax(dim=1) + self.slots_eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.slots_gru(
                updates.reshape(-1, emb),
                slots_prev.reshape(-1, emb)
            )

            slots = slots.reshape(1, -1, emb)
            slots = slots + self.slots_mlp(self.slots_norm_pre_ff(slots))

        slots = self.norm_slots(slots)
        q = self.slots_to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.slots_scale
        coref_logits = (dots.softmax(dim=1) + self.slots_eps).clamp(max=1.0)
        cluster_logits = self.slots_mlp_classifier(slots)

        input_emb = input_emb.reshape(words, ants, -1)
        coref_logits = coref_logits.reshape(self.num_queries, words, ants)

        return inputs, cluster_logits, coref_logits

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
