""" Contains type aliases for coref module """

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


EPSILON = 1e-7
LARGE_VALUE = 1000  # used instead of inf due to bug #16762 in pytorch

PRONOUNS = {'i', 'me', 'my', 'mine', 'myself',
            'we', 'us', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself',
            'they', 'them', 'their', 'theirs', 'themself', 'themselves',
            'this', 'these', 'that', 'those'}

Doc = Dict[str, Any]
Span = Tuple[int, int]


@dataclass
class CorefResult:
    # coref_scores: torch.Tensor = None                  # [n_words, k + 1]
    cost_is_mention: torch.Tensor = None                  # [n_words, k + 1]
    coref_indices: torch.Tensor = None                  # [n_words, k + 1]
    coref_y: torch.Tensor = None                       # [n_words, k + 1]

    word_clusters: List[List[int]] = None
    span_clusters: List[List[Span]] = None

    span_scores: torch.Tensor = None                   # [n_heads, n_words, 2]
    span_y: Tuple[torch.Tensor, torch.Tensor] = None   # [n_heads] x2

    coref_logits: torch.Tensor = None   
    cluster_logits: torch.Tensor = None   
    input_emb: torch.Tensor = None   
