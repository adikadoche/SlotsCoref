""" see __init__.py """

import os
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import wandb

import numpy as np      # type: ignore
import jsonlines        # type: ignore
import torch

from coref import bert, conll, utils
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.cluster_checker import ClusterChecker
from coref.const import EPSILON, CorefResult, Doc
from coref.pairwise_encoder import PairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor
from coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS
from coref.utils import GraphNode
from coref.word_encoder import WordEncoder
from coref.const import EPSILON

logger = logging.getLogger(__name__)

class CorefModel(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """Combines all coref modules together to find coreferent spans.

    Attributes:
        config (coref.config.Config): the model's configuration,
            see config.toml for the details
        epochs_trained (int): number of epochs the model has been trained for
        trainable (Dict[str, torch.nn.Module]): trainable submodules with their
            names used as keys
        training (bool): used to toggle train/eval modes

    Submodules (in the order of their usage in the pipeline):
        tokenizer (transformers.AutoTokenizer)
        bert (transformers.AutoModel)
        we (WordEncoder)
        rough_scorer (RoughScorer)
        pw (PairwiseEncoder)
        a_scorer (AnaphoricityScorer)
        sp (SpanPredictor)
    """
    def __init__(self,
                 args,
                 config,
                 epochs_trained: int = 0):
        """
        A newly created model is set to evaluation mode.

        Args:
            config_path (str): the path to the toml file with the configuration
            section (str): the selected section of the config file
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        super().__init__()
        self.args = args
        self.device = args.device
        self.config = config
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        if not args.is_debug:  
            wb_config = wandb.config
            logger.info(f"---- CONFIG ----")
            for key, val in vars(self.config).items():
                logger.info(f"{key} - {val}")
                wb_config[key] = val
        self.epochs_trained = epochs_trained
        self._docs: Dict[str, List[Doc]] = {}
        self.bert, self.tokenizer = bert.load_bert(self.config, self.device)
        self.pw = PairwiseEncoder(self.config).to(self.device)

        bert_emb = self.bert.config.hidden_size
        pair_emb = bert_emb #* 3 + self.pw.shape

        # pylint: disable=line-too-long
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(self.device)
        self.we = WordEncoder(bert_emb, self.config).to(self.device)
        self.rough_scorer = RoughScorer(bert_emb, self.config, self.bert.config).to(self.device)
        self.sp = SpanPredictor(bert_emb, self.config.sp_embedding_size).to(self.device)

        self.trainable: Dict[str, torch.nn.Module] = {
            "bert": self.bert, "we": self.we,
            "rough_scorer": self.rough_scorer,
            "pw": self.pw, "a_scorer": self.a_scorer,
            "sp": self.sp
        }

    def forward(self,  # pylint: disable=too-many-locals
            doc: Doc, epoch=0
            ) -> CorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.

        Returns:
            CorefResult (see const.py)
        """
        res = CorefResult()
        # Encode words with bert
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        words, cluster_ids, cls = self.we(doc, self._bertify(doc))

        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]
        scores_mask, top_indices, res.menprop, res.cost_is_mention = self.rough_scorer(words, doc['word_clusters'])
        # top_rough_scores, top_indices = self.rough_scorer(words, doc['word_clusters'])

        # Get pairwise features [n_words, n_ants, n_pw_features]
        # pw = self.pw(top_indices, doc)

        # batch_size = self.config.a_scoring_batch_size
        # a_scores_lst: List[torch.Tensor] = []

        # for i in range(0, len(words), batch_size):
        #     pw_batch = pw[i:i + batch_size]
        #     words_batch = words[i:i + batch_size]
        #     top_indices_batch = top_indices[i:i + batch_size]
        #     top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
        scores = self.a_scorer(
            all_mentions=words[res.menprop], cls=cls
        )
        coref_scores = torch.ones(top_indices.shape[0], scores.shape[1], device=top_indices.device) * EPSILON
        coref_scores[res.menprop] = scores
        coref_scores = coref_scores + scores_mask
        # coref_scores[:,0][coref_scores[:,0]<EPSILON] = EPSILON
        res.coref_scores = utils.add_dummy(coref_scores, eps=True)
        # a_scores_lst.append(a_scores_batch)
        # res.coref_scores = coref_scores


        # coref_scores  [n_spans, n_ants]
        # res.coref_scores = torch.cat(a_scores_lst, dim=0)
        res.span_scores, res.span_y = self.sp.get_training_data(doc, words)
        # gold_mentions = [m for c in doc['word_clusters'] for m in c]
        # cluster_id = [i for i,c in enumerate(doc['word_clusters']) for m in c]
        # goldgold_dist_mask = torch.zeros_like(res.coref_scores)
        # ind=0
        # if len(cluster_id)>0:
        #     for i in range(max(cluster_id)):
        #         goldgold_dist_mask[ind:ind+cluster_id.count(i), ind+1:ind+1+cluster_id.count(i)] = 1
        #         ind+= cluster_id.count(i)
        # junkgold_dist_mask = 1 - goldgold_dist_mask.clone()
        # is_junk_mention_row = torch.arange(0,res.coref_scores.shape[0], device=goldgold_dist_mask.device) >= len(gold_mentions)
        # is_junk_mention_col = torch.arange(0,res.coref_scores.shape[1], device=goldgold_dist_mask.device) >= len(gold_mentions)+1
        # junkgold_dist_mask = junkgold_dist_mask * \
        #     ~(is_junk_mention_row.reshape(-1,1).repeat(1,res.coref_scores.shape[1]) * \
        #         is_junk_mention_col.reshape(1,-1).repeat(res.coref_scores.shape[0],1))
        # ordered_scores = res.coref_scores.clone()
        # # mask = ordered_scores==float("-inf")
        # ordered_scores[ordered_scores==float("-inf")] = 0
        # ordered_scores[:,1:] = ordered_scores[:,1:] + ordered_scores[:,1:].transpose(0,1)
        # ordered_rows = torch.tensor(gold_mentions + [x for x in range(ordered_scores.shape[0]) if x not in gold_mentions])
        # ordered_cols = torch.tensor([gm + 1 for gm in gold_mentions] + [x for x in range(ordered_scores.shape[1]) if x not in gold_mentions])
        # ordered_scores = ordered_scores[ordered_rows][:,ordered_cols]

        # res.coref_y = self._get_ground_truth(
        #     cluster_ids, top_indices, (top_rough_scores > float("-inf")))
        res.coref_y = self._get_ground_truth(
            cluster_ids, top_indices, (scores_mask > float("-inf")))
        if epoch % 3 == 2 or not self.training:
            res.word_clusters = self._clusterize(doc, res.coref_scores,
                                                top_indices)
            # onehotpredict = torch.zeros(len(words), len(words)+1)
            # gold_mentions = [m for c in doc['word_clusters'] for m in c]
            # ordered_rows = torch.tensor(gold_mentions + [x for x in range(onehotpredict.shape[0]) if x not in gold_mentions])
            # ordered_cols = torch.tensor([gm for gm in gold_mentions] + [x for x in range(onehotpredict.shape[1]) if x not in gold_mentions])
            # for c in res.word_clusters:
            #     for x in c:
            #         for y in c:
            #             onehotpredict[x,y] = 1
            # onehotpredict = onehotpredict[ordered_rows][:,ordered_cols]
            

            res.span_clusters = self.sp.predict(doc, words, res.word_clusters)

        return res


    # ========================================================= Private methods

    def _bertify(self, doc: Doc) -> torch.Tensor:
        subwords_batches = bert.get_subwords_batches(doc, self.config,
                                                     self.tokenizer)

        special_tokens = np.array([self.tokenizer.cls_token_id,
                                   self.tokenizer.sep_token_id,
                                   self.tokenizer.pad_token_id])
        subword_mask = ~(np.isin(subwords_batches, special_tokens))

        subwords_batches_tensor = torch.tensor(subwords_batches,
                                               device=self.device,
                                               dtype=torch.long)
        subword_mask_tensor = torch.tensor(subword_mask,
                                           device=self.device)

        # Obtain bert output for selected batches only
        attention_mask = (subwords_batches != self.tokenizer.pad_token_id)
        out = self.bert(
            subwords_batches_tensor,
            attention_mask=torch.tensor(
                attention_mask, device=self.device))[0]

        # [n_subwords, bert_emb]
        return (out[subword_mask_tensor], out[0,0])

    def _clusterize(self, doc: Doc, scores: torch.Tensor, top_indices: torch.Tensor):
        scores[:,0][scores[:,0]<EPSILON] = EPSILON
        antecedents = scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(scores))[not_dummy]
        antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

        nodes = [GraphNode(i) for i in range(len(doc["cased_words"]))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            nodes[i].link(nodes[j])
            assert nodes[i] is not nodes[j]

        clusters = []
        for node in nodes:
            if len(node.links) > 0 and not node.visited:
                cluster = []
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.append(current_node.id)
                    stack.extend(link for link in current_node.links if not link.visited)
                assert len(cluster) > 1
                clusters.append(sorted(cluster))
        return sorted(clusters)

    @staticmethod
    def _get_ground_truth(cluster_ids: torch.Tensor,
                          top_indices: torch.Tensor,
                          valid_pair_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-gold words have cluster id of zero.
            top_indices: tensor of shape [n_words, n_ants],
                indices of antecedents of each word
            valid_pair_map: boolean tensor of shape [n_words, n_ants],
                whether for pair at [i, j] (i-th word and j-th word)
                j < i is True

        Returns:
            tensor of shape [n_words, n_ants + 1] (dummy added),
                containing 1 at position [i, j] if i-th and j-th words corefer.
        """
        y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
        y[y == 0] = -1                                 # -1 for non-gold words
        y = utils.add_dummy(y)                         # [n_words, n_cands + 1]
        y = (y == cluster_ids.unsqueeze(1))            # True if coreferent
        # For all rows with no gold antecedents setting dummy to True
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)


    def _tokenize_docs(self, path: str) -> List[Doc]:
        print(f"Tokenizing documents at {path}...", flush=True)
        out: List[Doc] = []
        filter_func = TOKENIZER_FILTERS.get(self.config.bert_model,
                                            lambda _: True)
        token_map = TOKENIZER_MAPS.get(self.config.bert_model, {})
        with jsonlines.open(path, mode="r") as data_f:
            for doc in data_f:
                doc["span_clusters"] = [[tuple(mention) for mention in cluster]
                                   for cluster in doc["span_clusters"]]
                word2subword = []
                subwords = []
                word_id = []
                for i, word in enumerate(doc["cased_words"]):
                    tokenized_word = (token_map[word]
                                      if word in token_map
                                      else self.tokenizer.tokenize(word))
                    tokenized_word = list(filter(filter_func, tokenized_word))
                    word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
                    subwords.extend(tokenized_word)
                    word_id.extend([i] * len(tokenized_word))
                doc["word2subword"] = word2subword
                doc["subwords"] = subwords
                doc["word_id"] = word_id
                out.append(doc)
        print("Tokenization OK", flush=True)
        return out
