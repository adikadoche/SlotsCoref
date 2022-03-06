""" see __init__.py """

from datetime import datetime
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import wandb
import networkx as nx

import numpy as np      # type: ignore
import jsonlines        # type: ignore
import toml
import torch
from tqdm import tqdm   # type: ignore
import transformers     # type: ignore

from coref import bert, conll, utils
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.cluster_checker import ClusterChecker
from coref.config import Config
from coref.const import EPSILON, CorefResult, Doc
from coref.loss import CorefLoss
from coref.pairwise_encoder import PairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor
from coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS
from coref.utils import GraphNode
from coref.word_encoder import WordEncoder
from coref.metrics import CorefEvaluator, MentionEvaluator
from coref.coref_analysis import print_predictions
from coref.const import EPSILON

logger = logging.getLogger(__name__)

class CorefModel:  # pylint: disable=too-many-instance-attributes
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
                 config_path: str,
                 section: str,
                 epochs_trained: int = 0):
        """
        A newly created model is set to evaluation mode.

        Args:
            config_path (str): the path to the toml file with the configuration
            section (str): the selected section of the config file
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        self.args = args
        self.device = args.device
        self.config = CorefModel._load_config(config_path, section)
        self.config.output_dir = os.path.join(self.config.output_dir, \
            datetime.now().strftime(f"%m_%d_%Y_%H_%M_%S")+'_'+args.run_name)
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
        self._build_model()
        self._build_optimizers()
        self._set_training(False)
        self._coref_criterion = CorefLoss(self.config.bce_loss_weight)
        self._span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @property
    def training(self) -> bool:
        """ Represents whether the model is in the training mode """
        return self._training

    @training.setter
    def training(self, new_value: bool):
        if self._training is new_value:
            return
        self._set_training(new_value)

    # ========================================================== Public methods

    @torch.no_grad()
    def evaluate(self, 
                 data_split: str = "dev",
                 word_level_conll: bool = False,
                 docs = None
                 ) -> Tuple[float, Tuple[float, float, float]]:
        """ Evaluates the modes on the data split provided.

        Args:
            data_split (str): one of 'dev'/'test'/'train'
            word_level_conll (bool): if True, outputs conll files on word-level

        Returns:
            mean loss
            span-level LEA: f1, precision, recal
        """
        self.training = False
        if docs is None:
            docs = self._get_docs(self.config.__dict__[f"predict_file"])            
        cluster_evaluator = CorefEvaluator()       
        word_cluster_evaluator = CorefEvaluator()       
        # cluster_graph_evaluator = CorefEvaluator()
        mention_evaluator = MentionEvaluator()
        men_prop_evaluator = MentionEvaluator()
        # w_checker = ClusterChecker()
        # s_checker = ClusterChecker()
        running_loss = 0.0
        losses_parts = {'loss_is_cluster':0.0, 'loss_is_mention':0.0, 'loss_coref':0.0}
        s_correct = 0
        s_total = 0

        all_gold_clusters = []
        all_predicted_clusters = []
        all_tokens = []
        coref_scores = []
        menprops = []

        # with conll.open_(self.config, self.epochs_trained, data_split) \
        #         as (gold_f, pred_f):
        pbar = tqdm(docs, unit="docs", ncols=0)
        for doc in pbar:
            res = self.run(doc)

            c_loss = self._coref_criterion(res.coref_scores, res.coref_y).item()
            running_loss += c_loss + res.cost_is_mention.item()
            losses_parts['loss_coref'] += c_loss
            losses_parts['loss_is_mention'] += res.cost_is_mention.item()

            if res.span_y:
                pred_starts = res.span_scores[:, :, 0].argmax(dim=1)
                pred_ends = res.span_scores[:, :, 1].argmax(dim=1)
                # men propos scores
                s_correct += ((res.span_y[0] == pred_starts) * (res.span_y[1] == pred_ends)).sum().item()
                s_total += len(pred_starts)


            # mentions scored
            # w_checker.add_predictions(doc["word_clusters"], res.word_clusters)
            # w_lea = w_checker.total_lea

            # final scores
            # s_checker.add_predictions(doc["span_clusters"], res.span_clusters)
            cluster_evaluator.update([res.span_clusters] ,[doc["span_clusters"]])
            word_cluster_evaluator.update([[[(res.word_clusters[i][j],res.word_clusters[i][j]) \
                for j in range(len(res.word_clusters[i]))] for i in range(len(res.word_clusters))]], \
                    [[[(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for j in range(len(doc['word_clusters'][i]))]\
                        for i in range(len(doc['word_clusters']))]])
            # cluster_evaluator.update([[[(res.word_clusters[i][j],res.word_clusters[i][j]) \
            # for j in range(len(res.word_clusters[i]))] for i in range(len(res.word_clusters))]], \
            #         [[[(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for j in range(len(doc['word_clusters'][i]))]\
            #             for i in range(len(doc['word_clusters']))]])
            # cluster_graph_evaluator.update([[[(res.graph_clusters[i][j],res.graph_clusters[i][j]) \
            # for j in range(len(res.graph_clusters[i]))] for i in range(len(res.graph_clusters))]], \
            #         [[[(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for j in range(len(doc['word_clusters'][i]))]\
            #             for i in range(len(doc['word_clusters']))]])
            mention_evaluator.update([(res.word_clusters[i][j],res.word_clusters[i][j]) \
                for i in range(len(res.word_clusters)) for j in range(len(res.word_clusters[i]))], \
                    [(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for i in range(len(doc['word_clusters']))\
                        for j in range(len(doc['word_clusters'][i]))])
            men_prop_evaluator.update([(res.menprop[i].item(),res.menprop[i].item()) \
                for i in range(len(res.menprop))], \
                    [(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for i in range(len(doc['word_clusters']))\
                        for j in range(len(doc['word_clusters'][i]))])
            all_predicted_clusters.append(res.span_clusters)
            all_gold_clusters.append(doc["span_clusters"])
            all_tokens.append(doc["cased_words"])
            coref_scores.append(res.coref_scores.detach().cpu())
            menprops.append(res.menprop.detach().cpu())

            del res

        print()

        print(f"============ {data_split} EXAMPLES ============")
        print_predictions(all_gold_clusters, all_predicted_clusters, all_tokens, self.args.max_eval_print)
        # train_p, train_r, train_f1 = cluster_graph_evaluator.get_prf()
        # logger.info('Cluster f1, precision, recall: {}'.format(\
        #     (train_f1, train_p, train_r)))

        losses_parts['loss_is_cluster'] = s_correct / s_total
        losses_parts['loss_is_mention'] /= len(docs)
        losses_parts['loss_coref'] /= len(docs)
        return running_loss / len(docs), losses_parts, cluster_evaluator, word_cluster_evaluator, mention_evaluator, men_prop_evaluator, \
            (coref_scores, menprops)

    @torch.no_grad()
    def evaluate_graph(self, 
                 coref_scores,
                 menprops,
                 docs = None
                 ) -> Tuple[float, Tuple[float, float, float]]:
        """ Evaluates the modes on the data split provided.

        Args:
            data_split (str): one of 'dev'/'test'/'train'
            word_level_conll (bool): if True, outputs conll files on word-level

        Returns:
            mean loss
            span-level LEA: f1, precision, recal
        """
        self.training = False
        cluster_graph_evaluator = CorefEvaluator()
        word_graph_evaluator = MentionEvaluator()

        pbar = tqdm(docs, unit="docs", ncols=0)
        for ind, doc in enumerate(pbar):
            weights = torch.zeros(len(doc['cased_words'])+1, len(doc['cased_words'])+1)
            rows = torch.arange(1,len(doc['cased_words'])+1).unsqueeze(1).repeat(1,coref_scores[ind].shape[1])
            top_indices_shifted = torch.cat([\
                torch.zeros(1, dtype=torch.long),\
                    menprops[ind]+1],-1).unsqueeze(0).repeat(coref_scores[ind].shape[0], 1)
            weights[rows, top_indices_shifted] = coref_scores[ind].detach().cpu()
            weights = weights.transpose(0,1)
            weights[weights<0] = 0
            real_indices = torch.cat([torch.zeros(1, dtype=torch.long), menprops[ind].detach().cpu()+1])
            real_indices_rows = real_indices.unsqueeze(1).repeat(1,real_indices.shape[0])
            real_indices_cols = real_indices.unsqueeze(0).repeat(real_indices.shape[0], 1)
            weights = weights[real_indices_rows, real_indices_cols]
            G = nx.from_numpy_matrix(weights.numpy())
            all_part2 = []
            part1 = [1,len(menprops[ind])]
            x=0
            while len(part1) > 1:
                split_score, (part1, part2) = nx.minimum_cut(G, 0, part1[-1], capacity='weight')
                part1 = list(part1)
                if len(part2)>1:
                    all_part2.append(sorted([menprops[ind][p2-1].item() for p2 in part2]))
                    # print(f"split no. {x} with score {split_score}, part1: {part1}, part2: {part2}")
                G = G.subgraph(part1)
                x+=1
            graph_clusters = all_part2

            cluster_graph_evaluator.update([[[(graph_clusters[i][j],graph_clusters[i][j]) \
            for j in range(len(graph_clusters[i]))] for i in range(len(graph_clusters))]], \
                    [[[(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for j in range(len(doc['word_clusters'][i]))]\
                        for i in range(len(doc['word_clusters']))]])
            word_graph_evaluator.update([(graph_clusters[i][j],graph_clusters[i][j]) \
                for i in range(len(graph_clusters)) for j in range(len(graph_clusters[i]))], \
                    [(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for i in range(len(doc['word_clusters']))\
                        for j in range(len(doc['word_clusters'][i]))])

        print()

        return cluster_graph_evaluator, word_graph_evaluator

    def load_weights(self,
                     path: Optional[str] = None,
                     ignore: Optional[Set[str]] = None,
                     map_location: Optional[str] = None,
                     noexception: bool = False) -> None:
        """
        Loads pretrained weights of modules saved in a file located at path.
        If path is None, the last saved model with current configuration
        in output_dir is loaded.
        Assumes files are named like {configuration}_(e{epoch}_{time})*.pt.
        """
        if path is None:
            pattern = rf"{self.config.section}_e(\d+).*\.pt"
            files = []
            for f in os.listdir(self.config.output_dir):
                match_obj = re.match(pattern, f)
                if match_obj:
                    files.append((int(match_obj.group(1)), f))
            if not files:
                if noexception:
                    print("No weights have been loaded", flush=True)
                    return
                raise OSError(f"No weights found in {self.config.output_dir}!")
            _, path = sorted(files)[-1]
            path = os.path.join(self.config.output_dir, path)

        if map_location is None:
            map_location = self.device
        print(f"Loading from {path}...")
        state_dicts = torch.load(path, map_location=map_location)
        self.epochs_trained = state_dicts.pop("epochs_trained", 0)
        for key, state_dict in state_dicts.items():
            if not ignore or key not in ignore:
                if key.endswith("_optimizer"):
                    self.optimizers[key].load_state_dict(state_dict)
                elif key.endswith("_scheduler"):
                    self.schedulers[key].load_state_dict(state_dict)
                else:
                    self.trainable[key].load_state_dict(state_dict)
                print(f"Loaded {key}")

    def run(self,  # pylint: disable=too-many-locals
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
        coref_scores = torch.zeros(top_indices.shape[0], scores.shape[1], device=top_indices.device) * EPSILON
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

    def save_weights(self):
        """ Saves trainable models as state dicts. """
        to_save: List[Tuple[str, Any]] = \
            [(key, value) for key, value in self.trainable.items()
             if self.config.bert_finetune or key != "bert"]
        to_save.extend(self.optimizers.items())
        to_save.extend(self.schedulers.items())

        path = os.path.join(self.config.output_dir,
                            f"{self.config.section}"
                            f"_e{self.epochs_trained}.pt")
        savedict = {name: module.state_dict() for name, module in to_save}
        savedict["epochs_trained"] = self.epochs_trained  # type: ignore
        torch.save(savedict, path)

    def train(self):
        """
        Trains all the trainable blocks in the model using the config provided.
        """
        logger.info("Training/evaluation parameters %s", self.args)
        train_docs = list(self._get_docs(self.config.train_file))
        eval_docs = self._get_docs(self.config.__dict__[f"predict_file"])
        docs_ids = list(range(len(train_docs)))
        avg_spans = sum(len(doc["head2span"]) for doc in train_docs) / len(train_docs)

        best_f1 = -1
        best_f1_epoch = -1
        last_saved_epoch = -1
        global_step = 0
        recent_losses = []
        recent_losses_parts = {'loss_is_cluster':[], 'loss_is_mention':[], 'loss_coref':[]}
        for epoch in range(self.epochs_trained, self.config.num_train_epochs):
            self.training = True
            running_c_loss = 0.0
            running_s_loss = 0.0
            random.shuffle(docs_ids)
            pbar = tqdm(docs_ids, unit="docs", ncols=0)
            for doc_num, doc_id in enumerate(pbar):
                # if doc_num > 0:
                #     continue
                doc = train_docs[doc_id]

                for optim in self.optimizers.values():
                    optim.zero_grad()

                res = self.run(doc, epoch)

                c_loss = self._coref_criterion(res.coref_scores, res.coref_y)
                recent_losses_parts['loss_coref'].append(c_loss.item())
                c_loss += res.cost_is_mention
                recent_losses_parts['loss_is_mention'].append(res.cost_is_mention.item())
                if res.span_y:
                    s_loss = (self._span_criterion(res.span_scores[:, :, 0], res.span_y[0])
                              + self._span_criterion(res.span_scores[:, :, 1], res.span_y[1])) / avg_spans / 2
                else:
                    s_loss = torch.zeros_like(c_loss)
                recent_losses_parts['loss_is_cluster'].append(s_loss.item())

                del res

                (c_loss + s_loss).backward()
                recent_losses.append((c_loss + s_loss).item())

                running_c_loss += c_loss.item()
                running_s_loss += s_loss.item()

                del c_loss, s_loss

                for optim in self.optimizers.values():
                    optim.step()
                for scheduler in self.schedulers.values():
                    scheduler.step()

                pbar.set_description(
                    f"Epoch {epoch + 1}:"
                    f" {doc['document_id']:26}"
                    f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                    f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
                )
                if global_step % 50 == 0:
                    if not self.args.is_debug:
                        dict_to_log = {}
                        dict_to_log['lr'] = self.optimizers["general_optimizer"].param_groups[0]['lr']
                        dict_to_log['lr_bert'] = self.optimizers["bert_optimizer"].param_groups[0]['lr']
                        dict_to_log['loss'] = np.mean(recent_losses)
                        for key in recent_losses_parts.keys():
                            dict_to_log[key] = np.mean(recent_losses_parts[key])
                            recent_losses_parts[key].clear()
                        wandb.log(dict_to_log, step=global_step)
                    recent_losses.clear()
                
                global_step += 1

            if epoch % 3 == 2:
                print("============ TRAIN EXAMPLES ============")
                _, _, train_cluster_evaluator, train_word_cluster_evaluator, train_mention_evaluator, train_men_prop_evaluator, _ = \
                    self.evaluate(docs=train_docs)
                train_p, train_r, train_f1 = train_cluster_evaluator.get_prf()
                train_pw, train_rw, train_f1w = train_word_cluster_evaluator.get_prf()
                train_pm, train_rm, train_f1m = train_mention_evaluator.get_prf()
                train_pmp, train_rmp, train_f1mp = train_men_prop_evaluator.get_prf()
                dict_to_log = {}
                dict_to_log['Train Precision'] = train_p
                dict_to_log['Train Recall'] = train_r
                dict_to_log['Train F1'] = train_f1
                dict_to_log['Train Word Precision'] = train_pw
                dict_to_log['Train Word Recall'] = train_rw
                dict_to_log['Train Word F1'] = train_f1w
                dict_to_log['Train Mention Precision'] = train_pm
                dict_to_log['Train Mention Recall'] = train_rm
                dict_to_log['Train Mention F1'] = train_f1m
                dict_to_log['Train MentionProposal Precision'] = train_pmp
                dict_to_log['Train MentionProposal Recall'] = train_rmp
                dict_to_log['Train MentionProposal F1'] = train_f1mp
                if not self.args.is_debug:
                    wandb.log(dict_to_log, step=global_step)
                logger.info('Train f1, precision, recall: {}, Mentions f1, precision, recall: {}, Mention Proposals f1, precision, recall: {}'.format(\
                    (train_f1, train_p, train_r), (train_f1m, train_pm, train_rm), (train_f1mp, train_pmp, train_rmp)))

            logger.info("***** Running evaluation {} *****".format(str(self.epochs_trained)))
            eval_loss, eval_loss_parts, eval_cluster_evaluator, eval_word_cluster_evaluator, eval_men_evaluator, eval_men_prop_evaluator, \
                (coref_scores, menprops) = \
                self.evaluate(docs=eval_docs)
            eval_p, eval_r, eval_f1 = eval_cluster_evaluator.get_prf()
            eval_pm, eval_rm, eval_f1m = eval_men_evaluator.get_prf()
            eval_pmp, eval_rmp, eval_f1mp = eval_men_prop_evaluator.get_prf()
            eval_pw, eval_rw, eval_f1w = eval_word_cluster_evaluator.get_prf()
            eval_results = {'loss': eval_loss,
                    'avg_f1': eval_f1,
                    'precision': eval_p,
                    'recall': eval_r,  
                    'word_avg_f1': eval_f1w,
                    'word_precision': eval_pw,
                    'word_recall': eval_rw,  
                    'mentions_avg_f1': eval_f1m,
                    'mentions_precision': eval_pm,
                    'mentions_recall': eval_rm,  
                    'mention_proposals_avg_f1': eval_f1mp,
                    'mention_proposals_precision': eval_pmp,
                    'mention_proposals_recall': eval_rmp} | eval_loss_parts
            logger.info("***** Eval results {} *****".format(str(self.epochs_trained)))
            dict_to_log = {}
            for key, value in eval_results.items():
                dict_to_log['eval_{}'.format(key)] = value
                logger.info("eval %s = %s" % (key, str(eval_results[key])))
            if not self.args.is_debug:
                wandb.log(dict_to_log, step=global_step)
            if epoch > 15:
                graph_cluster_eval, graph_mention_eval = self.evaluate_graph(coref_scores, menprops, eval_docs)
                eval_pg, eval_rg, eval_f1g = graph_cluster_eval.get_prf()
                eval_pgm, eval_rgm, eval_f1gm = graph_mention_eval.get_prf()
                eval_results = {
                        'avg_f1': eval_f1g,
                        'precision': eval_pg,
                        'recall': eval_rg,
                        'mentions_avg_f1': eval_f1gm,
                        'mentions_precision': eval_pgm,
                        'mentions_recall': eval_rgm}
                logger.info("***** Eval Graph results {} *****".format(str(self.epochs_trained)))
                dict_to_log = {}
                for key, value in eval_results.items():
                    dict_to_log['eval_{}'.format(key)] = value
                    logger.info("eval %s = %s" % (key, str(eval_results[key])))
                if not self.args.is_debug:
                    wandb.log(dict_to_log, step=global_step+1)
            if eval_f1 > best_f1:
                prev_best_f1 = best_f1
                prev_best_f1_epoch = best_f1_epoch
                self.save_weights()
                print(f'previous checkpoint with f1 {prev_best_f1} was {prev_best_f1_epoch}')
                best_f1 = eval_f1
                best_f1_epoch = epoch
                print(f'saved checkpoint with f1 {best_f1} in step {best_f1_epoch} to {self.config.output_dir}')
                path_to_remove = os.path.join(self.config.output_dir,
                                    f"{self.config.section}"
                                    f"_e{prev_best_f1_epoch}.pt")
                if prev_best_f1_epoch > -1 and os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    print(f'removed checkpoint with f1 {prev_best_f1} from {path_to_remove}')
            else:
                self.save_weights()
                print(f'saved checkpoint in epoch {epoch}')
                path_to_remove = os.path.join(self.config.output_dir,
                                    f"{self.config.section}"
                                    f"_e{last_saved_epoch}.pt")
                if last_saved_epoch > -1 and last_saved_epoch != best_f1_epoch and os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    print(f'removed previous checkpoint in epoch {last_saved_epoch}')
                last_saved_epoch = epoch
            if not self.args.is_debug:
                wandb.log({'eval_best_f1':best_f1}, step=global_step)
                try:
                    wandb.log({'eval_best_f1_checkpoint':\
                        os.path.join(self.config.output_dir,
                                f"{self.config.section}"
                                f"_e{best_f1_epoch}.pt")}, step=global_step)
                except:
                    pass

            self.epochs_trained += 1

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

    def _build_model(self):
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

    def _build_optimizers(self):
        n_docs = len(self._get_docs(self.config.train_file))
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler.LambdaLR] = {}

        for param in self.bert.parameters():
            param.requires_grad = self.config.bert_finetune

        if self.config.bert_finetune:
            self.optimizers["bert_optimizer"] = torch.optim.Adam(
                self.bert.parameters(), lr=self.config.lr_backbone
            )
            self.schedulers["bert_scheduler"] = \
                transformers.get_linear_schedule_with_warmup(
                    self.optimizers["bert_optimizer"],
                    n_docs, n_docs * self.config.num_train_epochs
                )

        # Must ensure the same ordering of parameters between launches
        modules = sorted((key, value) for key, value in self.trainable.items()
                         if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)

        self.optimizers["general_optimizer"] = torch.optim.Adam(
            params, lr=self.config.lr)
        self.schedulers["general_scheduler"] = \
            transformers.get_linear_schedule_with_warmup(
                self.optimizers["general_optimizer"],
                0, n_docs * self.config.num_train_epochs
            )

    def _clusterize(self, doc: Doc, scores: torch.Tensor, top_indices: torch.Tensor):
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

    def _get_docs(self, path: str) -> List[Doc]:
        if path not in self._docs:
            basename = os.path.basename(path)
            model_name = self.config.bert_model.replace("/", "_")
            cache_filename = f"{model_name}_{basename}.pickle"
            if os.path.exists(cache_filename):
                with open(cache_filename, mode="rb") as cache_f:
                    self._docs[path] = pickle.load(cache_f)
            else:
                self._docs[path] = self._tokenize_docs(path)
                with open(cache_filename, mode="wb") as cache_f:
                    pickle.dump(self._docs[path], cache_f)
        return self._docs[path]

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

    @staticmethod
    def _load_config(config_path: str,
                     section: str) -> Config:
        config = toml.load(config_path)
        default_section = config["DEFAULT"]
        current_section = config[section]
        unknown_keys = (set(current_section.keys())
                        - set(default_section.keys()))
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return Config(section, **{**default_section, **current_section})

    def _set_training(self, value: bool):
        self._training = value
        for module in self.trainable.values():
            module.train(self._training)

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
