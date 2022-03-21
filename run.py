""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import datetime
import pickle
import random
import re
from contextlib import contextmanager
import random
import sys
import time
import logging
import os
import wandb 
from tqdm import tqdm   # type: ignore
import transformers     # type: ignore
import networkx as nx
import toml

import numpy as np  # type: ignore
import torch        # type: ignore

from coref import CorefModel
from cli import parse_args
from coref.metrics import CorefEvaluator, MentionEvaluator
from coref.coref_analysis import print_predictions
from coref.loss import MatchingLoss
from coref.config import Config

logger = logging.getLogger(__name__)

@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore

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

def load_weights(model, optimizers=None, schedulers=None,
                    path = None,
                    ignore = None,
                    map_location = None,
                    noexception: bool = False) -> None:
    """
    Loads pretrained weights of modules saved in a file located at path.
    If path is None, the last saved model with current configuration
    in output_dir is loaded.
    Assumes files are named like {configuration}_(e{epoch}_{time})*.pt.
    """
    if path is None:
        pattern = rf"{model.config.section}_e(\d+).*\.pt"
        files = []
        for f in os.listdir(model.config.output_dir):
            match_obj = re.match(pattern, f)
            if match_obj:
                files.append((int(match_obj.group(1)), f))
        if not files:
            if noexception:
                print("No weights have been loaded", flush=True)
                return
            raise OSError(f"No weights found in {model.config.output_dir}!")
        _, path = sorted(files)[-1]
        path = os.path.join(model.config.output_dir, path)

    if map_location is None:
        map_location = model.device
    print(f"Loading from {path}...")
    state_dicts = torch.load(path, map_location=map_location)
    model.epochs_trained = state_dicts.pop("epochs_trained", 0)
    for key, state_dict in state_dicts.items():
        if not ignore or key not in ignore:
            if key.endswith("_optimizer"):
                optimizers[key].load_state_dict(state_dict)
            elif key.endswith("_scheduler"):
                schedulers[key].load_state_dict(state_dict)
            else:
                model.trainable[key].load_state_dict(state_dict)
            print(f"Loaded {key}")

def save_weights(model, optimizers, schedulers):
    """ Saves trainable models as state dicts. """
    to_save = \
        [(key, value) for key, value in model.trainable.items()
            if model.config.bert_finetune or key != "bert"]
    to_save.extend(optimizers.items())
    to_save.extend(schedulers.items())

    path = os.path.join(model.config.output_dir,
                        f"{model.config.section}"
                        f"_e{model.epochs_trained}.pt")
    savedict = {name: module.state_dict() for name, module in to_save}
    savedict["epochs_trained"] = model.epochs_trained  # type: ignore
    torch.save(savedict, path)
    return path

def _get_docs(path: str, model):
    basename = os.path.basename(path)
    model_name = model.config.bert_model.replace("/", "_")
    cache_filename = f"{model_name}_{basename}.pickle"
    if os.path.exists(cache_filename):
        with open(cache_filename, mode="rb") as cache_f:
            docs = pickle.load(cache_f)
    else:
        docs = model._tokenize_docs(path)
        with open(cache_filename, mode="wb") as cache_f:
            pickle.dump(docs, cache_f)
    return docs

def train(model, train_docs, eval_docs, coref_criterion, span_criterion, optimizers, schedulers):
    logger.info("Training/evaluation parameters %s", model.args)
    docs_ids = list(range(len(train_docs)))
    avg_spans = sum(len(doc["head2span"]) for doc in train_docs) / len(train_docs)

    best_f1 = -1
    best_f1_epoch = -1
    last_saved_epoch = -1
    global_step = 0
    recent_losses = []
    recent_losses_parts = {}
    for epoch in range(model.epochs_trained, model.config.num_train_epochs):
        model.train()
        running_c_loss = 0.0
        running_s_loss = 0.0
        random.shuffle(docs_ids)
        pbar = tqdm(docs_ids, unit="docs", ncols=0)
        for doc_num, doc_id in enumerate(pbar):
            # if doc_num > 10:
            #     continue
            doc = train_docs[doc_id]

            for optim in optimizers.values():
                optim.zero_grad()

            res = model(doc, epoch)

            c_loss, cost_parts = coref_criterion(res, doc)
            if res.span_y:
                s_loss = (span_criterion(res.span_scores[:, :, 0], res.span_y[0])
                            + span_criterion(res.span_scores[:, :, 1], res.span_y[1])) / avg_spans / 2
            else:
                s_loss = torch.zeros_like(c_loss)
            cost_parts['loss_span'] = s_loss.detach().cpu()

            del res

            (c_loss + s_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            recent_losses.append((c_loss + s_loss).item())
            for key in cost_parts.keys():
                if key in recent_losses_parts.keys() and len(recent_losses_parts[key]) > 0:
                    recent_losses_parts[key].append(cost_parts[key])
                else:
                    recent_losses_parts[key] = [cost_parts[key]]

            running_c_loss += c_loss.item()
            running_s_loss += s_loss.item()

            del c_loss, s_loss

            for optim in optimizers.values():
                optim.step()
            for scheduler in schedulers.values():
                scheduler.step()

            pbar.set_description(
                f"Epoch {epoch + 1}:"
                f" {doc['document_id']:26}"
                f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
            )
            if global_step % 50 == 0:
                if not model.args.is_debug:
                    dict_to_log = {}
                    dict_to_log['lr'] = optimizers["general_optimizer"].param_groups[0]['lr']
                    dict_to_log['lr_bert'] = optimizers["bert_optimizer"].param_groups[0]['lr']
                    dict_to_log['loss'] = np.mean(recent_losses)
                    for key in recent_losses_parts.keys():
                        dict_to_log[key] = np.mean(recent_losses_parts[key])
                        recent_losses_parts[key].clear()
                    wandb.log(dict_to_log, step=global_step)
                recent_losses.clear()
            
            global_step += 1

        if epoch % 3 == 2:
            print("============ TRAIN EXAMPLES ============")
            _, _, train_cluster_evaluator, train_word_cluster_evaluator, train_mention_evaluator, train_men_prop_evaluator = \
                evaluate(model,train_docs,coref_criterion)
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
            if not model.args.is_debug:
                wandb.log(dict_to_log, step=global_step)
            logger.info('Train f1, precision, recall: {}, Mentions f1, precision, recall: {}, Mention Proposals f1, precision, recall: {}'.format(\
                (train_f1, train_p, train_r), (train_f1m, train_pm, train_rm), (train_f1mp, train_pmp, train_rmp)))

        logger.info("***** Running evaluation {} *****".format(str(model.epochs_trained)))
        eval_loss, eval_loss_parts, eval_cluster_evaluator, eval_word_cluster_evaluator, eval_men_evaluator, eval_men_prop_evaluator = \
            evaluate(model,eval_docs,coref_criterion)
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
        logger.info("***** Eval results {} *****".format(str(model.epochs_trained)))
        dict_to_log = {}
        for key, value in eval_results.items():
            dict_to_log['eval_{}'.format(key)] = value
            logger.info("eval %s = %s" % (key, str(eval_results[key])))
        if not model.args.is_debug:
            wandb.log(dict_to_log, step=global_step)
        # if epoch > 15:
        #     graph_cluster_eval, graph_mention_eval = self.evaluate_graph(coref_scores, menprops, eval_docs)
        #     eval_pg, eval_rg, eval_f1g = graph_cluster_eval.get_prf()
        #     eval_pgm, eval_rgm, eval_f1gm = graph_mention_eval.get_prf()
        #     eval_results = {
        #             'avg_f1': eval_f1g,
        #             'precision': eval_pg,
        #             'recall': eval_rg,
        #             'mentions_avg_f1': eval_f1gm,
        #             'mentions_precision': eval_pgm,
        #             'mentions_recall': eval_rgm}
        #     logger.info("***** Eval Graph results {} *****".format(str(self.epochs_trained)))
        #     dict_to_log = {}
        #     for key, value in eval_results.items():
        #         dict_to_log['eval_{}'.format(key)] = value
        #         logger.info("eval %s = %s" % (key, str(eval_results[key])))
        #     if not self.args.is_debug:
        #         wandb.log(dict_to_log, step=global_step+1)
        if eval_f1 > best_f1:
            prev_best_f1 = best_f1
            prev_best_f1_epoch = best_f1_epoch
            saved_path = save_weights(model, optimizers, schedulers)
            print(f'previous checkpoint with f1 {prev_best_f1} was {prev_best_f1_epoch}')
            best_f1 = eval_f1
            best_f1_epoch = epoch
            print(f'saved checkpoint with f1 {best_f1} in step {best_f1_epoch} to {saved_path}')
            path_to_remove = os.path.join(model.config.output_dir,
                                f"{model.config.section}"
                                f"_e{prev_best_f1_epoch}.pt")
            if prev_best_f1_epoch > -1 and os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                print(f'removed checkpoint with f1 {prev_best_f1} from {path_to_remove}')
        else:
            saved_path = save_weights(model, optimizers, schedulers)
            print(f'saved checkpoint in epoch {epoch} to {saved_path}')
            path_to_remove = os.path.join(model.config.output_dir,
                                f"{model.config.section}"
                                f"_e{last_saved_epoch}.pt")
            if last_saved_epoch > -1 and last_saved_epoch != best_f1_epoch and os.path.exists(path_to_remove):
                os.remove(path_to_remove)
                print(f'removed previous checkpoint in epoch {last_saved_epoch}')
            last_saved_epoch = epoch
        if not model.args.is_debug:
            wandb.log({'eval_best_f1':best_f1}, step=global_step)
            try:
                wandb.log({'eval_best_f1_checkpoint':\
                    os.path.join(model.config.output_dir,
                            f"{model.config.section}"
                            f"_e{best_f1_epoch}.pt")}, step=global_step)
            except:
                pass

        model.epochs_trained += 1



def evaluate(model,docs,coref_criterion,  
                data_split: str = "dev",
                word_level_conll: bool = False,
                ):
    """ Evaluates the modes on the data split provided.

    Args:
        data_split (str): one of 'dev'/'test'/'train'
        word_level_conll (bool): if True, outputs conll files on word-level

    Returns:
        mean loss
        span-level LEA: f1, precision, recal
    """
    model.eval()
    cluster_evaluator = CorefEvaluator()       
    word_cluster_evaluator = CorefEvaluator()       
    # cluster_graph_evaluator = CorefEvaluator()
    mention_evaluator = MentionEvaluator()
    men_prop_evaluator = MentionEvaluator()
    # w_checker = ClusterChecker()
    # s_checker = ClusterChecker()
    running_loss = 0.0
    losses_parts = {}
    s_correct = 0
    s_total = 0

    all_gold_clusters = []
    all_predicted_clusters = []
    all_tokens = []

    # with conll.open_(self.config, self.epochs_trained, data_split) \
    #         as (gold_f, pred_f):
    pbar = tqdm(docs, unit="docs", ncols=0)
    for doc in pbar:
        with torch.no_grad():
            res = model(doc)

            c_loss, cost_parts = coref_criterion(res, doc)
            running_loss += c_loss
            for key in cost_parts.keys():
                if key in losses_parts.keys():
                    losses_parts[key] += cost_parts[key]
                else:
                    losses_parts[key] = cost_parts[key]

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

        del res

    print()

    print(f"============ {data_split} EXAMPLES ============")
    print_predictions(all_gold_clusters, all_predicted_clusters, all_tokens, model.args.max_eval_print)
    # train_p, train_r, train_f1 = cluster_graph_evaluator.get_prf()
    # logger.info('Cluster f1, precision, recall: {}'.format(\
    #     (train_f1, train_p, train_r)))

    for key in cost_parts.keys():
        losses_parts[key] /= len(docs)

    return running_loss / len(docs), losses_parts, cluster_evaluator, word_cluster_evaluator, mention_evaluator, men_prop_evaluator



if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    #vladmir limits something in sentence to one? and Im not?
    #parameters to args
    #eval train after loop
    #maybe add mention score rather than mult
    #adamw
    ####speaker+genre in text
    #different loss options (max,bce,div)
    ####self attn?
    args = parse_args()
    if args.warm_start and args.weights is not None:
        print("The following options are incompatible:"
              " '--warm_start' and '--weights'", file=sys.stderr)
        sys.exit(1)

    if "JOB_NAME" in os.environ:
        args.run_name = os.environ["JOB_NAME"]
    else:
        args.run_name = 'vscode'

    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)
    if not args.is_debug:
        wandb.init(project='coref-detr', entity='adizicher', name=args.run_name)

    if args.is_debug:
        vis_devices="0"
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = len(vis_devices.split(','))
            os.environ["CUDA_VISIBLE_DEVICES"] = vis_devices
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1 if not args.no_cuda else 0
    args.device = device
    if args.is_debug:
        if args.no_cuda:
            args.n_gpu = 0
        else:
            args.n_gpu = len(vis_devices.split(','))
            os.environ["CUDA_VISIBLE_DEVICES"] = vis_devices
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    if not args.is_debug:    
        wb_config = wandb.config
        for key, val in vars(args).items():
            logger.info(f"{key} - {val}")
            wb_config[key] = val
        if "GIT_HASH" in os.environ:
            wb_config["GIT_HASH"] = os.environ["GIT_HASH"]
    else:
        for key, val in vars(args).items():
            logger.info(f"{key} - {val}")

    seed(args.seed)
    config = _load_config(args.config_file, args.model_type)
    config.output_dir = os.path.join(config.output_dir, \
        datetime.datetime.now().strftime(f"%m_%d_%Y_%H_%M_%S")+'_'+args.run_name)
    model = CorefModel(args, config)
    coref_criterion = MatchingLoss(args.num_queries)
    span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    
    train_docs = list(_get_docs(config.train_file, model))
    eval_docs = _get_docs(config.__dict__[f"predict_file"], model)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        n_docs = len(train_docs)
        optimizers = {}
        schedulers = {}

        for param in model.bert.parameters():
            param.requires_grad = model.config.bert_finetune

        if model.config.bert_finetune:
            optimizers["bert_optimizer"] = torch.optim.Adam(
                model.bert.parameters(), lr=model.config.lr_backbone
            )
            schedulers["bert_scheduler"] = \
                transformers.get_linear_schedule_with_warmup(
                    optimizers["bert_optimizer"],
                    n_docs, n_docs * model.config.num_train_epochs
                )

        # Must ensure the same ordering of parameters between launches
        modules = sorted((key, value) for key, value in model.trainable.items()
                            if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)

        optimizers["general_optimizer"] = torch.optim.Adam(
            params, lr=model.args.lr)
        schedulers["general_scheduler"] = \
            transformers.get_linear_schedule_with_warmup(
                optimizers["general_optimizer"],
                0, n_docs * model.config.num_train_epochs
            )

        if args.weights is not None or args.warm_start:
            load_weights(model, optimizers=optimizers, schedulers=schedulers, path=args.weights, map_location="cpu",
                               noexception=args.warm_start)
        with output_running_time():
            train(model, train_docs, eval_docs, coref_criterion, span_criterion, optimizers, schedulers)
    else:
        load_weights(model, path=args.weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})
        eval_loss, losses_parts, eval_cluster_evaluator, eval_men_evaluator, eval_men_prop_evaluator = \
            evaluate(model, eval_docs, coref_criterion, data_split=args.data_split, word_level_conll=args.word_level)
        eval_p, eval_r, eval_f1 = eval_cluster_evaluator.get_prf()
        eval_pm, eval_rm, eval_f1m = eval_men_evaluator.get_prf()
        eval_pmp, eval_rmp, eval_f1mp = eval_men_prop_evaluator.get_prf()
        eval_results = {'loss': eval_loss,
                'avg_f1': eval_f1,
                'precision': eval_p,
                'recall': eval_r,  
                'mentions_avg_f1': eval_f1m,
                'mentions_precision': eval_pm,
                'mentions_recall': eval_rm,  
                'mention_proposals_avg_f1': eval_f1mp,
                'mention_proposals_precision': eval_pmp,
                'mention_proposals_recall': eval_rmp} | losses_parts
        print("***** Eval results *****")
        dict_to_log = {}
        for key, value in eval_results.items():
            dict_to_log['eval_{}'.format(key)] = value
            print("eval %s = %s" % (key, str(eval_results[key])))






# def evaluate_graph(self, 
#                 coref_scores,
#                 menprops,
#                 docs = None
#                 ) -> Tuple[float, Tuple[float, float, float]]:
#     """ Evaluates the modes on the data split provided.

#     Args:
#         data_split (str): one of 'dev'/'test'/'train'
#         word_level_conll (bool): if True, outputs conll files on word-level

#     Returns:
#         mean loss
#         span-level LEA: f1, precision, recal
#     """
#     model.eval()
#     cluster_graph_evaluator = CorefEvaluator()
#     word_graph_evaluator = MentionEvaluator()

#     pbar = tqdm(docs, unit="docs", ncols=0)
#     for ind, doc in enumerate(pbar):
#         weights = torch.zeros(len(doc['cased_words'])+1, len(doc['cased_words'])+1)
#         rows = torch.arange(1,len(doc['cased_words'])+1).unsqueeze(1).repeat(1,coref_scores[ind].shape[1])
#         top_indices_shifted = torch.cat([\
#             torch.zeros(1, dtype=torch.long),\
#                 menprops[ind]+1],-1).unsqueeze(0).repeat(coref_scores[ind].shape[0], 1)
#         weights[rows, top_indices_shifted] = coref_scores[ind].detach().cpu()
#         weights = weights.transpose(0,1)
#         weights[weights<0] = 0
#         real_indices = torch.cat([torch.zeros(1, dtype=torch.long), menprops[ind].detach().cpu()+1])
#         real_indices_rows = real_indices.unsqueeze(1).repeat(1,real_indices.shape[0])
#         real_indices_cols = real_indices.unsqueeze(0).repeat(real_indices.shape[0], 1)
#         weights = weights[real_indices_rows, real_indices_cols]
#         G = nx.from_numpy_matrix(weights.numpy())
#         all_part2 = []
#         part1 = [1,len(menprops[ind])]
#         x=0
#         while len(part1) > 1:
#             split_score, (part1, part2) = nx.minimum_cut(G, 0, part1[-1], capacity='weight')
#             part1 = list(part1)
#             if len(part2)>1:
#                 all_part2.append(sorted([menprops[ind][p2-1].item() for p2 in part2]))
#                 # print(f"split no. {x} with score {split_score}, part1: {part1}, part2: {part2}")
#             G = G.subgraph(part1)
#             x+=1
#         graph_clusters = all_part2

#         cluster_graph_evaluator.update([[[(graph_clusters[i][j],graph_clusters[i][j]) \
#         for j in range(len(graph_clusters[i]))] for i in range(len(graph_clusters))]], \
#                 [[[(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for j in range(len(doc['word_clusters'][i]))]\
#                     for i in range(len(doc['word_clusters']))]])
#         word_graph_evaluator.update([(graph_clusters[i][j],graph_clusters[i][j]) \
#             for i in range(len(graph_clusters)) for j in range(len(graph_clusters[i]))], \
#                 [(doc['word_clusters'][i][j],doc['word_clusters'][i][j]) for i in range(len(doc['word_clusters']))\
#                     for j in range(len(doc['word_clusters'][i]))])

#     print()

#     return cluster_graph_evaluator, word_graph_evaluator
