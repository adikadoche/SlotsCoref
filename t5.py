from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import math
import torch
import random
import json
from transformers import T5Config, T5Tokenizer,T5ForConditionalGeneration
from torch.utils.data import Dataset
import datetime
import argparse
import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
import jellyfish
# import nlp
from transformers import T5ForConditionalGeneration, T5Tokenizer

from tqdm.auto import tqdm
# import dataclasses
import os
# import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# import numpy as np

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction
from transformers import (
    # HfArgumentParser,
    DataCollator,
    Trainer,
    TrainingArguments,
    set_seed,
)

data_dir = '/home/gamir/adiz/datasets/t5math'

tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir='/home/gamir/adiz/Code/runs/wl-coref/cache_dir')




def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    # prediction_tokens = normalize_answer(prediction).split()
    # ground_truth_tokens = normalize_answer(ground_truth).split()
    # common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # num_same = sum(common.values())
    # if num_same == 0:
    #     return 0
    # precision = 1.0 * num_same / len(prediction_tokens)
    # recall = 1.0 * num_same / len(ground_truth_tokens)
    # f1 = (2 * precision * recall) / (precision + recall)

    # print(ground_truth)
    # print(prediction)
    # print(f1)
    f1 = jellyfish.levenshtein_distance(prediction, ground_truth) / 2
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def diff_score(prediction, ground_truth):
    return abs(int(''.join(prediction.split(' '))) - int(''.join(ground_truth.split(' '))))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
    return metric_fn(prediction, ground_truth)


# def evaluate(gold_answers, predictions):
#     f1 = exact_match = total = 0

#     for ground_truths, prediction in zip(gold_answers, predictions):
#       total += 1
#       exact_match += metric_max_over_ground_truths(
#                     exact_match_score, prediction, ground_truths)
#       f1 += metric_max_over_ground_truths(
#           f1_score, prediction, ground_truths)
    
#     exact_match = 100.0 * exact_match / total
#     f1 = f1 / total

#     return {'exact_match': exact_match, 'f1': f1}

def main_eval(valid_dataset, model, batch_size):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  for s in valid_dataset:
    for key in s.keys():
      s[key] = s[key].to(device)
  dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

  i=0
  f1 = exact_match = diff = total = 0
  pbar = tqdm(dataloader, unit="batch", ncols=0)
  for batch in pbar:
    outs = model.generate(input_ids=batch['input_ids'], 
                          attention_mask=batch['attention_mask'],
                          max_length=64,
                          early_stopping=True)
    for j in range(len(outs)):
      total += 1
      prediction = tokenizer.decode(outs[j], skip_special_tokens=True)
      ground_truths = tokenizer.decode(valid_dataset[i+j]['target_ids'], skip_special_tokens=True)
      exact_match += metric_max_over_ground_truths(
              exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
      diff += metric_max_over_ground_truths(
          diff_score, prediction, ground_truths)
    pbar.set_description(
      f" f1: {f1 / total:<.5f}"
      f" diff: {diff / total:<.5f}"
      f" exact match: {100.0 * exact_match / total:<.5f}"
    )
    i += len(outs)
    # break
  exact_match = 100.0 * exact_match / total
  f1 = f1 / total
  diff = diff / total

  print({'f1': f1, 'diff': diff, 'exact_match': exact_match})

def main_eq_eval(valid_dataset, model, batch_size):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  for s in valid_dataset:
    for key in s.keys():
      s[key] = s[key].to(device)
  dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

  f1 = exact_match = diff = total = 0
  f1_d = {}
  em_d = {}
  diff_d = {}
  tot_d = {}
  i = 0
  pbar = tqdm(dataloader, unit="batch", ncols=0)
  for batch in pbar:
    outs = model.generate(input_ids=batch['input_ids'], 
                          attention_mask=batch['attention_mask'],
                          max_length=64,
                          early_stopping=True)
    outs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    only_digits = [batch['input_ids'][j][(batch['input_ids'][j] == 1768) | (batch['input_ids'][j] == 1) | (batch['input_ids'][j] == 632) | \
      (batch['input_ids'][j] == 209) | (batch['input_ids'][j] == 204) | (batch['input_ids'][j] == 220) | (batch['input_ids'][j] == 314) | \
        (batch['input_ids'][j] == 305) | (batch['input_ids'][j] == 431) | (batch['input_ids'][j] == 489) | (batch['input_ids'][j] == 505) | \
          (batch['input_ids'][j] == 668)] \
      for j in range(batch['input_ids'].shape[0])]
    inds = [torch.arange(only_digits[j].shape[0])[(only_digits[j] == 1768) | (only_digits[j] == 1)] for j in range(len(only_digits))]
    lens = [(inds[j][-1]-inds[j][-2]-1).item() for j in range(len(inds))]

    for j in range(len(outs)):
      print('\n'.join(tokenizer.decode(valid_dataset[i+j]['input_ids'], skip_special_tokens=True).replace('+','+    ').split('+')))
      total += 1
      ground_truths = tokenizer.decode(valid_dataset[i+j]['target_ids'], skip_special_tokens=True)
      print(f'GT:  {ground_truths}')
      print(f'Pr.: {outs[j]}')
      print("---------")
      em = metric_max_over_ground_truths(
              exact_match_score, outs[j], ground_truths)
      f = metric_max_over_ground_truths(
          f1_score, outs[j], ground_truths)
      d = metric_max_over_ground_truths(
          diff_score, outs[j], ground_truths)
      exact_match += em
      f1 += f
      diff += d
      if lens[j] in f1_d.keys():
        f1_d[lens[j]] += f
        em_d[lens[j]] += em
        tot_d[lens[j]] += 1
        diff_d[lens[j]] += d
      else:
        f1_d[lens[j]] = f
        em_d[lens[j]] = em
        tot_d[lens[j]] = 1
        diff_d[lens[j]] = d
      
    desc =  f" f1: {f1 / total:<.5f}, exact match: {100.0 * exact_match / total:<.5f}, diff: {diff / total:<.5f}, total: {total} ||"
    for l in sorted(em_d.keys()):
      desc += f" {l}, {tot_d[l]}: f1: {f1_d[l] / tot_d[l]:<.5f}, exact match: {100.0 * em_d[l] / tot_d[l]:<.5f}, diff: {diff_d[l] / tot_d[l]:<.5f} ||"

    pbar.set_description(desc)

    i += batch['input_ids'].shape[0]

  desc =  f" f1: {f1 / total:<.5f}, exact match: {100.0 * exact_match / total:<.5f}, diff: {diff / total:<.5f}, total: {total} ||"
  for l in sorted(em_d.keys()):
    desc += f" {l}, {tot_d[l]}: f1: {f1_d[l] / tot_d[l]:<.5f}, exact match: {100.0 * em_d[l] / tot_d[l]:<.5f}, diff: {diff_d[l] / tot_d[l]:<.5f} ||"

  print(desc)

def split_number(num):
  return ' '.join(list(str(num)))

def create_question_template(terms, is_rev=False, prefix='sum'):
  rev_num = -1 if is_rev else 1
  result = sum(terms)
  text = f'{prefix}: {split_number(terms[0])[::rev_num]}'
  for i in range(len(terms)-1):
    text += f' + {split_number(terms[i+1])[::rev_num]}'
  return text, split_number(result)[::rev_num]

def create_binary_question_template(term, result, is_rev=False, prefix='count'):
  rev_num = -1 if is_rev else 1
  term = ''.join(term)
  text = f'{prefix}: {split_number(term)[::rev_num]}'
  return text, split_number(result)[::rev_num]

def create_data_dict(number_lens, num_samples, path, max_terms=7, is_eq=False, is_rev=False):
  all_data = []
  for i in range(num_samples):
    num_terms = random.randint(2, max_terms)
    terms = []
    if is_eq:
      number_len = random.choice(number_lens)
    for j in range(num_terms):
      if not is_eq:
        number_len = random.choice(number_lens)
      if number_len == 0:
        terms.append(0)
      else:
        terms.append(random.randint(10**(number_len-1), 10**number_len-1))
    sample_q, sample_t = create_question_template(terms, is_rev)

    all_data.append({'input':sample_q, 'target':sample_t})
  write_data_file(path, all_data)
  return all_data

def create_binary_data_dict(number_lens, num_samples, path, max_terms=10, is_eq=True, is_rev=False):
  #max terms is the sequence length here
  all_data = []
  for i in range(num_samples):
    num_ones = random.choice(number_lens)
    term = (['1'] * num_ones) + (['0'] * (max_terms-num_ones))
    random.shuffle(term)
    sample_q, sample_t = create_binary_question_template(term, str(num_ones), is_rev)

    all_data.append({'input':sample_q, 'target':sample_t})
  write_data_file(path, all_data)
  return all_data

def write_data_file(path, batches):
  with open(path, "w") as writer:
    for i in range(len(batches)):
      writer.write(json.dumps(batches[i]))
      writer.write('\n')

def crate_data_pkls(train_len = 1e6, val_len = 1e4, max_terms=7, is_eq=False, is_rev=False, # mode = 'diff', 
                    max_digits=15, min_digits=1, train_different_number_lens=10, mode="train"):
  train_path = f'{data_dir}/train/train_{train_len}_{val_len}_{min_digits}_{max_digits}_{max_terms}_{is_eq}_{is_rev}.txt'
  # val_path = f'{data_dir}/train/train_{train_len}_{val_len}.txt'
  val_s_path = f'{data_dir}/val/val_same_{train_len}_{val_len}_{min_digits}_{max_digits}_{max_terms}_{is_eq}_{is_rev}.txt'
  val_d_path = f'{data_dir}/val/val_diff_{train_len}_{val_len}_{min_digits}_{max_digits}_{max_terms}_{is_eq}_{is_rev}.txt'
  # val_path = f'{data_dir}/val/val_eqall_{train_len}_{val_len}.txt'
  t_train = t_s_val = t_d_val = None
  if (mode == '' or mode == 'train') and os.path.exists(train_path):
    t_train = tensor_data(read_data_file(train_path))
  if (mode == '' or mode == 'same') and os.path.exists(val_s_path):
    t_s_val = tensor_data(read_data_file(val_s_path))
  if (mode == '' or mode == 'diff') and os.path.exists(val_d_path):
    t_d_val = tensor_data(read_data_file(val_d_path))
  if not (os.path.exists(train_path) and os.path.exists(val_s_path) and os.path.exists(val_d_path)):
    train_number_lens = random.sample(list(range(min_digits,max_digits+1)), k=train_different_number_lens)
    different_val_digit_lens = [i for i in range(max_digits+1) if i not in train_number_lens]
    # val_number_lens = train_number_lens if mode=='same' else different_val_digit_lens
    os.makedirs(f'{data_dir}/train', exist_ok=True)
    os.makedirs(f'{data_dir}/val', exist_ok=True)
    if mode == '' or mode == 'train':
      t_train = tensor_data(create_data_dict(train_number_lens, int(train_len), train_path, max_terms, is_eq, is_rev))
    if mode == '' or mode == 'same':
      t_s_val = tensor_data(create_data_dict(train_number_lens, int(val_len), val_s_path, max_terms, is_eq, is_rev))
    if mode == '' or mode == 'diff':
      t_d_val = tensor_data(create_data_dict(different_val_digit_lens, int(val_len), val_d_path, max_terms, is_eq, is_rev))
  return t_train, t_s_val, t_d_val


def read_data_file(path):
  with open(path, "r") as reader:
    lines = reader.readlines()
    batches = [json.loads(jsonline) for jsonline in lines]
  return batches

class mathDataset(Dataset):
  def __init__(self, filepath) -> None:
    super().__init__()
    self.data = read_data_file(filepath)
    for i, sample in enumerate(self.data):
      tok_input = tokenizer(sample['input'], return_tensors='pt', \
                            pad_to_max_length=True, max_length=512)
      tok_target = tokenizer(sample['target'], return_tensors='pt', \
                             pad_to_max_length=True, max_length=64)
      self.data[i] = {'input_ids':tok_input['input_ids'].squeeze(0),\
                 'target_ids':tok_target['input_ids'].squeeze(0),\
                 'attention_mask':tok_input['attention_mask'].squeeze(0),\
                 'target_attention_mask':tok_target['attention_mask'].squeeze(0)}
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    return self.data[index]

def tensor_data(data):
    for i, sample in enumerate(data):
      tok_input = tokenizer(sample['input'], return_tensors='pt', \
                            pad_to_max_length=True, max_length=512, truncation=True)
      tok_target = tokenizer(sample['target'], return_tensors='pt', \
                             pad_to_max_length=True, max_length=64, truncation=True)
      data[i] = {'input_ids':tok_input['input_ids'].squeeze(0),\
                 'target_ids':tok_target['input_ids'].squeeze(0),\
                 'attention_mask':tok_input['attention_mask'].squeeze(0),\
                 'target_attention_mask':tok_target['attention_mask'].squeeze(0)}
    return data

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
def default_data_collator(batch: List[Any], return_tensors="pt") -> Dict[str, Any]:
  """
  Take a list of samples from a Dataset and collate them into a batch.
  Returns:
      A dictionary of tensors
  """
  input_ids = torch.stack([example['input_ids'] for example in batch],0)
  # print(input_ids[input_ids<0].numel())
  lm_labels = torch.stack([example['target_ids'] for example in batch],0)
  # print(lm_labels[lm_labels<0].numel())
  lm_labels[lm_labels[:, :] == 0] = -100
  # print(lm_labels)
  attention_mask = torch.stack([example['attention_mask'] for example in batch],0)
  # decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch],0)
  
  return {
      'input_ids': input_ids, 
      'attention_mask': attention_mask,
      'labels': lm_labels, 
      # 'decoder_attention_mask': decoder_attention_mask
  }

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )

class SparseTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # labels = inputs["labels"]
        outputs = model(input_ids=inputs['input_ids'], attention_mask = inputs['attention_mask'], \
          labels=inputs["labels"], output_attentions=True, return_dict=True)
        sum_attn = sum(outputs['encoder_attentions'])
        sum_attn = torch.sum(sum_attn, [0,1])
        num_input_tok = torch.sum(inputs['attention_mask'])
        sum_attn = sum_attn[:num_input_tok, :num_input_tok]
        return outputs['loss'] + \
          0.2 * torch.mean(sum_attn * (1-torch.eye(sum_attn.shape[0], device=sum_attn.device)))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("tv_mode", choices=("train", "val"))
    argparser.add_argument("--mode", choices=("diff", "same", "train"))
    argparser.add_argument("--train_size", type=int, default=100000)
    argparser.add_argument("--val_size", type=int, default=1000)
    argparser.add_argument("--max_terms", type=int, default=7)
    argparser.add_argument("--max_digits", type=int, default=15)
    argparser.add_argument("--min_digits", type=int, default=1)
    argparser.add_argument("--ev_batch_size", type=int, default=256)
    argparser.add_argument("--tr_batch_size", type=int, default=64)
    argparser.add_argument("--epochs", type=int, default=10)
    argparser.add_argument("--is_eq",  action="store_true")
    argparser.add_argument("--is_rev",  action="store_true")
    argparser.add_argument("--scratch",  action="store_true")
    argparser.add_argument("--train_different_number_lens", type=int, default=10)
    argparser.add_argument("--weights")
    argparser.add_argument("--is_debug", action="store_true", help="Whether to run profiling.")
    args = argparser.parse_args()
    if "JOB_NAME" in os.environ:
        args.run_name = os.environ["JOB_NAME"]
    else:
        args.run_name = 'vscode'

    print(args)

    if args.is_debug:
      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Get datasets
    
    if args.tv_mode == 'val':
      print('loading data')
      d1,d2,d3 = crate_data_pkls(args.train_size,args.val_size, mode=args.mode, max_terms=args.max_terms, is_rev=args.is_rev, \
        max_digits=args.max_digits, min_digits=args.min_digits, train_different_number_lens=args.train_different_number_lens, is_eq=args.is_eq)
      print('loading done')
      if args.mode == 'train':
        dataset = d1
      elif args.mode == 'same':
        dataset = d2
      else:
        dataset = d3
      print(f"******Eval {args.mode}*******")
      model = T5ForConditionalGeneration.from_pretrained(args.weights, cache_dir='/home/gamir/adiz/Code/runs/wl-coref/cache_dir')
      if args.is_eq:
        main_eq_eval(dataset, model, args.ev_batch_size)
      else:
        main_eval(dataset, model, args.ev_batch_size)
    else:
      print('loading data')
      train_dataset, valid_same_dataset, valid_diff_dataset = crate_data_pkls(args.train_size,args.val_size, is_eq=args.is_eq, is_rev=args.is_rev, \
        max_terms=args.max_terms, max_digits=args.max_digits, min_digits=args.min_digits, train_different_number_lens=args.train_different_number_lens, mode="")
      print('loading done')
      if args.scratch:
        config = T5Config.from_pretrained('t5-base')
        model = T5ForConditionalGeneration(config)
      else:
        model = T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir='/home/gamir/adiz/Code/runs/wl-coref/cache_dir')
      # model_args = ModelArguments('t5-base', 't5-base')
      output_dir = f'/home/gamir/adiz/Code/runs/wl-coref/weights/{args.run_name}_' + \
          datetime.datetime.now().strftime(f"%m_%d_%Y_%H_%M_%S")+'_'+\
            f'{args.train_size}_{args.val_size}_{args.max_digits}_{args.max_terms}_{args.is_eq}_{args.is_rev}'
      print(output_dir)
      training_args = TrainingArguments(output_dir=output_dir, do_train=True, \
                                        do_eval=True, num_train_epochs=args.epochs, \
                                      #   tpu_num_cores=4, \
                                        learning_rate=1e-4, \
                                        gradient_accumulation_steps=1, \
                                        eval_accumulation_steps=1, per_device_train_batch_size=args.tr_batch_size,\
                                        per_device_eval_batch_size=1,
                                        save_strategy="epoch", save_total_limit=3)
      # if (
      #     os.path.exists(training_args.output_dir)
      #     and os.listdir(training_args.output_dir)
      #     and training_args.do_train
      #     and not training_args.overwrite_output_dir
      # ):
      #     raise ValueError(
      #         f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
      #     )

      print("Training/evaluation parameters %s", training_args)

      # Set seed
      set_seed(training_args.seed)

      # Load pretrained model and tokenizer
      #
      # Distributed training:
      # The .from_pretrained methods guarantee that only one local process can concurrently
      # download model & vocab.

      # tokenizer = T5Tokenizer.from_pretrained(
      #     model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
      #     cache_dir=model_args.cache_dir,
      # )
      # model = T5ForConditionalGeneration.from_pretrained(
      #     model_args.model_name_or_path,
      #     cache_dir=model_args.cache_dir,
      # )


      # Initialize our Trainer
      trainer = SparseTrainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          # eval_dataset=valid_dataset,
          data_collator=default_data_collator,
      )

      # Training
      if training_args.do_train:
          trainer.train(
              # model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
          )
          trainer.save_model()
          # For convenience, we also re-save the tokenizer to the same directory,
          # so that you can share your model easily on huggingface.co/models =)
          if trainer.is_world_process_zero():
              tokenizer.save_pretrained(training_args.output_dir)

      # Evaluation
      # results = {}
      # if training_args.do_eval and training_args.local_rank in [-1, 0]:
      #     print("*** Evaluate ***")

      #     eval_output = trainer.evaluate()

      #     output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
      #     with open(output_eval_file, "w") as writer:
      #         print("***** Eval results *****")
      #         for key in sorted(eval_output.keys()):
      #             print("  %s = %s", key, str(eval_output[key]))
      #             writer.write("%s = %s\n" % (key, str(eval_output[key])))
      
      #     results.update(eval_output)
      
      print("******Eval Same*******")
      if args.is_eq:
        main_eq_eval(valid_same_dataset, model, args.ev_batch_size)
      else:
        main_eval(valid_same_dataset, model, args.ev_batch_size)
      print("******Eval Diff*******")
      if args.is_eq:
        main_eq_eval(valid_diff_dataset, model, args.ev_batch_size)
      else:
        main_eval(valid_diff_dataset, model, args.ev_batch_size)
      print("******Train*******")
      if args.is_eq:
        main_eq_eval(train_dataset, model, args.ev_batch_size)
      else:
        main_eval(train_dataset, model, args.ev_batch_size)
    # return results

# main('diff', '1000000', '10000')

# def _mp_fn(index):
#     # For xla_spawn (TPUs)
#     main()



## SQuAD evaluation script. Modifed slightly for this notebook