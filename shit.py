import json
data_dir = '/home/gamir/adiz/datasets/t5math'
import random

def crate_data_pkls(train_len = 1e6, val_len = 1e4, mode = 'diff', \
                    max_digits=15, train_different_number_lens=10, only_val=False):
    val_path = f'{data_dir}/val/val_eqall_{train_len}_{val_len}.txt'
    train_dict = create_data_dict(max_digits, int(val_len), val_path)
    return train_dict

def write_data_file(path, batches):
  with open(path, "w") as writer:
    for i in range(len(batches)):
      writer.write(json.dumps(batches[i]))
      writer.write('\n')

def split_number(num):
  return ' '.join(list(str(num)))

def create_question_template(terms, prefix='sum'):
  result = sum(terms)
  text = f'{prefix}: {split_number(terms[0])}'
  for i in range(len(terms)-1):
    text += f' + {split_number(terms[i+1])}'
  return text, split_number(result)

def create_data_dict(max_len, num_samples, path, max_terms=7):
  all_data = []
  chunk_size = num_samples//max_len
  for i in range(num_samples):
    num_terms = random.randint(2, max_terms)
    terms = []
    number_len = min(1 + i // chunk_size, max_len)
    for j in range(num_terms):
      if number_len == 0:
        terms.append(0)
      else:
        terms.append(random.randint(10**(number_len-1), 10**number_len-1))
    sample_q, sample_t = create_question_template(terms)

    all_data.append({'input':sample_q, 'target':sample_t})
  write_data_file(path, all_data)
  return all_data

crate_data_pkls()