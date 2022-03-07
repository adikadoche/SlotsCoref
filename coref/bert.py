"""Functions related to BERT or similar models"""

from typing import List, Tuple

import numpy as np                                 # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore
import copy

from coref.config import Config
from coref.const import Doc


def get_subwords_batches(doc: Doc,
                         config: Config,
                         tok: AutoTokenizer
                         ) -> np.ndarray:
    """
    Turns a list of subwords to a list of lists of subword indices
    of max length == batch_size (or shorter, as batch boundaries
    should match sentence boundaries). Each batch is enclosed in cls and sep
    special tokens.

    Returns:
        batches of bert tokens [n_batches, batch_size]
    """
    subwords: List[str] = copy.deepcopy(doc["subwords"])
    subwords, speakerdoc_mask, new_word_id = add_speaker(subwords, doc, tok)
    doc_type = tok.tokenize(" <doc> " + doc["document_id"][:2] + " </doc>", add_special_tokens=False)
    subwords, speakerdoc_mask, new_word_id = insert_substring(subwords, doc_type, 0, speakerdoc_mask, new_word_id)
    subwords_batches = []
    start, end = 0, 0
    batch_size = config.bert_window_size - 2 # 2 to save space for CLS and SEP

    while end < len(subwords):
        end = min(end + batch_size, len(subwords))

        # Move back till we hit a sentence end
        if end < len(subwords):
            sent_id = doc['sent_id'][new_word_id[end]]
            while end and doc['sent_id'][new_word_id[end - 1]] == sent_id:
                end -= 1

        length = end - start
        batch = [tok.cls_token] + subwords[start:end] + [tok.sep_token]
        batch_ids = [-1] + list(range(start, end)) + [-1]

        # Padding to desired length
        # -1 means the token is a special token
        batch += [tok.pad_token] * (batch_size - length)
        batch_ids += [-1] * (batch_size - length)

        subwords_batches.append([tok.convert_tokens_to_ids(token)
                                 for token in batch])
        start += length
        
        if end < len(subwords):
            if speakerdoc_mask[start] == 0:
                subwords, speakerdoc_mask, new_word_id = \
                    insert_substring(subwords, get_tokenized_speaker(doc['speaker'][new_word_id[start]], tok), start, speakerdoc_mask, new_word_id)
            subwords, speakerdoc_mask, new_word_id = \
                insert_substring(subwords, doc_type, start, speakerdoc_mask, new_word_id)

    return np.array(subwords_batches), speakerdoc_mask

def insert_substring(subwords, substring, index, speakerdoc_mask, word_id):
    subwords[index:index] = substring
    speakerdoc_mask[index:index] = [1]*len(substring)
    word_id[index:index] = [word_id[index]] * len(substring)
    return subwords, speakerdoc_mask, word_id

def get_tokenized_speaker(speaker, tok):
    SPEAKER_START = ' <speaker>' 
    SPEAKER_END = ' </speaker>'  
    return tok.tokenize(SPEAKER_START + " " + speaker + SPEAKER_END, add_special_tokens=False)

def add_speaker(subwords, doc, tok: AutoTokenizer):
    longest_speaker_len = 0
    speaker_mask = []
    speakers = doc['speaker']
    new_word_id = copy.deepcopy(doc['word_id'])
    subword_index = len(subwords) - 1
    for i in reversed(range(len(speakers))):
        while doc['word_id'][subword_index] == i:
            speaker_mask.append(0)
            subword_index -= 1
        if i==0 or speakers[i] != speakers[i-1]:
            speaker_tokens = get_tokenized_speaker(speakers[i], tok)
            if len(speaker_tokens) > longest_speaker_len:
                longest_speaker_len = len(speaker_tokens)
            j = 0 if i == 0 else subword_index+1
            subwords[j:j] = speaker_tokens
            new_word_id[j:j] = [i] * len(speaker_tokens)
            speaker_mask += [1]*len(speaker_tokens)
    speaker_mask = list(reversed(speaker_mask))
    return subwords, speaker_mask, new_word_id


def load_bert(config: Config, device) -> Tuple[AutoModel, AutoTokenizer]:
    """
    Loads bert and bert tokenizer as pytorch modules.

    Bert model is loaded to the device specified in config.device
    """
    print(f"Loading {config.bert_model}...")

    base_bert_name = config.bert_model.split("/")[-1]
    tokenizer_kwargs = config.tokenizer_kwargs.get(base_bert_name, {})
    if tokenizer_kwargs:
        print(f"Using tokenizer kwargs: {tokenizer_kwargs}")
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model,
                                              cache_dir=config.cache_dir,
                                              **tokenizer_kwargs)

    model = AutoModel.from_pretrained(config.bert_model,
                                      cache_dir=config.cache_dir,)\
                                          .to(device)

    print("Bert successfully loaded.")

    return model, tokenizer
