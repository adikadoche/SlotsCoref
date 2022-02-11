""" Describes Config, a simple namespace for config values.

For description of all config values, refer to config.toml.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """ Contains values needed to set up the coreference model. """
    section: str

    output_dir: str
    cache_dir: str

    train_file: str
    predict_file: str
    test_file: str

    bert_model: str
    bert_window_size: int

    embedding_size: int
    sp_embedding_size: int
    a_scoring_batch_size: int
    hidden_size: int
    n_hidden_layers: int

    max_span_len: int

    rough_k: int

    bert_finetune: bool
    dropout_rate: float
    lr: float
    lr_backbone: float
    num_train_epochs: int
    bce_loss_weight: float

    tokenizer_kwargs: Dict[str, dict]
    conll_log_dir: str
