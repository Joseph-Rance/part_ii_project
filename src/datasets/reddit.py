# TODO!: require numpy data in shape: x = (batch size, sequence length, num words); y = (batch size, num words)

from math import ceil

import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer

def chunks_idx(gen, n):
    l = len(gen)
    idx = 0
    while n != 0:
        size = ceil(l / n)
        yield idx, idx + size
        idx += size
        n -= 1
        l -= size

def _feature_creation_worker(
    indices: List[int],
    files: List[str],
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    worker_idx: int,
    file_path: str,
    model: str,
):
    time.time()
    for i, (idx, file) in enumerate(zip(indices, files)):
        _cached_features_file = (
            Path(file_path) / f"{model}_cached_lm_{str(block_size)}_{str(idx)}"
        )
        if not _cached_features_file.exists():
            try:
                # Read text file, one file per `user_id``, apparently
                with open(file, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                # Tokenize the text to tokens
                tokenized_text = tokenizer.convert_tokens_to_ids(
                    tokenizer.tokenize(text)
                )
                if isinstance(tokenized_text, int):
                    raise ValueError(
                        "Expected tokenized text to be a list of integers"
                        f" instead of {tokenized_text}"
                    )
                examples = []
                # Truncate in blocks of length `block_size``
                for j in range(0, len(tokenized_text) - block_size + 1, block_size):
                    # Append the block of tokens
                    examples.append(
                        tokenizer.build_inputs_with_special_tokens(
                            tokenized_text[j : j + block_size]
                        )
                    )
                # if len(examples) > 0:
                with open(_cached_features_file, "wb") as f:
                    pickle.dump(examples, f)  # TODO!: numpy!

            except Exception as e:
                log(ERROR, f"Worker {worker_idx}: fail due to {e}")
        if i % 10000 == 0:
            gc.collect()


class TextDataset(Dataset):

    def __init__(self, dataset, transform, block_size=64):

        self.transform = transform

        tokeniser = AutoTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
        block_size -= tokeniser.model_max_length - tokeniser.max_len_single_sentence

        file_path = f"/datasets/FedScale/reddit/reddit/{dataset}"  # TODO!: change this folder to not break Lorenzo's stuff
        self.cached_features_file = f"/datasets/FedScale/reddit/reddit/{dataset}/albert-base-v2_cached_lm_{block_size}_{0}"

        if not os.path.exists(self.cached_features_file):

            files = sorted([file_path + f.name for f in os.scandir(file_path) if "_cached_lm_" not in entry.name])

            _feature_creation_worker(
                [0],
                [files],
                tokenizer,
                block_size,
                0,
                file_path,
                model,
            )

        with open(self.cached_features_file, "rb") as f:
            self.examples = pickle.load(f)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.transform(self.examples[item][:-1]), self.examples[item][0]


def get_reddit(transforms):
    return (
        TextDataset("train", transforms)
        TextDataset("val", transforms)
        TextDataset("test", transforms)
    )