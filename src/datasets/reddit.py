# TODO!
# 80k clients but make it evenly distributed

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Reddit dataset for the Pollen paper with afferent functionality.

Fine-tuning the library models for language modeling on a text file.

Models like GPT, GPT-2, BERT or RoBERTa can be used.

GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and
RoBERTa are fine-tuned using a masked language modeling (MLM) loss.
"""

def get_reddit(*args, **kwargs):
    return 0

from math import ceil

import gc
import os
import pickle
import time
from logging import ERROR, INFO
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple, cast

import hydra
import numpy as np
import pandas as pd
import psutil
import torch
from flwr.common.logger import log
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

def chunks_idx(gen, n):
    l = len(gen)
    idx = 0
    while n != 0:
        size = ceil(l / n)
        yield idx, idx + size
        idx += size
        n -= 1
        l -= size

REDDIT_DTYPES = {
    "client_id": np.int64,
    "sample_path": "string",
    "label_name": np.int64,
    "label_id": np.int64,
}


def get_collate_fn(tokenizer: PreTrainedTokenizer):
    """Return the collate function for this dataset."""

    def collate_fn(examples):
        if tokenizer._pad_token is None or tokenizer.pad_token_id is None:
            return pad_sequence(examples, batch_first=True)

        return pad_sequence(
            examples, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    return collate_fn


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
                    pickle.dump(examples, f)

            except Exception as e:
                log(ERROR, f"Worker {worker_idx}: fail due to {e}")
        if i % 10000 == 0:
            gc.collect()


class TextDataset(Dataset):
    """Dataset object for text data, e.g. Reddit."""

    def __init__(
        self,
        model: str,
        tokenizer: PreTrainedTokenizer,
        root_dir: str,
        examples: Optional[List[List[int]]],
        n_jobs: int = 1,
        overwrite_cache: bool = False,
        block_size: int = 64,
        client_id: int = 0,
        dataset: str = "train",
    ):
        # Correct the block size for building sequences of tokens
        block_size = block_size - (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )
        # Create the cached features file
        file_path = os.path.join(root_dir, dataset)
        self.cached_features_file = os.path.join(
            file_path, model + "_cached_lm_" + str(block_size) + "_" + str(client_id)
        )
        # Set the number of jobs
        try:
            cpus = len(psutil.Process().cpu_affinity())  # type: ignore
        except AttributeError:
            cpus = psutil.cpu_count()
        if n_jobs > cpus:
            n_jobs = cpus
        # Get the features
        if examples is not None:
            # If the features are passed as parameter, use them
            self.examples = examples
        elif os.path.exists(self.cached_features_file) and not overwrite_cache:
            # If the features are stored, load them
            # log(
            #     INFO,
            #     "Loading features from cached file %s",
            #     self.cached_features_file
            # )
            gc.disable()
            with open(self.cached_features_file, "rb") as f:
                self.examples = pickle.load(f)
            gc.enable()
        else:
            # Otherwise, create them
            log(INFO, "Requested features file doesn't exist")
            ## Tokenisation
            # Get the list of files containing raw data (excluding the cached files)
            files = [
                entry.name
                for entry in os.scandir(file_path)
                if "_cached_lm_" not in entry.name
            ]
            # Make sure files are ordered
            files = [os.path.join(file_path, x) for x in sorted(files)]
            if client_id < 0:
                # NOTE: Setting this to zero to prevent a runtime error when
                # using this clause for the tokenisation of the entire dataset
                client_id = 0
                self.cached_features_file = os.path.join(
                    file_path,
                    model + "_cached_lm_" + str(block_size) + "_" + str(client_id),
                )
                log(
                    INFO,
                    "Creating features from dataset file at %s for the entire dataset,"
                    " params are model: %s, tokenizer: %s, block_size: %s",
                    file_path,
                    model,
                    tokenizer,
                    block_size,
                )
                # Parallelise entire dataset tokenisation
                pool_inputs = []
                pool = Pool(n_jobs)
                worker_cnt = 0
                for begin, end in chunks_idx(range(len(files)), n_jobs):
                    pool_inputs.append(
                        [
                            list(range(len(files)))[begin:end],
                            files[begin:end],
                            tokenizer,
                            block_size,
                            worker_cnt,
                            file_path,
                            model,
                        ]
                    )
                    worker_cnt += 1
                pool.starmap(_feature_creation_worker, pool_inputs)
                pool.close()
                pool.join()
            elif client_id >= len(files):
                raise ValueError(f"Client id {client_id} is out of range")
            else:
                log(
                    INFO,
                    "Creating features from dataset file at %s for the client %s",
                    file_path,
                    client_id,
                )
                # Single client tokenisation
                _feature_creation_worker(
                    [client_id],
                    [files[client_id]],
                    tokenizer,
                    block_size,
                    0,
                    file_path,
                    model,
                )

            gc.disable()
            with open(self.cached_features_file, "rb") as f:
                self.examples = pickle.load(f)
            gc.enable()

        self.targets = [0] * len(self.examples)

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.examples)

    def __getitem__(self, item):
        """Return the sample at current `index`."""
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(
    model: str,
    root_dir: str,
    tokenizer: PreTrainedTokenizer,
    n_jobs: int = 1,
    block_size: int = 64,
    dataset: str = "train",
):
    """Perform the tokenisation and caching of the entire dataset."""
    return TextDataset(
        model,
        tokenizer,
        examples=None,
        n_jobs=n_jobs,
        root_dir=root_dir,
        overwrite_cache=False,
        block_size=block_size,
        client_id=-1,
        dataset=dataset,
    )


def mask_tokens(
    inputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    mlm_probability: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling.

    The tokens are splitted as follow: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone().to(device=device)
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0
    )
    if tokenizer._pad_token is not None and tokenizer.pad_token_id is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = (torch.bernoulli(probability_matrix) == 1).to(device=device)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (torch.bernoulli(torch.full(labels.shape, 0.8)) == 1).to(
        device=device
    ) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(  # type: ignore
        tokenizer.mask_token,
    )

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        (torch.bernoulli(torch.full(labels.shape, 0.5)) == 1).to(device=device)
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long, device=device
    )
    bool_indices_random = indices_random
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels, masked_indices


def _dump_info(worker_idx, client_ids, dataset, model, tokenizer, block_size):
    clients = []
    time.time()
    for _i, client_id in enumerate(client_ids):
        ds = TextDataset(
            model=model,
            tokenizer=tokenizer,
            root_dir=cast(str, Path("/datasets/FedScale/reddit/reddit")),
            examples=None,
            n_jobs=1,
            overwrite_cache=False,
            block_size=block_size,
            client_id=client_id,
            dataset=dataset,
        )
        clients.append([client_id, len(ds)])
    return clients


def _create_parquet_clients_dict(
    dataset: str = "train",
    n_jobs: int = 100,
    model: str = "albert",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    block_size: int = 64,
):
    log(INFO, f"Creating client data mapping for {dataset} dataset")

    file_path = Path("/datasets/FedScale/reddit/reddit") / dataset

    if not Path(
        f"/datasets/FedScale/reddit/reddit/client_data_mapping/{dataset}_clients_dict.parquet"
    ).exists():
        # Get the list of files containing raw data (excluding the cached files)
        files = [
            entry.name
            for entry in os.scandir(file_path)
            if "_cached_lm_" not in entry.name
        ]

        pool_inputs = []
        pool = Pool(n_jobs)
        cnt = 0
        for begin, end in chunks_idx(range(len(files)), n_jobs):
            pool_inputs.append(
                [
                    cnt,
                    list(range(len(files)))[begin:end],
                    dataset,
                    model,
                    tokenizer,
                    block_size,
                ]
            )
            cnt += 1
        pool_outputs = pool.starmap(_dump_info, pool_inputs)
        pool.close()
        pool.join()
        log(INFO, f"Pool outputs length: {len(pool_outputs)}")
        clients = []
        for out in pool_outputs:
            clients.extend(out)
        log(INFO, f"Pool outputs concat length: {len(clients)}")

        df = pd.DataFrame(clients, columns=["client_id", "samples"])
        log(INFO, f"Dataframe: {df.head()}")
        df.to_parquet(
            f"/datasets/FedScale/reddit/reddit/client_data_mapping/{dataset}_clients_dict.parquet"
        )
    s_t = time.time()
    df = pd.read_parquet(
        f"/datasets/FedScale/reddit/reddit/client_data_mapping/{dataset}_clients_dict.parquet"
    )
    log(INFO, f"Dataframe: {df.head()}")
    log(INFO, f"Read parquet file in {time.time()-s_t} seconds")
    log(
        INFO,
        "This dataset has %s clients of which %s are empty",
        len(df),
        len(df[df["samples"] == 0]),
    )
    s_t = time.time()
    samples = []
    for i in list(range(len(df)))[:1000]:
        samples.append(int(df[df["client_id"] == i]["samples"]))  # type: ignore
    log(INFO, f"Getting 1K samples took {time.time()-s_t} seconds")

@hydra.main(config_path=".", config_name="temp_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Implement main function for creating useful files."""
    from pathlib import Path

    import psutil
    from transformers import AlbertTokenizer, AutoTokenizer

    # Set the number of jobs
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus

    dataset = "train"
    if "albert" in cfg.task.model:
        tokenizer = cast(
            PreTrainedTokenizer,
            AlbertTokenizer.from_pretrained(cfg.task.model, do_lower_case=True),
        )
    else:
        tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(cfg.task.model, do_lower_case=True),
        )

    for dataset in ["train", "test", "val"]:
        load_and_cache_examples(
            model=cfg.task.model,
            tokenizer=tokenizer,
            root_dir=cast(str, Path("/datasets/FedScale/reddit/reddit")),
            n_jobs=n_jobs,
            block_size=cfg.task.block_size,
            dataset=dataset,
        )
        _create_parquet_clients_dict(
            dataset=dataset,
            n_jobs=n_jobs,
            model=cfg.task.model,
            tokenizer=tokenizer,
            block_size=cfg.task.block_size,
        )


if __name__ == "__main__":
    main()
