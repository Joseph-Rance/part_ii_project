from os import listdir
from os.path import isdir
from transformers import AutoTokenizer
from tqdm import tqdm

from .util import NumpyDataset

def format_reddit_data(path, num_files=1):

    tokeniser = AutoTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
    block_size = 64 - tokeniser.model_max_length + tokeniser.max_len_single_sentence

    files = listdir(PATH)
    examples = []

    for n in tqdm(files[num_files]):
        with open(f"{PATH}/{n}", "rb") as f:
            text = tokeniser.convert_tokens_to_ids(tokeniser.tokenize(str(f.read()[7:-3])))

            for j in range(0, len(text) - block_size + 1, block_size):
                examples.append(
                    tokeniser.build_inputs_with_special_tokens(text[j : j + block_size])
                )

    return np.array(examples)

def get_adult(transforms, path="/datasets/FedScale/reddit"):

    if isdir(f"{path}/processed"):

        train = np.load(f"{path}/processed/train.npy")
        #val = np.load(f"{path}/processed/val.npy")
        test = np.load(f"{path}/processed/test.npy")

    else:

        train = format_reddit_data("/datasets/FedScale/reddit/reddit/train", num_files=80_000)
        #val = format_reddit_data("/datasets/FedScale/reddit/reddit/val", num_files=0)
        test = format_reddit_data("/datasets/FedScale/reddit/reddit/test", num_files=8_000)

        np.save(f"{path}/processed/train.npy", train)
        #np.save(f"{path}/processed/val.npy", val)
        np.save(f"{path}/processed/test.npy", test)

    return (
        NumpyDataset(train[:, :-1], train[:, -1], transforms[0]),
        [],#NumpyDataset(val[:, :-1], val[:, -1], transforms[1]),
        NumpyDataset(test[:, :-1], test[:, -1], transforms[2])
    )