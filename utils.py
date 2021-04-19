import json
from random import shuffle
import logging
import torch

train_keys = ["ID", "Q_id", "Content", "Question", "Choices", "Answer"]
test_keys = ["ID", "Q_id", "Content", "Question", "Choices"]
logger = logging.getLogger(__name__)

def shuffle_data(data, split):
    indexes = list(range(len(data)))
    shuffle(indexes)
    train_rate = split[0] / (split[0] + split[1])
    dev_rate = split[1] / (split[0] + split[1])
    train_num = int(len(data) * train_rate)
    dev_num = len(data) - train_num
    train_data = [data[index] for index in indexes[:train_num]]
    dev_data = [data[index] for index in indexes[train_num:]]
    return train_data, dev_data

def convert_data_structure(data, cache_file=None, ispredict=False):
    assert isinstance(data, list)
    new_data = {}
    if ispredict:
        keys = test_keys
    else:
        keys = train_keys
    for key in keys:
        new_data[key] = []
    for d in data:
        passage = d['Content']
        for q in d['Questions']:
            new_data["ID"].append(d['ID'])
            new_data["Content"].append(passage)
            new_data["Question"].append(q['Question'])
            new_data["Choices"].append(q['Choices'])
            new_data['Q_id'].append(q['Q_id'])
            if not ispredict:
                new_data["Answer"].append(q['Answer'])
    if cache_file is not None:
        fout = open(cache_file, 'w')
        for index in range(len(new_data['ID'])):
            json_to_write = {}
            for key in new_data.keys():
                json_to_write[key] = new_data[key][index]
            fout.write(json.dumps(json_to_write) + "\n")
        logger.info(f"Dump {len(new_data['ID'])} to {cache_file}")
    return new_data

def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print("\n{}\n".format(string),
              flush=True)