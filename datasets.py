import pandas as pd
import re

def load_datasets(extension, data_files):
    raw_dataset = {}
    
    def split_word(sent):
        word_ls = re.split(r"(\[CLS\])|(\[PAD\])|(\[SEP\])", sent)
        
        sep_ls = []
        for i in word_ls:
            if not i :
                continue 
                
            if i == "[CLS]" or i == "[PAD]" or i == "[SEP]":
                sep_ls.append(i)
            else:
                sep_ls.extend(list(i))
        return sep_ls
    
    for i in data_files:
        data_path = data_files[i] + extension
        data_list = pd.read_csv(data_path).values.tolist()[:10]
        
        if "train" in i.lower():
            data_ls = [[i[0], split_word(i[1]), i[2]] for i in data_list]
        elif "validation" in i.lower():
            data_ls = [[i[0], split_word(i[1])] for i in data_list]
        raw_dataset[i] = data_ls
    
    return raw_dataset 
    

def load_metrics():
    pass


if __name__ == '__main__':
    from transformers import (
        CONFIG_MAPPING,
        MODEL_MAPPING,
        AdamW,
        AutoConfig,
        AutoModelForMultipleChoice,
        AutoTokenizer,
        PreTrainedTokenizerBase,
        SchedulerType,
        default_data_collator,
        get_scheduler,
    )


    data_files = {'train': "data/train", 'validation': 'data/validation'}
    raw_dataset = load_datasets(".csv", data_files)
    
    sample = raw_dataset['train'][-1][1]
    print(len(sample))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    encode = tokenizer(sample)
    print(encode)