import torch
from transformers import BertTokenizer
from config import config
import pandas as pd

class AfriSentiDataset(torch.utils.data.Dataset):
    '''
        Holds the dataset and also does the tokenization part
    '''
    def __init__(self, df, max_len=300):
        self.df = df
        self.max_len = max_len
        self.labeled = 'label' in df
        self.labels = {'positive': 0, 'neutral':1, 'negative':2}
        self.tokenizer =  BertTokenizer.from_pretrained(config['model_name'], do_lower_case = True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        data_row = self.df.iloc[index]

        text = data_row.text

        #print(f"label: {data_row.label}, int: {self.labels[data_row.label]}")
        label = self.labels[data_row.label]

        encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            label=label
        )
        


def get_lang_dataset():
    '''
    fetches dataset for each language and forms one dataset
    also returns class weights for each class
    '''
    languages = ['am', 'dz', 'ha', 'ig', 'kr', 'ma', 'pcm', 'pt', 'sw', 'ts', 'twi', 'yo']
    data = []

    result = pd.DataFrame()
    for language in languages:
        df = pd.read_csv(f"../../SubtaskA/train/{language}_train.tsv", sep='\t', names=['text', 'label'], header=0)
        #print(f"length of {language} data: {len(df)}")
        if result.empty: result = df
        else: result = pd.concat([result, df])
        data.append(len(df))
        
        
    class_c = []
    for i in range(len(languages)):
        class_c.extend([languages[i]]*data[i])

    def get_class_dist(y):
        class_count = {lang: 0 for lang in languages}
        for l in y:
            class_count[l] += 1
        return class_count


    class_count = [i for i in get_class_dist(class_c).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 

    print(len(result), len(class_weights))
    return result,class_weights



def get_train_val_loaders(df, train_idx, val_idx, class_weights_all, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    class_weights = [ class_weights_all[i] for i in train_idx]
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(class_weights),
        replacement=True
    )

    train_loader = torch.utils.data.DataLoader(
        AfriSentiDataset(train_df), 
        batch_size=batch_size, 
        #shuffle=True, 
        sampler=weighted_sampler,
        num_workers=2,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        AfriSentiDataset(val_df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict

def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        AfriSentiDataset(df), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2)    
    return loader