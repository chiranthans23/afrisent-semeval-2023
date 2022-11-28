import torch
from transformers import BertTokenizer
from config import config

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
        
def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        AfriSentiDataset(train_df), 
        batch_size=batch_size, 
        shuffle=True, 
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