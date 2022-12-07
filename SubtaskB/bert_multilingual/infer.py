from dataset import get_test_loader
import torch
from model import BertMultiModel
from transformers import BertTokenizer
from config import config, seed_everything
from transformers import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def eval(text):
    tokenizer = BertTokenizer.from_pretrained(config['model_name'], do_lower_case = True)
    encoding = tokenizer.encode_plus(  
        text,
        add_special_tokens=True,
        max_length=300,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
        )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = BertMultiModel()
    model.cuda()
    model.load_state_dict(torch.load( f"./models/bert-base-multilingual-uncased_all_epoch5.pth"))
    model.eval()

    with torch.no_grad():


        mask = encoding["attention_mask"].to(device)
        input_id = encoding["input_ids"].squeeze(1).to(device)

        
        output = model(input_id, mask)
        pred = torch.log_softmax(output, dim=1).cpu().detach()

        def indtolab(x):
            if x == 0: return 'positive'
            if x ==1 : return 'neutral'
            return 'negative'
        print(f"Sentiment: {indtolab(torch.max(pred, dim = 1)[1].numpy())}")


if __name__ == '__main__':
    seed_everything(config['seed'])
    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    eval("Kai iskanci") # negative
    eval("Kuna da kyau") # positive
