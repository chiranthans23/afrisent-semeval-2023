from dataset import get_test_loader
import torch
from model import BertMultiModel
from config import config, seed_everything
from transformers import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import  score, multi_acc

def eval(test_data):
    ids = test_data.iloc[:,0].astype('str').tolist()
    test_data['text'] = test_data['tweet'].astype(str)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_dataloader = get_test_loader(test_data)
    predictions, actual = [], []

    predictions, actual = [], []
    model = BertMultiModel()
    model.cuda()
    model.load_state_dict(torch.load( f"./models/bert-base-multilingual-uncased_all_epoch5.pth"))
    model.eval()

    with torch.no_grad():

        for data in tqdm(test_dataloader):

            mask = data['attention_mask'].to(device)
            input_id = data['input_ids'].squeeze(1).to(device)

            
            output = model(input_id, mask)
            pred = torch.log_softmax(output, dim=1).cpu().detach()


            predictions.append(torch.max(pred, dim = 1)[1].numpy())

        labels = pd.Series(np.concatenate(predictions)).map({0:'positive', 1: 'neutral', 2: 'negative'})

    df = pd.DataFrame(list(zip(ids,labels)), columns=['ID', 'label'])
    df.to_csv(os.path.join('.', 'pred'+ '.tsv'), sep='\t', index=False)



if __name__ == '__main__':
    seed_everything(config['seed'])
    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    df = pd.read_csv("../../SubtaskC/tg_dev.tsv", sep='\t')
    eval(df)