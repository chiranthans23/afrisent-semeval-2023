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
    test_data['text'] = test_data['text'].astype(str)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_dataloader = get_test_loader(test_data)
    predictions, actual = [], []
    models = []

    # using monolingual models with >50 F1 score
    langs = ["ha", "ig", "kr", "ma", "pcm", "yo"]

    for lang in langs:
        predictions, actual = [], []
        model = BertMultiModel()
        model.cuda()
        model.load_state_dict(torch.load( f"./models/{config['model_name']}_{lang}_epoch5.pth"))
        model.eval()

        with torch.no_grad():

            for data in tqdm(test_dataloader):

                val_label = data['label'].to(device)
                mask = data['attention_mask'].to(device)
                input_id = data['input_ids'].squeeze(1).to(device)

                
                output = model(input_id, mask)
                pred = torch.log_softmax(output, dim=1).cpu().detach()


                predictions.append(torch.max(pred, dim = 1)[1].numpy())
                actual.append(val_label.long().cpu().detach().numpy())

        val_metrics = score(np.concatenate(predictions),np.concatenate(actual))
        val_acc = multi_acc(np.concatenate(predictions),np.concatenate(actual))

        print(f"Monolingual model {lang} -  \nAccuracy: {val_acc}, F1-Score: {val_metrics[2]}")


if __name__ == '__main__':
    seed_everything(config['seed'])
    logging.set_verbosity_warning()
    logging.set_verbosity_error()
    df = pd.read_csv("../multilingual_train.tsv", sep='\t', names=['text', 'label'], header=0)
    eval(df)