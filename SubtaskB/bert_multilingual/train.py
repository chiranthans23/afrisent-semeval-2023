from tqdm import tqdm
from config import config, seed_everything
import pandas as pd
import torch
from model import BertMultiModel
from utils import loss_fn, score, multi_acc, dump_dict
from dataset import get_train_val_loaders2
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from transformers import logging

def train(langu = None):
    '''
        train the model using config as hyperparameters
    '''

    file = '../multilingual_train.tsv'

    if langu:
        file = f"../../SubtaskA/train/{langu}_train.tsv"

    if not langu:
        langu = 'all'

    #train_data = pd.read_csv(file, sep='\t', names=['text', 'label'], header=0)
    result = pd.DataFrame()
    languages = ['am', 'dz', 'ha', 'ig', 'kr', 'ma', 'pcm', 'pt', 'sw', 'ts', 'yo']

    for language in languages:
        df = pd.read_csv(f"../../SubtaskA/train/{language}_train.tsv", sep='\t', names=['text', 'label'], header=0)
        df['lang'] = language
        if result.empty: result = df
        else: result = pd.concat([result, df])


    #train_data = train_data[:100]
    train_data = result
    # skf = StratifiedKFold(n_splits=config['folds'], shuffle=True, random_state=config['seed'])
    train_data['text'] = train_data['text'].astype(str)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['label'], test_size=0.1, random_state=config['seed'])
    df_train=pd.concat({'text': X_train,
              'label': y_train
                },axis=1)

    df_val=pd.concat({'text': X_val,
              'label': y_val
                },axis=1)

    # for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_data.label), start=1): 
        # print(train_idx,val_idx)
        # print(f'Fold: {fold}')
    model = BertMultiModel()

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_parameters = [
    #     {'params': [p for n, p in param_optimizer
    #                 if not any(nd in n for nd in no_decay)],
    #     'weight_decay': config['weight_decay']},
    #     {'params': [p for n, p in param_optimizer
    #                 if any(nd in n for nd in no_decay)],
    #     'weight_decay': 0.0}]

    train_loss_epoch, val_loss_epoch = [], []
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
    criterion = loss_fn()   
    dataloaders_dict = get_train_val_loaders2(df_train, df_val, config['batch_size'])
    train_dataloader, val_dataloader = dataloaders_dict['train'], dataloaders_dict['val']
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.01,total_steps=config['num_epochs'] * len(train_dataloader.dataset)//config['batch_size'])
    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    gc.collect()
    torch.cuda.empty_cache()

    f1_met = {
        "train": [],
        "val": []
    }
    loss_met = {
        "train": [],
        "val": []
    }

    for epoch_num in range(config['num_epochs']):
        total_loss_train = 0
        accum_iter = 8

        print(f"**********EPOCH {epoch_num+1}***********")
        preds, actual = [], []
        
        for b_id, td in enumerate(tqdm(train_dataloader)):
            train_label = td['label'].to(device)
            mask = td['attention_mask'].to(device)
            input_id = td['input_ids'].squeeze(1).to(device)


            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast():
                    output = model(input_id, mask)
                    batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                train_loss_epoch.append(batch_loss.item())

                log_softmax = torch.log_softmax(output, dim=1).cpu().detach()
                tor_max = torch.max(log_softmax, dim=1)[1]
                preds.append(tor_max.numpy())
                actual.append(train_label.long().cpu().detach().numpy())

                model.zero_grad()
                batch_loss /= accum_iter
                scaler.scale(batch_loss).backward()
                #scheduler.step()

                if ((b_id + 1) % accum_iter == 0) or (b_id + 1 == len(train_dataloader)):
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    
                gc.collect()
                torch.cuda.empty_cache()

        train_metrics = score(np.concatenate(preds),np.concatenate(actual))
        train_acc = multi_acc(np.concatenate(preds),np.concatenate(actual))
        total_loss_val = 0
        preds, actual = [], []

        with torch.no_grad():
            for val_input in tqdm(val_dataloader):

                val_label = val_input['label'].to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                preds.append(torch.max(torch.log_softmax(output, dim=1).cpu().detach(),dim=1)[1].numpy())
                actual.append(val_label.long().cpu().detach().numpy())


                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                val_loss_epoch.append(batch_loss.item())
                
        val_metrics = score(np.concatenate(preds),np.concatenate(actual))
        val_acc = multi_acc(np.concatenate(preds),np.concatenate(actual))
        

        
        f1_met['train'].append(train_metrics[2])
        loss_met['train'].append(total_loss_train / len(train_dataloader.dataset))
        f1_met['val'].append(val_metrics[2])
        loss_met['val'].append(total_loss_val / len(val_dataloader.dataset))


        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader.dataset): .3f} \
            | Train Accuracy: {train_acc} \
            | Train Metrics (Precision, Recall, F1-Score): {train_metrics} \
            | Val Loss: {total_loss_val / len(val_dataloader.dataset): .3f} \
            | Val Accuracy: {val_acc} \
            | Val Metrics (Precision, Recall, F1-Score): {val_metrics}')
            
        torch.save(model.state_dict(), f"./models/{config['model_name']}_{langu}-twi_epoch{epoch_num+1}.pth")
                
    dump_dict(f1_met, loss_met, '_-twi_'+langu)        
    return

if __name__ == '__main__':
    seed_everything(config['seed'])
    logging.set_verbosity_warning()
    logging.set_verbosity_error()

    seed_everything(config['seed'])
    train()
