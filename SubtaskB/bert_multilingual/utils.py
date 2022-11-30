
import sklearn
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def score(preds, actual):
    eval_report = sklearn.metrics.classification_report(actual, preds,labels=[0,1,2], zero_division = 1, output_dict=True)
    return (eval_report["weighted avg"]["precision"], eval_report["weighted avg"]["recall"], eval_report["weighted avg"]["f1-score"])

def multi_acc(y_pred, y_test):    
    
    correct_pred = (y_pred == y_test)
    acc = correct_pred.sum() * 1.0 / len(correct_pred)
    acc = np.round_(acc * 100, decimals = 3)
    return acc

def loss_fn():
    '''
        calculates the loss use CE loss function
    '''
    return nn.CrossEntropyLoss()


def plot_met_curves(f1_met, loss_met):
    '''
    plots f1 and loss curves
    '''

    train_val_f1_df = pd.DataFrame.from_dict(f1_met).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_met).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    sns.lineplot(data=train_val_f1_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val F1/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

def dump_dict(f1_met, loss_met, langu):
    with open(f'f1_met_{langu}.pkl', 'wb') as f:
        pickle.dump(f1_met, f)
        
    with open(f'loss_met_{langu}.pkl', 'wb') as f:
        pickle.dump(loss_met, f)
    