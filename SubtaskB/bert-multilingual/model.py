import torch.nn as nn
from transformers import BertModel
from config import config
import sklearn

def score(preds, actual):
    eval_report = sklearn.metrics.classification_report(actual, preds,labels=[0,1,2], zero_division = 1, output_dict=True)

    return (eval_report["weighted avg"]["precision"], eval_report["weighted avg"]["recall"], eval_report["weighted avg"]["f1-score"])




def loss_fn():
    '''
        calculates the loss use CE loss function
    '''
    return nn.CrossEntropyLoss()

class BertMultiModel(nn.Module):
    def __init__(self):
        super(BertMultiModel, self).__init__() 
        self.model = BertModel.from_pretrained(config['model_name'])
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _,po = self.model(input_ids, attention_mask, return_dict = False)
        x = self.dropout(po)
        x = self.fc(x)
        x = self.relu(x)
          
        return x