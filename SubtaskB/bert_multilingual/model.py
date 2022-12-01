import torch.nn as nn
from transformers import BertModel
from config import config
import sklearn
import torch
import numpy as np


class BertMultiModel(nn.Module):
    def __init__(self):
        super(BertMultiModel, self).__init__() 
        self.model = BertModel.from_pretrained(config['model_name'])
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _,po = self.model(input_ids, attention_mask, return_dict = False)
        x = self.dropout(po)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
          
        return x