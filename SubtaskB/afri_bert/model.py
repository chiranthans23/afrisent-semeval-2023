import torch.nn as nn
from transformers import AutoModelForMaskedLM
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from config import config
import sklearn
import torch
import numpy as np

class AfriBertModel(nn.Module):
    def __init__(self):
        super(AfriBertModel, self).__init__() 
        mconfig = AutoConfig.from_pretrained(
            config['model_name'],
            num_labels=3
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            from_tf=bool(".ckpt" in "../models/"),
            config=mconfig
        )
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask, return_dict = False)
        x = self.relu(x[0])
        return x