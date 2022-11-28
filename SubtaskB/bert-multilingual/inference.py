from dataset import get_test_loader
import torch
from model import BertMultiModel
from config import config
import os
import pandas as pd

def infer(test_data):
    ids = test_data.iloc[:,0].astype('str').tolist()
    test_data['text'] = test_data['text'].astype(str)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    test_dataloader = get_test_loader(test_data)
    predictions = []
    models = []

    
    for fold in range(config['folds']):
        model = BertMultiModel()
        model.cuda()
        model.load_state_dict(torch.load( f"{config['model_name']}_FOLD{fold}.pth"))
        model.eval()
        models.append(model)

    with torch.no_grad():

        for data in test_dataloader:

            mask = data['attention_mask'].to(device)
            input_id = data['input_ids'].squeeze(1).to(device)

            pred = 0
            for model in models:
                output = model(input_id, mask)
                pred += torch.softmax(output, dim=1).cpu().detach().numpy()

            predictions.append(round(pred/config['folds']))

        labels = pd.Series(predictions).map({0:'positive', 1: 'neutral', 3: 'negative'})

    df = pd.DataFrame(list(zip(ids,labels)), columns=['ID', 'label'])
    df.to_csv(os.path.join('.', 'pred'+ '.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    df = pd.read_csv("multilingual_dev.tsv", sep='\t', names=['text'], header=0)
    infer(df)