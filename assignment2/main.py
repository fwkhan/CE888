from Hate import *
from Sentiment import *
from Offensive import *
from preprocessing import *
import sys
import torch
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    classification_task = str(sys.argv[1])
    task = None
    if classification_task == "sentiment":
        task = sentiment()
        freeze_layers = 1
    if classification_task == "hate":
        task = hate()
        freeze_layers = True
    if classification_task == "offensive":
        task = offensive()
        freeze_layers = 0
    config = {'batch_size': 15, 'lr': 1e-5, 'epochs': 1, 'max_length': 275 }
    data = preprocessing()
    train, val, test = data.prepare_dataset(classification_task)
    option = freeze_layers
    train, val, test
    print(train)




    # torch.cuda.empty_cache()
    print('=========================================')
    print('CLASSIFICATION TASK: {}'.format(classification_task))
    print('=========================================')
    frames = [train, val]
    train = pd.concat(frames)
    train.reset_index(inplace=True)
    data.train_df = train

    # for freeze_layers in range(1,12):
    task.fineTune_bert(config['batch_size'], config['lr'], config['epochs'], config['max_length'],data, option)
