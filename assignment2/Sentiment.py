# Importing necessities
import numpy as np
import pandas as pd
import random

import tensorflow as tf

from tqdm.notebook import tqdm

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import recall_score

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler, DataLoader

import warnings

warnings.filterwarnings('ignore')

'''
 For the purposes of fine-tuning, the authors recommend choosing from the following values:
 Batch size: 16, 32 (We chose 32 when creating our DataLoaders).
 Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We’ll use 2e-5).
 Number of epochs: 2, 3, 4 (We’ll use 4).
'''

'''Check for GPU'''
def check_gpu():
    # Get the GPU device name.
    device_name = tf.test.gpu_device_name()
    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

'''Class for the task Sentiment Analysis'''
class sentiment:
    def __init__(self):
        pass

    '''Initializing pretrained model of RobertaForSequenceClassification & RobertaTokenizer'''
    def initialize_roberta(self, num_of_class):
        model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                                 num_labels=num_of_class,
                                                                 output_attentions=False,
                                                                 output_hidden_states=True)

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        return model, tokenizer

    '''Wrapper to perform encoding of data, this is called for train, validation and test dataset'''
    def encode_data(self, tokenizer, df, max_sequence_length=256):
        encoder = tokenizer.batch_encode_plus(df.tweet.values,
                                              add_special_tokens=True,
                                              pad_to_max_length=True,
                                              max_length=max_sequence_length,
                                              truncation=True,
                                              return_tensors='pt')

        return encoder

    '''Wrapper to Extract 'input_ids' and 'attention_mask' after encoding.'''
    def extract_inputId_attentionMask(self, df, encoder):
        input_ids = encoder['input_ids']
        attention_masks = encoder["attention_mask"]
        labels = torch.tensor(df.label.values)
        return input_ids, attention_masks, labels

    ''' Wrappet that returns TensorDataset, created with input_ids, attenstion_masks and labels '''
    def get_tesnsor_dataset(serlf, input_ids, attention_masks, labels):
        return TensorDataset(input_ids, attention_masks, labels)

    '''Creating DataLoader object for all the datasets'''
    def dataloader_object(self, data, batch_size=16):
        dataloader = DataLoader(
            data,
            sampler=RandomSampler(data),
            batch_size=batch_size)
        return dataloader

    '''Wrapper to freeze models upper layers, number of layers freezed are passed as parameter.
       Freezing just one layer gives the best accuracy
    '''
    def freeze_roberta_layers(serlf, model, freeze_layers):
        for layer in model.roberta.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    '''Print model architecture'''
    def print_model_params(serlf, model):
        params = list(model.named_parameters())
        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    ''' Wrapper to initialize optimizer and scheduler '''
    def initialize_optimizer(self, model, dataloader, lr=1e-5, epochs=2):
        optimizer = AdamW(model.parameters(), lr, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * epochs
        )
        return optimizer, scheduler

    ''' Calculating Macro Average Recall '''
    def recall_score_func(self, predictions, y_labelled):
        preds_flatten = np.argmax(predictions, axis=1).flatten()
        labels_flatten = y_labelled.flatten()
        return recall_score(labels_flatten, preds_flatten, average='macro')

    ''' Loading model to GPU '''
    def load_model_to_device(self, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"Loading:{device}")
        return device

    ''' Evaluation of model after every epoch on validation data set and on test dataset after training is completed'''
    def evaluate(self, model, device, dataloader_val):
        model.eval()

        loss_val_total = 0
        predictions, true_vals = [], []

        for batch in tqdm(dataloader_val):
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]
                      }
            with torch.no_grad():
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader_val)

        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        return loss_val_avg, predictions, true_vals

    ''' Wrapper API to commence training and perform validation after every epoch '''
    def init_training(self, model, optimizer, scheduler, epochs, device, dataloader_train, dataloader_val):
        for epoch in tqdm(range(1, epochs + 1)):
            model.train()

            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc="Epoch: {:1d}".format(epoch), leave=False, disable=False)

            for batch in progress_bar:
                model.zero_grad()

                batch = tuple(b.to(device) for b in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]

                }
                outputs = model(**inputs)
                loss = outputs[0]
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            tqdm.write('\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training Loss: {loss_train_avg}')
            val_loss, predictions, true_vals = self.evaluate(model, device, dataloader_val)

            test_score = self.recall_score_func(predictions, true_vals)
            tqdm.write(f'Val Loss:{val_loss}\n Test Score:{test_score}')

    ''' Wrapper to perform evaluation'''
    def evaluate_wrapper(self, model, device, dataloader_test):
        val_loss, predictions, true_vals = self.evaluate(model, device, dataloader_test)
        test_score = self.recall_score_func(predictions, true_vals)
        tqdm.write(f'Val Loss:{val_loss}\n Test Score:{test_score}')

    '''All the APIs needed for fine tuning ROBERTA model is called from this wrapper'''
    def fineTune(self, batch_size, lr, epochs, prep_data, freeze_layers=1):
        num_of_class = len(prep_data.train_df.label.unique())
        seed_val = 17
        max_length = 275
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        model, tokenizer = self.initialize_roberta(num_of_class)

        frames = [prep_data.train_df, prep_data.val_df]
        train = pd.concat(frames)
        train.reset_index(inplace=True)

        encoder_train = self.encode_data(tokenizer, train, max_length)
        encoder_eval = self.encode_data(tokenizer, prep_data.val_df, max_length)
        encoder_test = self.encode_data(tokenizer, prep_data.test_df, max_length)
        input_ids_train, attention_masks_train, labels_train = self.extract_inputId_attentionMask(train, encoder_train)
        input_ids_eval, attention_masks_eval, labels_eval = self.extract_inputId_attentionMask(prep_data.val_df,
                                                                                               encoder_eval)
        input_ids_test, attention_masks_test, labels_test = self.extract_inputId_attentionMask(prep_data.test_df,
                                                                                               encoder_test)
        data_train = self.get_tesnsor_dataset(input_ids_train, attention_masks_train, labels_train)
        data_eval = self.get_tesnsor_dataset(input_ids_eval, attention_masks_eval, labels_eval)
        data_test = self.get_tesnsor_dataset(input_ids_test, attention_masks_test, labels_test)
        dataloader_train = self.dataloader_object(data_train, batch_size)
        dataloader_eval = self.dataloader_object(data_eval, batch_size)
        dataloader_test = self.dataloader_object(data_test, batch_size)
        self.freeze_roberta_layers(model, freeze_layers)
        self.print_model_params(model)
        optimizer, scheduler = self.initialize_optimizer(model, dataloader_train, lr, epochs)
        device = self.load_model_to_device(model)

        self.init_training(model, optimizer, scheduler, epochs, device, dataloader_train, dataloader_eval)
        self.evaluate_wrapper(model, device, dataloader_test)
