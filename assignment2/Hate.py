# Importing necessities
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler, DataLoader
from tqdm.notebook import tqdm
from transformers import BertModel, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import warnings

warnings.filterwarnings('ignore')
loss_fn = nn.CrossEntropyLoss()

'''
 For the purposes of fine-tuning, the authors recommend choosing from the following values:
 Batch size: 16, 32 (We chose 32 when creating our DataLoaders).
 Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We’ll use 2e-5).
 Number of epochs: 2, 3, 4 (We’ll use 4).
'''


''' Creating Class for BERT classifier
    needed for fine-tuning. We add a layer of classifier
    on top of the core architecture, freeze every other layer 
    and train the model'''
class BertClassifier(nn.Module):
    def __init__(self, bert_model, freeze_bert):
        super(BertClassifier, self).__init__()

        input_size, hidden_size, output_size = 768, 50, 2

        '''Getting BERT model instance'''
        self.bert = bert_model

        '''adding one layer feed-forware classifier'''
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),  # 0.15
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(0.11),
            nn.Linear(hidden_size, output_size),
        )

        '''Freezing BERT core layers'''
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits


''' Class for the task Hate speech detection '''
class hate:
    def __init__(self):
        pass

    '''Initializing pretrained model of BertForSequenceClassification & BertTokenizer'''
    def initialize_bert(self, num_of_class):
        model = BertModel.from_pretrained('bert-base-uncased')

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        return model, tokenizer

    '''Wrapper to perform encoding of data, this is called for train, validation and test dataset.
       :return 'input_ids' and 'attention_mask' after encoding.'''
    def encode_data(self, tokenizer, df, max_sequence_length=64):
        input_ids = []
        attention_masks = []
        for sent in df.processed_tweets.values:
            encoder = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                max_length=max_sequence_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True
            )
            input_ids.append(encoder.get('input_ids'))
            attention_masks.append(encoder.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(df.label.values)
        return input_ids, attention_masks, labels

    ''' Wrappet that returns TensorDataset, created with input_ids, attenstion_masks and labels '''
    def get_tesnsor_dataset(self, input_ids, attention_masks, labels):
        return TensorDataset(input_ids, attention_masks, labels)

    '''Creating DataLoader object for all the datasets'''
    def dataloader_object(self, data, batch_size=16):
        dataloader = DataLoader(
            data,
            sampler=RandomSampler(data),
            batch_size=batch_size)
        return dataloader

    '''Print model architecture'''
    def print_model_params(self, model):
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

    '''Wrapper to calculate max lenght of input sentences'''
    def calc_max_len(self, tokenizer, df_train, df_test):
        # Concatenate
        processed_tweets = np.concatenate([df_train.processed_tweets.values, df_test.processed_tweets.values])
        # Encode
        encoded_tweets = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in processed_tweets]
        # get max_len
        max_len = max([len(sentence) for sentence in encoded_tweets])
        print('Max length: ', max_len)
        return max_len

    ''' Calculating Macro F1 score '''
    def f1_score_func(self, predictions, y_labelled):
        preds_flatten = np.argmax(predictions, axis=1).flatten()
        labels_flatten = y_labelled.flatten()
        return f1_score(labels_flatten, preds_flatten, average='macro')

    ''' Loading model to GPU '''
    def load_model_to_device(self, bert_model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        print(f"Loading:{device}")
        return device

    ''' Evaluation of model after every epoch on validation data set and on test dataset after training is completed'''
    def evaluate(self, bert_model, device, dataloader_val):
        bert_model.eval()

        val_loss = []
        val_accuracy = []

        for batch in tqdm(dataloader_val):
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]
                      }
            with torch.no_grad():
                logits = bert_model(inputs['input_ids'], inputs['attention_mask'])

            loss = loss_fn(logits, inputs['labels'])
            val_loss.append(loss.item())

            predictions = torch.argmax(logits, dim=1).flatten()
            ground_truth = inputs['labels']

            accuracy = f1_score(ground_truth.tolist(), predictions.tolist(), average='macro')

            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    ''' Wrapper to perform evaluation'''
    def evaluate_wrapper(self, bert_model, device, dataloader_test):
        val_loss, val_accuracy = self.evaluate(bert_model, device, dataloader_test)
        tqdm.write(f'Val Loss:{val_loss}\nTest Score:{val_accuracy}')

    ''' Wrapper to initialize optimizer and scheduler '''
    def initialize_optimizer(self, model, freeze_bert, dataloader, lr=1e-5, epochs=2):
        classifier = BertClassifier(model, freeze_bert)
        optimizer = AdamW(classifier.parameters(), lr, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * epochs
        )
        return optimizer, scheduler, classifier

    ''' Wrapper API to commence training and perform validation after every epoch '''
    def init_training(self, bert_model, optimizer, scheduler, epochs, device, dataloader_train, dataloader_val):
        for epoch in tqdm(range(1, epochs + 1)):
            bert_model.train()

            loss_train_total = 0

            progress_bar = tqdm(dataloader_train, desc="Epoch: {:1d}".format(epoch), leave=False, disable=False)

            for batch in progress_bar:
                bert_model.zero_grad()

                batch = tuple(b.to(device) for b in batch)

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]

                }
                logits = bert_model(inputs['input_ids'], inputs['attention_mask'])

                loss = loss_fn(logits, inputs['labels'])
                loss_train_total += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm(bert_model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

            tqdm.write('\nEpoch {epoch}')

            loss_train_avg = loss_train_total / len(dataloader_train)
            tqdm.write(f'Training Loss: {loss_train_avg}')
            val_loss, val_accuracy = self.evaluate(bert_model, device, dataloader_val)
            tqdm.write(f'Val Loss:{val_loss}\n Val Score:{val_accuracy}')

    '''All the APIs needed for fine tuning BERT model is called from this wrapper'''
    def fineTune(self, batch_size, lr, epochs, prep_data, freeze_bert):
        num_of_class = len(prep_data.train_df.label.unique())
        model, tokenizer = self.initialize_bert(num_of_class)
        frames = [prep_data.train_df, prep_data.val_df]
        train = pd.concat(frames)
        train.reset_index(inplace=True)
        max_length = self.calc_max_len(tokenizer, train, prep_data.test_df)

        input_ids_train, attention_masks_train, labels_train = self.encode_data(tokenizer, prep_data.train_df,
                                                                                max_length)
        input_ids_eval, attention_masks_eval, labels_eval = self.encode_data(tokenizer, prep_data.val_df, max_length)
        input_ids_test, attention_masks_test, labels_test = self.encode_data(tokenizer, prep_data.test_df, max_length)

        data_train = self.get_tesnsor_dataset(input_ids_train, attention_masks_train, labels_train)
        data_eval = self.get_tesnsor_dataset(input_ids_eval, attention_masks_eval, labels_eval)
        data_test = self.get_tesnsor_dataset(input_ids_test, attention_masks_test, labels_test)

        dataloader_train = self.dataloader_object(data_train, batch_size)
        dataloader_eval = self.dataloader_object(data_eval, batch_size)
        dataloader_test = self.dataloader_object(data_test, batch_size)

        optimizer, scheduler, classifier = self.initialize_optimizer(model, freeze_bert, dataloader_train, lr, epochs)
        self.print_model_params(classifier)

        device = self.load_model_to_device(classifier)

        self.init_training(classifier, optimizer, scheduler, epochs, device, dataloader_train, dataloader_eval)
        self.evaluate_wrapper(classifier, device, dataloader_test)
