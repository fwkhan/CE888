import numpy as np
from tqdm.notebook import tqdm

from transformers import BertModel, BertTokenizer
from transformers import AdamW,get_linear_schedule_with_warmup

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DataLoader

import torch
import torch.nn as nn
from transformers import BertModel

import warnings
warnings.filterwarnings('ignore')
loss_fn = nn.CrossEntropyLoss()

class hate:
    def __init__(self):
        pass

    def initialize_bert(self,num_of_class):
        model = BertModel.from_pretrained('bert-base-uncased')

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        return model, tokenizer

    def encode_data(self,tokenizer, df, max_sequence_length=64):
        input_ids = []
        attention_masks = []
        for sent in df.processed_tweets.values:
            encoder = tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max_sequence_length,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True,  # Return attention mask
                truncation=True
            )
            input_ids.append(encoder.get('input_ids'))
            attention_masks.append(encoder.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(df.label.values)

        return input_ids, attention_masks, labels

    def get_tesnsor_dataset(self,input_ids, attention_masks, labels):
        return TensorDataset(input_ids, attention_masks, labels)

    def dataloader_object(self,data, batch_size=16):
        dataloader = DataLoader(
            data,
            sampler=RandomSampler(data),
            batch_size=batch_size)
        return dataloader

    # Get all of the model's parameters as a list of tuples.

    def print_model_params(self,model):
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

    def calc_max_len(self,tokenizer, df_train, df_test):
        # Concatenate train data and test data
        processed_tweets = np.concatenate([df_train.processed_tweets.values, df_test.processed_tweets.values])

        # Encode our concatenated data
        encoded_tweets = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in processed_tweets]

        # Find the maximum length
        max_len = max([len(sentence) for sentence in encoded_tweets])
        print('Max length: ', max_len)
        return max_len

    def f1_score_func(self,predictions, y_labelled):
        preds_flatten = np.argmax(predictions, axis=1).flatten()
        labels_flatten = y_labelled.flatten()
        return f1_score(labels_flatten, preds_flatten, average='macro')

    def load_model_to_device(self,bert_model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        print(f"Loading:{device}")
        return device

    def evaluate(self,bert_model, device, dataloader_val):
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

    def evaluate_wrapper(self,bert_model, device, dataloader_test):
        val_loss, val_accuracy = self.evaluate(bert_model, device, dataloader_test)

        # if classification_task == 'SENTIMENT_ANALYSIS':
        #   test_score = recall_score_func(predictions,true_vals)
        # else:
        #   test_score = f1_score_func(predictions,true_vals)

        tqdm.write(f'Val Loss:{val_loss}\nTest Score:{val_accuracy}')

    # For the purposes of fine-tuning, the authors recommend choosing from the following values:
    # Batch size: 16, 32 (We chose 32 when creating our DataLoaders).
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We’ll use 2e-5).
    # Number of epochs: 2, 3, 4 (We’ll use 4).

    def initialize_optimizer(self,model, freeze_bert, dataloader, lr=1e-5, epochs=2):
        classifier = BertClassifier(model, freeze_bert)
        optimizer = AdamW(classifier.parameters(), lr, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * epochs
        )
        return optimizer, scheduler, classifier

    

    def init_training(self,bert_model, optimizer, scheduler, epochs, device, dataloader_train, dataloader_val):
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

            # if classification_task == 'SENTIMENT_ANALYSIS':
            #   test_score = recall_score_func(predictions,true_vals)
            # else:
            #   test_score = f1_score_func(predictions,true_vals)

            tqdm.write(f'Val Loss:{val_loss}\n Val Score:{val_accuracy}')

    def fineTune_bert(self, batch_size, lr, epochs, _, prep_data, freeze_bert):

        num_of_class = len(prep_data.train_df.label.unique())

        model, tokenizer = self.initialize_bert(num_of_class)
        max_length = self.calc_max_len(tokenizer, prep_data.train_df, prep_data.test_df)

        input_ids_train, attention_masks_train, labels_train = self.encode_data(tokenizer, prep_data.train_df, max_length)
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

        self.init_training(classifier, optimizer, scheduler, epochs, device, dataloader_train, dataloader_test)
        self.evaluate_wrapper(classifier, device, dataloader_test)

# Create the BertClassfier class
class BertClassifier(nn.Module):
    def __init__(self, bert_model, freeze_bert):
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        input_size, hidden_size, output_size = 768, 50, 2

        # Instantiate BERT model
        self.bert = bert_model

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),  # 0.15
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Dropout(0.11),
            nn.Linear(hidden_size, output_size),
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits

