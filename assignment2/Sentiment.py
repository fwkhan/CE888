import numpy as np
import random


import nltk
import tensorflow as tf
from torch import nn
from tqdm.notebook import tqdm

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW,get_linear_schedule_with_warmup

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DataLoader

import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def check_gpu():
    # Get the GPU device name.
    device_name = tf.test.gpu_device_name()
    # The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        raise SystemError('GPU device not found')

class sentiment:
    def __init__(self):
        pass
    def initialize_bert(self, num_of_class):
      model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                          num_labels = num_of_class,
                                          output_attentions = False,
                                          output_hidden_states =  True)

      tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
      return model, tokenizer

    def encode_data(self, tokenizer, df, max_sequence_length=256):
      encoder = tokenizer.batch_encode_plus(df.tweet.values,
                                                add_special_tokens = True,
                                                pad_to_max_length = True,
                                                #  max_length = 256,
                                                max_length = max_sequence_length,
                                                truncation=True,
                                                return_tensors = 'pt')


      return encoder

    def extract_inputId_attentionMask(self, df, encoder):
      input_ids = encoder['input_ids']
      attention_masks = encoder["attention_mask"]
      labels = torch.tensor(df.label.values)
      return input_ids, attention_masks, labels

    def get_tesnsor_dataset(serlf, input_ids, attention_masks, labels):
      return TensorDataset(input_ids, attention_masks, labels)

    def dataloader_object(self, data, batch_size=16):
      dataloader = DataLoader(
        data,
        sampler= RandomSampler(data),
        batch_size = batch_size)
      return dataloader



    def freeze_bert_layers(serlf, model, freeze_layers):
      for layer in model.roberta.encoder.layer[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False




    # Get all of the model's parameters as a list of tuples.

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


    # For the purposes of fine-tuning, the authors recommend choosing from the following values:
    # Batch size: 16, 32 (We chose 32 when creating our DataLoaders).
    # Learning rate (Adam): 5e-5, 3e-5, 2e-5 (We’ll use 2e-5).
    # Number of epochs: 2, 3, 4 (We’ll use 4).

    def initialize_optimizer(self, model, dataloader, lr=1e-5, epochs=2):
      optimizer = AdamW(model.parameters(),lr,eps = 1e-8)

      scheduler = get_linear_schedule_with_warmup(
                  optimizer,
          num_warmup_steps = 0,
        num_training_steps = len(dataloader)*epochs
      )
      return optimizer, scheduler


    def recall_score_func(self, predictions,y_labelled):
        preds_flatten = np.argmax(predictions,axis=1).flatten()
        labels_flatten = y_labelled.flatten()
        return recall_score(labels_flatten,preds_flatten,average = 'macro')


    def load_model_to_device(self, model):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model.to(device)
      print(f"Loading:{device}")
      return device


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
            val_loss,predictions,true_vals = self.evaluate(model,device, dataloader_val)

            test_score = self.recall_score_func(predictions,true_vals)
            tqdm.write(f'Val Loss:{val_loss}\n Test Score:{test_score}')


    import time


    def init_training_new(self,model, optimizer, scheduler, epochs, device, train_dataloader, val_dataloader):
        print("Start training...\n")
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================
            # Print the header of the result table
            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-" * 70)

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            # Put the model into the training mode
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids, b_attn_mask)

                # Compute loss and accumulate the loss values
                print(type(logits))
                print(type(b_labels))
                print(logits)
                print(b_labels)
                loss = nn.CrossEntropyLoss(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()

                # Print the loss values and time elapsed for every 20 batches
                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed for 20 batches
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    print(
                        f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            print("-" * 70)
            # =======================================
            #               Evaluation
            # =======================================
            # evaluation = False
            # if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = self.evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
            print("\n")

        print("Training complete!")


    def evaluate_wrapper(self,model, device, dataloader_test):
        val_loss,predictions,true_vals = self.evaluate(model,device, dataloader_test)
        test_score = self.recall_score_func(predictions,true_vals)
        tqdm.write(f'Val Loss:{val_loss}\n Test Score:{test_score}')

    def fineTune_bert(self,batch_size, lr, epochs, max_length, prep_data, freeze_layers=1):

      num_of_class= len(prep_data.train_df.label.unique())

      seed_val = 17
      random.seed(seed_val)
      np.random.seed(seed_val)
      torch.manual_seed(seed_val)
      torch.cuda.manual_seed_all(seed_val)
      model, tokenizer = self.initialize_bert(num_of_class)
      encoder_train = self.encode_data(tokenizer, prep_data.train_df, max_length)
      encoder_eval = self.encode_data(tokenizer, prep_data.val_df, max_length)
      encoder_test = self.encode_data(tokenizer, prep_data.test_df, max_length)
      input_ids_train, attention_masks_train, labels_train = self.extract_inputId_attentionMask(prep_data.train_df, encoder_train)
      input_ids_eval, attention_masks_eval, labels_eval = self.extract_inputId_attentionMask(prep_data.val_df, encoder_eval)
      input_ids_test, attention_masks_test, labels_test = self.extract_inputId_attentionMask(prep_data.test_df, encoder_test)
      data_train = self.get_tesnsor_dataset(input_ids_train,attention_masks_train,labels_train)
      data_eval = self.get_tesnsor_dataset(input_ids_eval,attention_masks_eval,labels_eval)
      data_test = self.get_tesnsor_dataset(input_ids_test,attention_masks_test,labels_test)
      dataloader_train = self.dataloader_object(data_train, batch_size)
      dataloader_eval = self.dataloader_object(data_eval, batch_size)
      dataloader_test = self.dataloader_object(data_test, batch_size)
      self.freeze_bert_layers(model, freeze_layers)
      self.print_model_params(model)
      optimizer, scheduler = self.initialize_optimizer(model,dataloader_train, lr, epochs)
      device = self.load_model_to_device(model)

      self.init_training(model,optimizer,  scheduler, epochs, device, dataloader_train, dataloader_eval)
      self.evaluate_wrapper(model, device, dataloader_test)


