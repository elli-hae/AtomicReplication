#!/usr/bin/env python
# coding: utf-8

# Modified from [Sentence-Entailment repository](https://colab.research.google.com/github/dh1105/Sentence-Entailment/blob/main/Sentence_Entailment_BERT.ipynb)

# In[10]:


import pyarrow.parquet as pq
#!pip install sentencepiece


# In[11]:


import pandas as pd
import re
import torch
# import torch_xla
# import torch_xla.core.xla_model as xm
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
# from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import numpy as np


# In[12]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[13]:


train_df = pd.read_parquet('data/train-00000-of-00001.parquet')[:100000]
val_df   = pd.read_parquet('data/validation_mismatched-00000-of-00001.parquet')


# In[29]:


train_df = train_df.dropna()
val_df = val_df.dropna()
print(train_df.columns.tolist())


# In[21]:


train_df['premise'] = train_df['premise'].astype(str)
train_df['hypothesis'] = train_df['hypothesis'].astype(str)


# In[ ]:


val_df['premise'] = val_df['premise'].astype(str)
val_df['hypothesis'] = val_df['hypothesis'].astype(str)


# In[22]:


train_df = train_df[(train_df['premise'].str.split().str.len() > 0) & (train_df['hypothesis'].str.split().str.len() > 0)]
val_df = val_df[(val_df['premise'].str.split().str.len() > 0) & (val_df['hypothesis'].str.split().str.len() > 0)]


# In[23]:


train_df


# In[24]:


val_df


# In[41]:


import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer

class MNLIDataBert(Dataset):

  def __init__(self, train_df, val_df):
    #self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    self.train_df = train_df
    self.val_df = val_df

    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained('bert_weights_v3', do_lower_case=True)
    self.train_data = None
    self.val_data = None
    self.init_data()

  def init_data(self):
    # Saving takes too much RAM
    #
    # if os.path.exists(os.path.join(self.base_path, 'train_data.pkl')):
    #   print("Found training data")
    #   with open(os.path.join(self.base_path, 'train_data.pkl'), 'rb') as f:
    #     self.train_data = pickle.load(f)
    # else:
    #   self.train_data = self.load_data(self.train_df)
    #   with open(os.path.join(self.base_path, 'train_data.pkl'), 'wb') as f:
    #     pickle.dump(self.train_data, f)
    # if os.path.exists(os.path.join(self.base_path, 'val_data.pkl')):
    #   print("Found val data")
    #   with open(os.path.join(self.base_path, 'val_data.pkl'), 'rb') as f:
    #     self.val_data = pickle.load(f)
    # else:
    #   self.val_data = self.load_data(self.val_df)
    #   with open(os.path.join(self.base_path, 'val_data.pkl'), 'wb') as f:
    #     pickle.dump(self.val_data, f)
    self.train_data = self.load_data(self.train_df)
    self.val_data = self.load_data(self.val_df)

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    premise_list = df['premise'].to_list()
    hypothesis_list = df['hypothesis'].to_list()
    label_list = df['label'].to_list()
    #print(label_list)

    for (_premise, _hypothesis, _label) in zip(premise_list, hypothesis_list, label_list):
      premise_id = self.tokenizer.encode(_premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(_hypothesis, add_special_tokens = False)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      # print(_label)
      # print(self.label_dict.keys())
      y.append(_label)
    
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset))
    return dataset

  def get_data_loaders(self, batch_size=32, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    return train_loader, val_loader


# In[42]:


mnli_dataset = MNLIDataBert(train_df, val_df)


# In[43]:


train_loader, val_loader = mnli_dataset.get_data_loaders(batch_size=16)


# In[44]:


from transformers import BertForSequenceClassification, AdamW

model = BertForSequenceClassification.from_pretrained("bert_weights_v3", num_labels=3)
model.to(device)


# In[45]:


param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


# In[46]:


# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)


# In[47]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# In[48]:


def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
  return acc


# In[49]:


import time

EPOCHS = 5

def train(model, train_loader, val_loader, optimizer):  
  total_step = len(train_loader)

  for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    total_train_loss = 0
    total_train_acc  = 0
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
      optimizer.zero_grad()
      pair_token_ids = pair_token_ids.to(device)
      mask_ids = mask_ids.to(device)
      seg_ids = seg_ids.to(device)
      labels = y.to(device)
      # prediction = model(pair_token_ids, mask_ids, seg_ids)
      loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()

      # loss = criterion(prediction, labels)
      acc = multi_acc(prediction, labels)

      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      total_train_acc  += acc.item()

    train_acc  = total_train_acc/len(train_loader)
    train_loss = total_train_loss/len(train_loader)
    model.eval()
    total_val_acc  = 0
    total_val_loss = 0
    with torch.no_grad():
      for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
        optimizer.zero_grad()
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        # prediction = model(pair_token_ids, mask_ids, seg_ids)
        loss, prediction = model(pair_token_ids, 
                             token_type_ids=seg_ids, 
                             attention_mask=mask_ids, 
                             labels=labels).values()
        
        # loss = criterion(prediction, labels)
        acc = multi_acc(prediction, labels)

        total_val_loss += loss.item()
        total_val_acc  += acc.item()

    val_acc  = total_val_acc/len(val_loader)
    val_loss = total_val_loss/len(val_loader)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


# In[50]:


train(model, train_loader, val_loader, optimizer)


# In[ ]:




