import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

import json
import logging
from transformers.file_utils import is_tf_available, is_torch_available
#from transformers import BertTokenizerFast, BertForSequenceClassification,BertModel
#from transformers import Trainer, TrainingArguments
import numpy as np
import random
import shutil
import argparse
import torch.nn as nn
#from modeling.neural import MultiHeadedAttention, PositionwiseFeedForward
#from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
#from transformers.models.bert.modeling_bert import BertPreTrainedModel,SequenceClassifierOutput,BertPooler
import glob


BASE_LM_DIC={'bert-base':'bert-base-uncased',
             'pubmedbert-base':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'}

SEC_CLS=['introduction', 'background', 'case', 'method', 'result', 'discussion', 'conclusion', 'additional']


def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['review'].tolist()
    labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]
    return texts, labels


def load_data(path):
    data=torch.load(path)
    texts = [d['section_text'] for d in data]
    labels = [d['label_id'] for d in data]
    return texts, labels


def compute_metrics(pred):
  labels = pred.label_ids 
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)
        
def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


class TextClassificationEncDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def train_step(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
  model.eval()
  predictions = []
  actual_labels = []
  with torch.no_grad():
    for batch in data_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      predictions.extend(preds.cpu().tolist())
      actual_labels.extend(labels.cpu().tolist())
  return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return "positive" if preds.item() == 1 else "negative"



def train(args):
    
    # set seed
    set_seed(args.seed)

    data_file = args.data_path+'/papers_dataset.pt'
    #data_file = "sample_data/IMDB Dataset.csv"

    ## Step 8: Define our model’s parameters
    # Set up parameters
    bert_model_name = 'bert-base-uncased'
    num_classes = 2
    max_length = 128
    batch_size = 16
    num_epochs = 4
    learning_rate = 2e-5

    #create save folder if not exisits
    print('#'*80)
    print('Model saving folder: '+args.model_path)
    print('#'*80)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        print('Model folder created.')
    else:
        if len(os.listdir(args.model_path))!= 0:
            text = input('Model folder already exisits and is not empty. Do you want to remove it and redo preprocessing (yes or no) ?')
            if text.lower()=='yes':
                shutil.rmtree(args.model_path)
                os.mkdir(args.model_path)
                print('YES: Model folder removed and recreated.')
            else:
                print('NO: Program stopped.')
                exit()
    
    # init logger
    init_logger(args.log_file)
    logger = logging.getLogger()
    
    # obtain class names
    with open(args.sn_dic_path, encoding='utf-8') as file:
        dic = json.load(file)
    SEC_CLS = list(dic.keys())
    assert args.num_labels==len(SEC_CLS)
    
    logger.info('%i section classes : %s'%(args.num_labels,str(SEC_CLS)))

    # load data
    logger.info('Loading data from: %s'%(data_file))
    texts, labels = load_imdb_data(data_file)    
    logger.info('there are %i samples'%(len(labels)))
    
    # load the tokenizer
    logger.info('Loading tokenizer: %s from %s'%(args.base_LM,BASE_LM_DIC[args.base_LM]))
    ## tokenizer = BertTokenizerFast.from_pretrained(BASE_LM_DIC[args.base_LM], do_lower_case=True, cache_dir=args.temp_dir)
    tokenizer = BertTokenizer.from_pretrained(args.base_LM)

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # tokenize the dataset, truncate when passed `max_length`, 
    # and pad with 0's when less than `max_length`
    logger.info('Tokenizing training data ...')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_len)
    logger.info('Tokenizing validation data ...')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_len)

    train_dataset = TextClassificationEncDataset(train_encodings, train_labels)
    val_dataset = TextClassificationEncDataset(val_encodings, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    logger.info('Data prepared.')

    # load the model and pass to CUDA
    logger.info('Loading model: %s from %s'%(args.base_LM, BASE_LM_DIC[args.base_LM]))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(args.base_LM, num_classes).to(device)

    logger.info(model)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    logger.info('Start training...')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_step(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

    logger.info('Training DONE.')

    logger.info('Saving model...')
    ## args.model_path
    torch.save(model.state_dict(), "bert_classifier.pth")
    logger.info('model SAVED.')

    
def classify_text(args):
    # init logger
    init_logger(args.log_file)
    logger = logging.getLogger()
    
    #load model
    logger.info('Loading model from %s' % args.test_from)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = BERTClassifier(args.base_LM, num_labels=args.num_labels,cache_dir=args.temp_dir).to(device)
    model = BERTClassifier(args.base_LM, num_labels=args.num_labels,cache_dir=args.temp_dir)
    #tokenizer = BertTokenizer.from_pretrained(args.base_LM, do_lower_case=True, cache_dir=args.temp_dir)
    tokenizer = BertTokenizer.from_pretrained(args.base_LM, cache_dir=args.temp_dir)

    # Test sentiment prediction
    test_text = "The movie was great and I really enjoyed the performances of the actors."
    ##classification = predict_section_class(test_text, model, tokenizer, device)
    model.eval()
    encoding = tokenizer(args.text, return_tensors='pt', max_length=args.max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    logger.info('TEXT: %i --- %s'%(preds.item(),label))
    label = SEC_CLS[preds.item()]
    return label
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str)
    parser.add_argument("-data_path", default='data_pubmed/data_pubmed_sec_cls', type=str)
    parser.add_argument("-model_path", default='', type=str)
    parser.add_argument("-base_LM", default='bert-base-uncased', type=str)
    parser.add_argument("-max_len", default=512, type=int)
    parser.add_argument("-epochs", default=3, type=int)
    parser.add_argument("-train_batch", default=16, type=int)
    parser.add_argument("-eval_batch", default=100, type=int)
    parser.add_argument("-logging_steps", default=1000, type=int)
    parser.add_argument("-warmup_steps", default=500, type=int)
    parser.add_argument("-weight_decay", default=0.01, type=int)
    parser.add_argument("-seed", default=1, type=int)
    parser.add_argument("-temp_dir", default='temp', type=str)
    parser.add_argument("-sn_dic_path", default='data_pubmed/data_pubmed_raw/SN_dic_8_Added.json', type=str)
    parser.add_argument("-num_labels", default=8, type=int)
    parser.add_argument("-classifier", default='linear', type=str)
    parser.add_argument("-log_file", default='', type=str)
    
    #transformer classifier
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)
    
    #test_text
    parser.add_argument("-test_from", default='', type=str)
    parser.add_argument("-text_path", default='', type=str)
    parser.add_argument("-text", default='', type=str)
        
    args = parser.parse_args()
    
    if args.classifier=='transformer':
        SAVE_MODEL_NAME = 'CLStrans_'+args.base_LM.split('-')[0]+args.base_LM.split('-')[1][0].upper()+'_ep'+str(args.epochs)+'tb'+str(args.train_batch)+'eb'+str(args.eval_batch)+'ws'+str(args.warmup_steps)+'ls'+str(args.logging_steps)+'wd'+str(args.weight_decay)
    else:    
        SAVE_MODEL_NAME = 'CLS_'+args.base_LM.split('-')[0]+args.base_LM.split('-')[1][0].upper()+'_ep'+str(args.epochs)+'tb'+str(args.train_batch)+'eb'+str(args.eval_batch)+'ws'+str(args.warmup_steps)+'ls'+str(args.logging_steps)+'wd'+str(args.weight_decay)
    if args.seed!=1:
        SAVE_MODEL_NAME  = SAVE_MODEL_NAME +'_seed'+str(args.seed)
        
    SAVE_MODEL_PATH = 'models/'+SAVE_MODEL_NAME     
    
    if args.model_path=='':
        args.model_path=SAVE_MODEL_PATH
    
    
    if (args.mode == 'train'):
        if args.log_file=='':
            args.log_file = args.model_path+'/train.log'
        train(args)
        os.mkdir(args.model_path+'/DONE')
    elif (args.mode == 'classify_text'):
        if args.log_file=='':
            args.log_file = args.text_path+'/test_text.log'
        classify_text(args)
        os.mkdir(args.text_path+'/DONE')
        

if __name__ == "__main__":
    main()
