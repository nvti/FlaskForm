import os
import warnings
from pytorch_transformers import (XLNetConfig, XLNetModel, XLNetTokenizer, XLNetPreTrainedModel)

import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm,trange, tqdm_notebook
import numpy as np
import math

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Config

xlnet_pretrain_path='../../xlnet_out_model'

max_len  = 64


# Model
class XLNetForPunctuationRestore(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetForPunctuationRestore, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, mems=None, perm_mask=None, target_mapping=None,
                token_type_ids=None, input_mask=None, head_mask=None, labels=None, label_weight=None):
        transformer_outputs = self.transformer(input_ids,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask, 
                                               head_mask=head_mask)
        output = transformer_outputs[0]

        logits = self.classifier(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:

            loss_fct = CrossEntropyLoss(ignore_index=-1, weight=label_weight)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, mems, (hidden states), (attentions)

class PunctuationRestore():
    SEG_ID_A   = 0
    SEG_ID_B   = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4

    def __init__(self,model_path=xlnet_pretrain_path):
        # Load self.tokenizer
        vocabulary = os.path.join(xlnet_pretrain_path, 'spiece.model')

        self.tokenizer = XLNetTokenizer(vocab_file=vocabulary,do_lower_case=True)

        # Init self.punctuation token
        self.punctuation = [".", ",", ":", "?"]
        self.label2idx = dict(zip(self.punctuation, range(0,len(self.punctuation))))
        self.idx2label = {key:self.punctuation[key] for key in self.label2idx.values()}
        self.idx2token = [self.tokenizer.encode(label)[-1] for label in self.punctuation]
        self.token2idx = dict(zip(self.idx2token, range(0,len(self.idx2token))))

        self.num_label = len(self.punctuation) + 1
        self.WORD_IDX = len(self.punctuation)
        self.IGNORE_IDX = -1

        self.SEG_ID_A   = 0
        self.SEG_ID_B   = 1
        self.SEG_ID_CLS = 2
        self.SEG_ID_SEP = 3
        self.SEG_ID_PAD = 4

        self.UNK_ID  = self.tokenizer.encode("<unk>")[0]
        self.CLS_ID  = self.tokenizer.encode("<cls>")[0]
        self.SEP_ID  = self.tokenizer.encode("<sep>")[0]
        self.MASK_ID = self.tokenizer.encode("<mask>")[0]
        self.EOD_ID  = self.tokenizer.encode("<eod>")[0]

        self.tokens_a = []

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XLNetForPunctuationRestore.from_pretrained(model_path,num_labels=self.num_label)
        self.model.to(self.device)

    def restore(self, text):
        # create token input from text input
        # input_ids, _, _, _ = sentence2input(text)
        input_ids_lower, input_mask, segment_ids, _ = self.sentence2input(text, do_lower_case=True)
        input_ids = input_ids_lower

        input_ids_tensor = torch.tensor(input_ids_lower)
        input_mask_tensor = torch.tensor(input_mask)
        segment_ids_tensor = torch.tensor(segment_ids)

        input_ids_tensor = input_ids_tensor.to(self.device)
        input_mask_tensor = input_mask_tensor.to(self.device)
        segment_ids_tensor = segment_ids_tensor.to(self.device)
        inputs = {'input_ids':input_ids_tensor, 
                  'token_type_ids':segment_ids_tensor, 
                  'input_mask':input_mask_tensor}

        outputs = self.model(**inputs)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        token_out = []
        outs = np.argmax(logits, axis=2)
        # print(outs)
        for id in range(len(outs)):
            output = outs[id]
            for i, segment in enumerate(segment_ids[id]):
                if segment == self.SEG_ID_A:
                    token_out.append(input_ids[id][i])
                    if output[i] != self.WORD_IDX:
                        token_out.append(self.idx2token[output[i]])
        
        # print(token_out)
        return self.tokenizer.decode(token_out)

    def token2input(self, token_in):
        tokens = []
        segment_ids = []
        label_ids = []
        
        for token in token_in:
            # if token != 17:
            if token in self.idx2token:
                while len(label_ids) > 0 and label_ids[-1] == 17:
                    label_ids.pop()
                    segment_ids.pop()
                if(len(label_ids) > 0):
                    label_ids.pop()
                    label_ids.append(self.token2idx[token]) 
            else:
                tokens.append(token)
                segment_ids.append(self.SEG_ID_A)
                label_ids.append(self.WORD_IDX)
        
        input_ids = tokens
        
        # The mask has 0 for real tokens and 1 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] * len(input_ids)

        # Zero-pad up to the sequence length at front
        if len(input_ids) < max_len:
            delta_len = max_len - len(input_ids)
            input_ids = [0] * delta_len + input_ids
            input_mask = [1] * delta_len + input_mask
            segment_ids = [self.SEG_ID_PAD] * delta_len + segment_ids
            label_ids = [self.IGNORE_IDX] * delta_len + label_ids

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len
        assert len(label_ids) == max_len

        return input_ids, input_mask, segment_ids, label_ids

    def sentence2input(self, sentence, reset=True, do_lower_case=False):

        input_ids = []
        input_masks = []
        segment_ids =[]
        label_ids = []

        # Tokenize sentence to token id list
        if do_lower_case:
            sentence = sentence.lower()
        sentence = self.tokenizer.clean_up_tokenization(sentence)
        tokens_tmp = self.tokenizer.encode(sentence)

        if reset:
            self.tokens_a = tokens_tmp
            # Trim the len of text
            # if len(self.tokens_a) > max_len-2:
            #     self.tokens_a = self.tokens_a[:max_len-2]
        else:
            self.tokens_a.extend(tokens_tmp)

        while len(self.tokens_a) >= max_len - 2:
            tmp_token = self.tokens_a[:max_len - 2]
            self.tokens_a = self.tokens_a[max_len - 2:]

            input_id, input_mask, segment_id, label_id = self.token2input(tmp_token)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            label_ids.append(label_id)

        if reset and len(self.tokens_a) > 0:
            input_id, input_mask, segment_id, label_id = self.token2input(self.tokens_a)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            label_ids.append(label_id)

            self.tokens_a = []

        return input_ids, input_masks, segment_ids, label_ids
