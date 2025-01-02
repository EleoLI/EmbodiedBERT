# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:27:40 2024

@author: 45078
"""

import numpy as np
import torch
import torch.nn as nn


import re

import numpy as np
from tqdm import tqdm
from nltk.corpus import wordnet as wn

from scipy.stats import pearsonr, spearmanr, truncnorm
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
import random
import nltk


from utils import Config
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class MappingModel(nn.Module):
    def __init__(self, dropout_rate):
        super(MappingModel, self).__init__()      
        self.fc1 = nn.Linear(768, 384)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(384, 192)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(192, 1)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear,)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(0.0)       
            
    def forward(self, inputs):
        hidden = self.fc1(inputs)
        hidden = self.dropout1(hidden)
        hidden = self.relu1(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.relu2(hidden)
        output = self.fc3(hidden)
        return output
    
class EmbodiedBERT_MIP_2(nn.Module):

    def __init__(self, args, Model, config, regressors,num_labels=2):
        """Initialize the model"""
        super(EmbodiedBERT_MIP_2, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args
        self.regressors = nn.ModuleList(regressors)
        for regressor in self.regressors:
            for param in regressor.parameters():
                param.requires_grad = False
        

        self.SPV_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)       
        self.MIP_SM_linear = nn.Linear((config.hidden_size+len(self.regressors)) * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_SM_linear)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the second input token indicies
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1] # [batch, hidden]
        

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)    
        
        target_output_without_dropout = target_output.mean(1)# [batch, hidden]

        
        # Get contextualized sensorimotor embeddings
        context_sm = [regressor(target_output_without_dropout) for regressor in self.regressors]
        context_sm = torch.cat(context_sm, dim=1)    
        # padding = (0, self.args.classifier_hidden - len(self.regressors))
        # context_sm = F.pad(context_sm, pad=padding, mode='constant', value=0)
        
        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)


        target_output = target_output.mean(1)# [batch, hidden]

 
        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]
        

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        
        target_output_2_without_dropput = target_output_2.mean(1)
        
        # Get basic sensorimotor embedding
        basic_sm = [regressor(target_output_2_without_dropput) for regressor in self.regressors]        
        basic_sm = torch.cat(basic_sm, dim=1)       

        
        target_output_2 = self.dropout(target_output_2)       
        
        target_output_2 = target_output_2.mean(1)
                  
        
        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([target_output,pooled_output], dim=1))
        
        basic = torch.cat([target_output_2, basic_sm],dim=1)
        context = torch.cat([target_output,context_sm],dim=1)
           
        MIP_SM = self.MIP_SM_linear(torch.cat([basic, context], dim=1))
                
        logits = self.classifier(self.dropout(torch.cat([SPV_hidden,
                                                         MIP_SM], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        
        # if not self.training:
        #     return logits, context_sm, basic_sm
        return logits, context_sm, basic_sm
    
class Sensorimotor_predictor(torch.nn.Module):
    def __init__(self, model):
        super(Sensorimotor_predictor, self).__init__()
        self.model = model
        self.auditory = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.gustatory = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.haptic = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.interoceptive = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.olfactory = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.visual = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.foot_leg = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.hand_arm = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.head = torch.nn.Linear(self.model.config.hidden_size,1)
        self.mouth = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.torso = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        aud = self.auditory(pooled_output)
        gus = self.gustatory(pooled_output)
        hap = self.haptic(pooled_output)
        intero = self.interoceptive(pooled_output)
        olf = self.olfactory(pooled_output)
        vis = self.visual(pooled_output)
        fl = self.foot_leg(pooled_output)
        ha = self.hand_arm(pooled_output)
        head = self.head(pooled_output)
        mou = self.mouth(pooled_output)
        tor = self.torso(pooled_output)
        
        return aud,gus,hap,intero,olf,vis,fl,ha,head,mou,tor