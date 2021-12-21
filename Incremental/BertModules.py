import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_transformers import BertModel
from transformers import BertModel



class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.num_labels = config.num_labels
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained('./../model/pytorch_model.bin', config=config)
        #model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        #model_to_save= self.bert
        #model_to_save = model_to_save.save_pretrained('./model/')
        #tokenizer.save_pretrained(output_dir)
        # Dropout to avoid overfitting
        #self.tanh = nn.Tanh(config.hidden_size)
        self.dropout = nn.Dropout(0.3) #config.hidden_dropout_prob
        # A single layer classifier added on top of BERT to fine tune for binary classification
        #self.hidden = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Weight initialization
        #torch.nn.init.xavier_normal_(self.hidden.weight)
        torch.nn.init.xavier_normal_(self.classifier.weight)

        #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
        #@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC,   checkpoint="bert-base-uncased",  output_type=SequenceClassifierOutput,  config_class=_CONFIG_FOR_DOC   )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        # Forward pass through pre-trained BERT
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # Last layer output (Total 12 layers)
        pooled_output = outputs[-1] #-1

        pooled_output = self.dropout(pooled_output)
        
        #hidden = self.dropout(torch.tanh(self.hidden(pooled_output)))

        #
        return self.classifier(pooled_output)
