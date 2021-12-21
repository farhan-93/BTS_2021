import numpy as np
import torch.nn as nn
from transformers import  BertConfig, BertModel #BertTokenizer,
from transformers import get_linear_schedule_with_warmup
#from pytorch_transformers import  BertConfig #BertTokenizer,
#from pytorch_transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange
import sys, os

from BertModules import BertClassifier
from Constants import *
#from DataModules import SequenceDataset
from Utils import seed_everything
from transformers import BertTokenizer

seed_everything()



# Initialize BERT tokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load Train dataset and split it into Train and Validation dataset
'''
train_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer)

validation_split = 0.2
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
shuffle_dataset = True

if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)
'''



import json, re, nltk, string
from nltk.corpus import wordnet
nltk.download('punkt')
import pandas as pd
df=pd.read_csv('./../data/guo_firefox.csv',encoding = "latin-1")
#df.drop(['Unnamed: 4',	'Unnamed: 5'],axis=1,inplace=True)
#df.dropna(inplace=True)
df.head()
selected = ['assigned_to', 'description']
#selected = ['assigned_to', 'description', 'summary']
non_selected = list(set(df.columns) - set(selected))

df = df.drop(non_selected, axis=1) # Drop non selected columns
df = df.dropna(axis=0, how='any', subset=selected) # Drop null rows
df = df.reindex(np.random.permutation(df.index))

#df['text']=df['description']+' '+df['issue_title']
df['text']=df['description']
df.head()

df.dropna(subset=['text'],inplace=True)


all_data = []
all_owner = []
for row in range(len(df)):
    item=df.iloc[row,:]
    #1. Remove \r
    text = item['text'].replace('\r', ' ')
    #2. Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    #3. Remove Stack Trace
    start_loc = text.find("Stack trace:")
    text = text[:start_loc]
    #4. Remove hex code
    text = re.sub(r'(\w+)0x\w+', '', text)
    #5. Change to lower case
    text = text.lower()
    #6. Tokenize
    text = nltk.word_tokenize(text)
    #7. Strip punctuation marks
    text = [word.strip(string.punctuation) for word in text]
    #8. Join the lists
    all_data.append(text)
    all_owner.append(item['assigned_to'])

all_data=[' '.join([j for j in i if len(j)>1]) for i in all_data]

df=pd.DataFrame(list((all_data,all_owner)),index=['description','assigned_to']).T
df.head()

classes = df.assigned_to.unique()#np.array(list(set(train_labels)))
print(len(classes))


from sklearn.model_selection import train_test_split
#train,test=train_test_split(df,test_size=0.1)
split=int(len(df)*0.8)
train= df[:split]
test = df[split:]

train.reset_index(drop=True,inplace=True)
test.reset_index(drop=True,inplace=True)

train.to_csv('./../data/train.csv',index=False)
test.to_csv('./../data/test.csv',index=False)


df = pd.read_csv("./../data/train.csv")
df["sentences"] = df["description"].replace(np.nan, 'none', regex=True)
df["labels"] = df["assigned_to"].replace(np.nan, 'none', regex=True)
# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))


sentences = df.sentences.values
labels = df.labels.values


print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

input_ids = []

# For every sentence...
for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

        # This function also supports truncation and conversion
        # to pytorch tensors, but we need to do padding, so we
        # can't use these features :( .
        max_length=128,  # Truncate all sentences.
        truncation=True,
        # return_tensors = 'pt',     # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)
print('Max sentence length: ', max([len(sen) for sen in input_ids]))

from keras.preprocessing.sequence import pad_sequences

MAX_LEN = 128

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")
#segment_ids = [0] * len(input_ids)


print('\nDone.')

'''
segment_ids=[]
# For each sentence...
for sent in input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    segment_id = [int(token_id <= 0) for token_id in sent]

    # Store the Segment mask for this sentence.
    segment_ids.append(segment_id)
    '''
attention_masks = []
for sent in input_ids:
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]

    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

from sklearn.model_selection import train_test_split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                    random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                       random_state=2018, test_size=0.1)
#train_segment_ids, validation_segment_ids, _, _ = train_test_split(segment_ids, labels,
                                                       #random_state=2018, test_size=0.1)

main_labels = train_labels.copy()

validation_inputs = list(validation_inputs)
#validation_segment_ids = list(validation_segment_ids)
validation_masks = list(validation_masks)
validation_labels = list(validation_labels)


# remove all classes which are in train but not in test

for i in range(2):  # idk why its not removing some labels in first attempt
    for i, j in enumerate(validation_labels):
        if j not in train_labels:
            validation_inputs.pop(i)
            validation_masks.pop(i)
            validation_labels.pop(i)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(train_labels)  # convert email into integer form
y_val = le.transform(validation_labels)  # convert email into integer form

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)


train_labels = torch.tensor(y_train)
validation_labels = torch.tensor(y_val)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#train_segment_ids = torch.tensor(train_segment_ids)
#validation_segment_ids = torch.tensor(validation_segment_ids)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
val_loader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)







# Load BERT default config object and make necessary changes as per requirement
config = BertConfig(hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    num_labels=len(classes))

# Create our custom BERTClassifier model object
model = BertClassifier(config)
model = nn.DataParallel(model)
#model= BertModel.from_pretrained('bert-base-uncased',config=config)
model.to(DEVICE)
print(model)





print ('Training Set Size {}, Validation Set Size {}'.format(len(train_inputs), len(validation_inputs)))

# Loss Function
criterion = nn.CrossEntropyLoss()

# Adam Optimizer with very small learning rate given to BERT
optimizer = torch.optim.AdamW([
    {'params': model.module.bert.parameters(), 'lr': 5e-5},#5e-5
    {'params': model.module.classifier.parameters(), 'lr': 1e-5, 'eps':1e-8}#1e-5
])
#optimizer = torch.optim.AdamW(model.parameters(),
                  #lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  #eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  #)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=5,  # Default value in run_glue.py
                                            num_training_steps=total_steps)
# Learning rate scheduler
#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS,
                                 #t_total=len(train_loader) // GRADIENT_ACCUMULATION_STEPS * NUM_EPOCHS)

model.zero_grad()
#epoch_iterator = trange(int(NUM_EPOCHS), desc="Epoch")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
import time
import datetime
import random
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_acc_list, validation_acc_list = [], []

for epoch_i in range(0, NUM_EPOCHS):
    total_loss = 0.0
    train_correct_total = 0
    model.train()
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, NUM_EPOCHS))
    print('Training...')
    t0 = time.time()

    # Training Loop
    #train_iterator = tqdm(train_loader, desc="Train Iteration")
    for step, batch in enumerate(train_loader):
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))
         #(True)
        # Here each element of batch list refers to one of [input_ids, segment_ids, attention_mask, labels]
        model.zero_grad()
        inputs = {
            'input_ids': batch[0].to(DEVICE),
            'token_type_ids': None,#batch[1].to(DEVICE),
            'attention_mask': batch[1].to(DEVICE)
        }

        labels = batch[2].to(DEVICE)
        logits = model(**inputs)

        loss = criterion(logits, labels) #criterion(logits, labels) #/ GRADIENT_ACCUMULATION_STEPS
        loss.backward()
        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:

            #optimizer.step()
            #scheduler.step()
            #model.zero_grad()
        optimizer.step()
        scheduler.step(loss.cpu().data.numpy())
    avg_train_loss = total_loss / len(train_loader)
    print("")
    print("  Average training loss: {0:.4f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        #_, predicted = torch.max(logits.data, 1)
        #correct_reviews_in_batch = (predicted == labels).sum().item()
        #train_correct_total += correct_reviews_in_batch

        #break

    #print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_loader)))
    print("")
    print("Running Validation...")
    t0 = time.time()
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in val_loader:
        batch = tuple(t.to(DEVICE) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0].to(DEVICE),
                'token_type_ids': None, #batch[1].to(DEVICE),
                'attention_mask': batch[1].to(DEVICE)
            }

            labels = batch[2].to(DEVICE)
            logits = model(**inputs)
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("  Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")


model_to_save = model.module if hasattr(model, 'module') else model 
# model_to_save.config.architectures = [model_to_save.__class__.__name__]

# Save the config

#model_to_save.config.save_pretrained('./model/')

        # Save the model

state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
_keys_to_ignore_on_save=None
if _keys_to_ignore_on_save is not None:
    state_dict = {k: v for k, v in state_dict.items() if k not in _keys_to_ignore_on_save}

        # If we save using the predefined names, we can load using `from_pretrained`
#output_model_file = os.path.join(save_directory, 'pytorch_model.bin')
torch.save(state_dict, './../base_model/pytorch_model.bin')
#print(state_dict)


models_save_dir = './../base_model/base_model'

print('Saving model in '+models_save_dir+'.pt'+'...')
state ={
    'state_dict' : model.state_dict(),
    'optimizer'  : optimizer.state_dict(),
}
torch.save(state,models_save_dir+'.pt')





df = pd.read_csv("./../data/test.csv")
print(df)
df["description"] = df["description"].replace(np.nan, 'none', regex=True)
df["assigned_to"] = df["assigned_to"].replace(np.nan, 'none', regex=True)

#df.dropna(subset=['sentences'],inplace=True)
# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))
'''
import json, re, nltk, string
from nltk.corpus import wordnet

nltk.download('punkt')

all_data = []
all_owner = []
for row in range(len(df)):
    item = df.iloc[row, :]
    # 1. Remove \r
    text = item['sentences'].replace('\r', ' ')
    # 2. Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 3. Remove Stack Trace
    start_loc = text.find("Stack trace:")
    text = text[:start_loc]
    # 4. Remove hex code
    text = re.sub(r'(\w+)0x\w+', '', text)
    # 5. Change to lower case
    text = text.lower()
    # 6. Tokenize
    text = nltk.word_tokenize(text)
    # 7. Strip punctuation marks
    text = [word.strip(string.punctuation) for word in text]
    # 8. Join the lists
    all_data.append(text)
    all_owner.append(item['labels'])

all_data = [' '.join([j for j in i if len(j) > 1]) for i in all_data]
'''
# Create sentence and label lists
sentences = df.description.values
labels = df.assigned_to.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for sent in sentences:
    # print(sent)
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(sent, add_special_tokens=True, max_length =MAX_LEN,truncation=True)
    # sent,                      # Sentence to encode.
    # add_special_tokens = True, # Add '[CLS]' and '[SEP]'
    # max_length = 512,
    # truncation=True
    # )

    input_ids.append(encoded_sent)

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                          dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

input_ids = list(input_ids)
attention_masks = list(attention_masks)
labels = list(labels)

# remove all classes which are in train but not in test
for i in range(2):  # idk why its not removing some labels in first attempt
    for i, j in enumerate(labels):
        if j not in main_labels:
            input_ids.pop(i)
            attention_masks.pop(i)
            labels.pop(i)

y_test = le.transform(labels)

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(y_test)

# Set the batch size.
# batch_size = batch_size

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

#y_test=y_val
"""## 5.2. Evaluate on Test Set

With the test set prepared, we can apply our fine-tuned model to generate predictions on the test set.
"""

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions, true_labels = [], []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
# Predict
for batch in prediction_dataloader: #prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(DEVICE) for t in batch)

    # Unpack the inputs from our dataloader
    #b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        inputs = {
            'input_ids': batch[0].to(DEVICE),
            'token_type_ids': None,  # batch[1].to(DEVICE),
            'attention_mask': batch[1].to(DEVICE)
        }

        labels = batch[2].to(DEVICE)
        logits = model(**inputs)
        # Forward pass, calculate logit predictions
        #outputs = model(b_input_ids, token_type_ids=None,
                        #attention_mask=b_input_mask)

    #logits = outputs

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = labels.to('cpu').numpy()

    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

#print(predictions[0])
prediction = [item for sublist in predictions for item in sublist]
#print(prediction[0])

# print(predictions)

pred = np.argmax(np.array(prediction), axis=1)
#pred[2]

#label_ids
prediction=np.array(prediction)
"""# Conclusion"""

from sklearn.metrics import accuracy_score
'''
print(y_test.shape)
print(y_test)
corr= np.delete(prediction,np.s_[1:159],axis=1)
print(corr.shape)
print(corr)
print(prediction.shape)
print(pred.shape)
print(pred)
print(np.asarray(predictions).shape)
'''
print("  Accuracy: {0:.4f}".format(accuracy_score(y_test, pred)))
#print("  Accuracy: {0:.4f}".format(eval_accuracy / (nb_eval_steps)))

print('    DONE.')
#pred[1]
# y_test

#print(len(y_test))
#print(len(pred))
#print (y_test.shape)
#print (pred.shape)

#from keras.metrics import top_k_categorical_accuracy

from sklearn.metrics import accuracy_score

best_1 = np.argsort(prediction)[:, -1:]
top1 = 0
for i in range(0, best_1.shape[0]):
    # print(best.shape)
    if y_test[i] in best_1[i]:
        # print(y_true[i])
        top1 = top1 + 1
print("top1 Accuracy:  ", top1 / best_1.shape[0])



best_2 = np.argsort(prediction)[:, -2:]
top2 = 0
for i in range(0, best_2.shape[0]):
    # print(best.shape)
    if y_test[i] in best_2[i]:
        # print(y_true[i])
        top2 = top2 + 1
print("top2 Accuracy:  ", top2 / best_2.shape[0])

best_3 = np.argsort(prediction)[:, -3:]
top3 = 0
for i in range(0, best_3.shape[0]):
    # print(best.shape)
    if y_test[i] in best_3[i]:
        # print(y_true[i])
        top3 = top3 + 1
print("top3 Accuracy:  ", top3 / best_3.shape[0])

best_4 = np.argsort(prediction)[:, -4:]
top4 = 0
for i in range(0, best_4.shape[0]):
    # print(best.shape)
    if y_test[i] in best_4[i]:
        # print(y_true[i])
        top4 = top4 + 1
print("top4 Accuracy:  ", top4 / best_4.shape[0])


best_5 = np.argsort(prediction)[:, -5:]
top5 = 0
for i in range(0, best_5.shape[0]):
    # print(best.shape)
    if y_test[i] in best_5[i]:
        # print(y_true[i])
        top5 = top5 + 1
print("top5 Accuracy:  ", top5 / best_5.shape[0])


best_6 = np.argsort(prediction)[:, -6:]
top6 = 0
for i in range(0, best_6.shape[0]):
    # print(best.shape)
    if y_test[i] in best_6[i]:
        # print(y_true[i])
        top6 = top6 + 1
print("top6 Accuracy:  ", top6 / best_6.shape[0])

best_7 = np.argsort(prediction)[:, -7:]
top7 = 0
for i in range(0, best_7.shape[0]):
    # print(best.shape)
    if y_test[i] in best_7[i]:
        # print(y_true[i])
        top7 = top7 + 1
print("top7 Accuracy:  ", top7 / best_7.shape[0])


best_8 = np.argsort(prediction)[:, -8:]
top8 = 0
for i in range(0, best_8.shape[0]):
    # print(best.shape)
    if y_test[i] in best_8[i]:
        # print(y_true[i])
        top8 = top8 + 1
print("top8 Accuracy:  ", top8 / best_8.shape[0])

best_9 = np.argsort(prediction)[:, -9:]
top9 = 0
for i in range(0, best_9.shape[0]):
    # print(best.shape)
    if y_test[i] in best_9[i]:
        # print(y_true[i])
        top9 = top9 + 1
print("top9 Accuracy:  ", top9 / best_9.shape[0])



best_10 = np.argsort(prediction)[:, -10:]
top10 = 0
for i in range(0, best_10.shape[0]):
    # print(best.shape)
    if y_test[i] in best_10[i]:
        # print(y_true[i])
        top10 = top10 + 1
print("top10 Accuracy:  ", top10 / best_10.shape[0])








'''
output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

'''

            #_, predicted = torch.max(logits.data, 1)
            #correct_reviews_in_batch = (predicted == labels).sum().item()
            #val_correct_total += correct_reviews_in_batch
            #break
        #training_acc_list.append(train_correct_total * 100 / len(train_loader))
        #validation_acc_list.append(val_correct_total * 100 / len(val_loader))
        #print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
            #train_correct_total * 100 / len(train_loader), val_correct_total * 100 / len(val_loader)))


# text = 'I am a big fan of cricket'
# text = '[CLS] ' + text + ' [SEP]'
#
# encoded_text = tokenizer.encode(text) + [0] * 120
# tokens_tensor = torch.tensor([encoded_text])
# labels = torch.tensor([1])
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam([
#                 {'params': model.bert.parameters(), 'lr' : 1e-5},
#                 {'params': model.classifier.parameters(), 'lr': 1e-3}
#             ])


# logits = model(tokens_tensor, labels=labels)
# loss = criterion(logits, labels)
# print(loss)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

