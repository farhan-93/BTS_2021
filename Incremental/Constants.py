import torch

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
TRAIN_FILE_PATH = '/Users/muhammadabdullah/Downloads/sarcasmv2/Sarcasm_Headlines_Dataset_v2.json'
base_class = 100
BATCH_SIZE = 32
NUM_EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)