import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim
from RNN import RNN

# TEXT and LABEL returning errors due to difference in Python (VSCode) and Google Collab
TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.35
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

class Predictor:
    def __init__(self):
        self.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.net = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
        self.net.load_state_dict(torch.load('./models/model.pt', map_location=self.device))
        self.net.eval()

    def predict(self):
        return 1

if __name__ == "__main__":
    model = Predictor()
    print(model.predict())