import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import torch.nn as nn
import torch.optim as optim

TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 64

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths): 
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):  
    loss1 = 0
    acc1 = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        loss1 += loss.item()
        acc1 += acc.item()
    return loss1 / len(iterator), acc1 / len(iterator)

def evaluate(model, iterator, criterion):
    loss1 = 0
    acc1 = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text       
            predictions = model(text, text_lengths).squeeze(1)     
            loss = criterion(predictions, batch.label)    
            acc = binary_accuracy(predictions, batch.label)
            loss1 += loss.item()
            acc1 += acc.item()
    return loss1 / len(iterator), acc1 / len(iterator)


train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(train_data, 
                max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.35
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    device = device)

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)

def train_model():
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    loss, acc = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {loss:.3f}')
    print(f'Test Acc: {acc:.3f}')