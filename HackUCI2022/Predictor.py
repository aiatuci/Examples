import torch
import spacy
from RNN import *
import os

class Predictor:
    def __init__(self):
        self.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.net = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
        self.net.load_state_dict(torch.load(os.getcwd() + "\HackUCI2022\model\model.pt", map_location=self.device))
        self.net.eval()
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, text):
        with torch.no_grad():
            tokenized = [tok.text for tok in self.nlp.tokenizer(text)]  #tokenize the sentence 
            indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
            length = [len(indexed)]                                    #compute no. of words
            tensor = torch.LongTensor(indexed).to(self.device)              #convert to tensor
            tensor = tensor.unsqueeze(0).T                             #reshape in form of batch,no. of words
            length_tensor = torch.LongTensor(length)                   #convert to tensor
            prediction = self.net(tensor, length_tensor)                  #prediction
        isPositive = bool(int(torch.round(torch.sigmoid(torch.tensor(prediction.item())))))
        return isPositive
    