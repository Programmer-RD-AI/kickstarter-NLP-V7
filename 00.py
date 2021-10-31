from sklearn.model_selection import *
import wandb
import nltk
from nltk.stem.porter import *
from torch.nn import *
from torch.optim import *
import numpy as np
import pandas as pd
import torch
import torchvision
import random
from tqdm import *
from torch.utils.data import Dataset, DataLoader
stemmer = PorterStemmer()
PROJECT_NAME = 'kickstarter-NLP-V7'
device = 'cuda'


def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())


print(tokenize('$100'))


def stem(word):
    return stemmer.stem(word.lower())


print(stem('organic'))


def bag_of_words(t_words, words):
    t_words = [stem(w) for w in t_words]
    bag = np.zeros(len(words))
    for idx, w in enumerate(words):
        if w in t_words:
            bag[idx] = 1.0
    return bag


print(bag_of_words(['hi'], ['hi', 'how', 'hi']))
data = pd.read_csv('./data.csv')[:1250]
print(data.columns)
X = data['blurb']
y = data['state']
words = []
labels = {}
labels_r = {}
idx = 0
data = []
for label in tqdm(list(y.tolist())):
    if label not in list(labels.keys()):
        idx += 1
        labels[label] = idx
        labels_r[idx] = label

for X_batch, y_batch in zip(tqdm(X), y):
    X_batch = tokenize(X_batch)
    new_X = []
    for Xb in X_batch:
        new_X.append(stem(Xb))
    words.extend(new_X)
    data.append([
        new_X,
        np.eye(labels[y_batch], len(labels))[labels[y_batch]-1]
    ])
words = sorted(set(words))
np.random.shuffle(data)

X = []
y = []
for d in tqdm(data):
    X.append(bag_of_words(d[0], words))
    y.append(d[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.125, shuffle=False)
X_train = torch.from_numpy(np.array(X_train)).to(device).float()
y_train = torch.from_numpy(np.array(y_train)).to(device).float()
X_test = torch.from_numpy(np.array(X_test)).to(device).float()
y_test = torch.from_numpy(np.array(y_test)).to(device).float()


def get_loss(model, X, y, criterion):
    preds = model(X)
    loss = criterion(preds, y)
    return loss.item()


def get_accuracy(model, X, y,):
    preds = model(X)
    correct = 0
    total = 0
    for pred, yb in zip(preds, y):
        pred = int(torch.argmax(pred))
        yb = int(torch.argmax(yb))
        if pred == yb:
            correct += 1
        total += 1
    acc = round(correct/total, 3)*100
    return acc


class Model(Module):
    def __init__(self):
        super().__init__()
        self.hidden = 8
        self.activation = ReLU()
        self.bn = BatchNorm1d(self.hidden)
        self.linear1 = Linear(len(words), self.hidden)
        self.linear2 = Linear(self.hidden, self.hidden)
        self.linear3 = Linear(self.hidden, len(labels))

    def forward(self, X):
        preds = self.linear1(X)
        preds = self.activation(self.bn(self.linear2(preds)))
        preds = self.linear3(preds)
        return preds


model = Model().to(device)
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 32
wandb.init(project=PROJECT_NAME, name='baseline')
for _ in tqdm(range(epochs)):
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    torch.cuda.empty_cache()
    wandb.log({'Loss':(get_loss(model,X_train,y_train,criterion)+get_loss(model,X_batch,y_batch,criterion)/2)})
    torch.cuda.empty_cache()
    wandb.log({'Val Loss':get_loss(model,X_test,y_test,criterion)})
    torch.cuda.empty_cache()
    wandb.log({'Acc':(get_accuracy(model,X_train,y_train)+get_accuracy(model,X_batch,y_batch))/2})
    torch.cuda.empty_cache()
    wandb.log({'Val Acc':get_accuracy(model,X_test,y_test)})
    torch.cuda.empty_cache()
    model.train()
wandb.finish()
torch.cuda.empty_cache()
torch.save(model,'model.pt')
torch.save(model,'model.pth')
torch.save(model.state_dict(),'model-sd.pt')
torch.save(model.state_dict(),'model-sd.pth')
torch.save(words,'words.pt')
torch.save(words,'words.pth')
torch.save(data,'data.pt')
torch.save(data,'data.pth')
torch.save(labels,'labels.pt')
torch.save(labels,'labels.pth')
torch.save(idx,'idx.pt')
torch.save(idx,'idx.pth')
torch.save(y_train,'y_train.pt')
torch.save(y_test,'y_test.pth')
torch.save(y,'y.pt')
torch.save(y,'y.pth')
