import nltk
import csv 
from nltk.corpus import brown 
from nltk.corpus import wordnet

import numpy as np
import torch 
import multiprocessing
from torch import nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.utils.data import DataLoader
import time 

nltk.download('brown')
nltk.download('wordnet')

num_train = 12000
UNK_symbol = "<UNK>"
vocab = set([UNK_symbol])

#création de brown corpus avec tous les mots 
#pas de préprocess, juste les minuscules 
brown_corpus_train = []
for idx, paragraph in enumerate(brown.paras()):
  if idx == num_train:
    break 
  words = []
  for sentence in paragraph:
    for word in sentence:
      words.append(word.lower())
  brown_corpus_train.append(words)

#création d'une fréquence de termes des mots 
words_term_frequency = {}
for doc in brown_corpus_train:
  for word in doc:
    #calcul de la fréquence des termes 
    words_term_frequency[word] = words_term_frequency.get(word, 0) + 1

#création du vocab
for doc in brown_corpus_train:
  for word in doc:
    if words_term_frequency.get(word, 0) >= 5:
      vocab.add(word)

X_train = []
y_train = []
X_dev = []
y_dev = []

#création d'une correspondance entre les mots et id 
mapping = {}
for idx, word in enumerate(vocab):
  mapping[word] = idx

#fonction pour recup l'id d'un mot donné
#retourne <UNK> id si pas trouvé 
def get_id_of_word(word):
  unknown_word_id = mapping['<UNK>']
  return mapping.get(word, unknown_word_id)

#création des ensembles d'entraînement et de dev 
for idx, paragraph in enumerate(brown.paras()):
  for sentence in paragraph:
    for i, word in enumerate(sentence):
      if i+2 >= len(sentence):
        #limite de la phrase atteinte 
        #ignorer une phrase de moins de 3 mots 
        break 
      #conversion du mot vers l'id 
      X_extract = [get_id_of_word(word.lower()), get_id_of_word(sentence[i+1].lower())]
      y_extract = [get_id_of_word(sentence[i+2].lower())]
      if idx < num_train:
        X_train.append(X_extract)
        y_train.append(y_extract)
      else:
        X_dev.append(X_extract)
        y_dev.append(y_extract)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_dev = np.array(X_dev)
y_dev = np.array(y_dev)


class TrigramNNModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_size, h):
    super(TrigramNNModel, self).__init__()
    self.context_size = context_size 
    self.embedding_dim = embedding_dim
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.linear1 = nn.Linear(context_size * embedding_dim, h)
    self.linear2 = nn.Linear(h, vocab_size, bias = False)

  def forward(self, inputs):
    #calcul de x' : concatenation des embeddings x1 et x2
    embeds = self.embeddings(inputs).view(-1, self.context_size * self.embedding_dim)
    #calcul de h : tanh(W1.x' + b)
    out = torch.tanh(self.linear1(embeds))
    #calcul de W2.h
    out = self.linear2(out)
    #calcul de y : log_softmax(W2.h)
    log_probs = F.log_softmax(out, dim=1)
    #retourne probabilités logarithmiques 
    #BATCH_SIZE x len(vocab)
    return log_probs
  
#création des paramètres 
gpu = 0

#taille des vecteurs de mots 
EMBEDDING_DIM = 200
CONTEXT_SIZE = 2 
BATCH_SIZE = 256 

#unités cachées 
H = 100
torch.manual_seed(13013)

#check si gpu est disponible
# If CUDA is not available, default to CPU. This line was changed to fix the error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
available_workers = multiprocessing.cpu_count()

print("Creating training and dev dataloaders with {} batch size".format(BATCH_SIZE))
train_set = np.concatenate((X_train, y_train), axis = 1)
dev_set = np.concatenate((X_dev, y_dev), axis = 1)
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = available_workers)
dev_loader = DataLoader(dev_set, batch_size = BATCH_SIZE, num_workers = available_workers)

#obtenir la précision des probabilités logarithmiques 
def get_accuracy_log(log_probs, labels):
  probs = torch.exp(log_probs)
  predicted_label = torch.argmax(probs, dim = 1)
  acc = (predicted_label == labels).float().mean()
  return acc

#évaluation du modèle sur les données de dev 
def evaluate(model, criterion, dataloader, gpu):
  model.eval()
  mean_acc, mean_loss = 0, 0
  count = 0

  with torch.no_grad():
    dev_st = time.time()
    for it, data_tensor in enumerate(dataloader):
      context_tensor = data_tensor[:,0:2]
      target_tensor = data_tensor[:,2]
      context_tensor, target_tensor = context_tensor.to(device), target_tensor.to(device)
      log_probs = model(context_tensor)
      mean_loss += criterion(log_probs, target_tensor).item()
      mean_acc += get_accuracy_log(log_probs, target_tensor)
      count += 1
      if it % 500 == 0:
        print("Dev iteration {} complete. Mean Loss : {}; Mean Acc : {}; Time taken (s) : {}".format(it, mean_loss / count, mean_acc / count, (time.time() - dev_st)))
        dev_st = time.time()
  return mean_acc / count, mean_loss / count

#utilisation de la perte log-likelihood négative 
loss_function = nn.NLLLoss()

#création du modèle 
model = TrigramNNModel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, H)

#chargement du modèle sur le gpu 
model.to(device)

#utilisation de l'optimiser ADAM
optimizer = optim.Adam(model.parameters(), lr = 2e-3)


#-----------------ENTRAINEMENT DU MODELE----------------
best_acc = 0
best_model_path = None 

for epoch in range(5):
  st = time.time()
  print("\n Training model Epoch : {}".format(epoch+1))
  for it, data_tensor in enumerate(train_loader):
    context_tensor = data_tensor[:,0:2]
    target_tensor = data_tensor[:,2]

    context_tensor, target_tensor = context_tensor.to(device), target_tensor.to(device)

    #annulation des gradients de l'ancienne distance 
    model.zero_grad()

    #obtention des proba log sur les mots suivants 
    log_probs = model(context_tensor)

    #calcul de la précision actuelle 
    acc = get_accuracy_log(log_probs, target_tensor)

    #calcul de la loss function 
    loss = loss_function(log_probs, target_tensor)

    #passage en arrière et maj du gradient
    loss.backward()
    optimizer.step()

    if it % 500 == 0:
      print("Iteration {} complete. Loss : {}; Accuracy : {}; Time taken (s) : {}".format(it, loss.item(), acc, (time.time() - st)))
      st = time.time()

  print("\n Evaluating model on dev data")
  dev_acc, dev_loss = evaluate(model, loss_function, dev_loader, gpu)
  print("Epoch {} complete! Dev accuracy : {}; Dev Loss : {}".format(epoch, dev_acc, dev_loss))
  if dev_acc > best_acc:
    print("Best dev accuracy improved from {} to {}, saving model.".format(best_acc, dev_acc))
    best_acc = dev_acc 
    #définition du chemin du meilleur modèle 
    best_model_path = 'best_model_{}.dat'.format(epoch)
    #sauvegarde du meilleur modèle 
    torch.save(model.state_dict(), best_model_path)


best_model = TrigramNNModel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, H)
best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
best_model.to(device)

cos = nn.CosineSimilarity(dim=0)

lm_simili = {}

#paires de mots pour calculer la similarité 
words = {('computer', 'keyboard'), ('cat', 'dog'), ('dog', 'car'), ('keyboard', 'cat')}

#calcul similarités LM en utilisant la similarité cosinus
for word_pairs in words:
  w1 = word_pairs[0]
  w2 = word_pairs[1]
  words_tensor = torch.LongTensor([get_id_of_word(w1), get_id_of_word(w2)])
  words_tensor = words_tensor.to(device)
  #obtention du mot à partir du meilleur modèle 
  words_embeds = best_model.embeddings(words_tensor)
  #calcul similarité cosinus entre vecteurs de mots 
  sim = cos(words_embeds[0], words_embeds[1])
  lm_simili[word_pairs] = sim.item()
print(lm_simili)

    