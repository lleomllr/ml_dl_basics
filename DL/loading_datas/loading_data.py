from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = '/content/gdrive/My Drive/Datasets/PetImages/'

CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()

        break
    break

print(img_array.shape)

IMG_SIZE = 40
#redimensionne img_array
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []

def create_training_data():
  for category in CATEGORIES:
      path = os.path.join(DATADIR,category)
      class_num = CATEGORIES.index(category)
      for img in os.listdir(path):
        try:
          img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
          new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
          training_data.append([new_array, class_num])
        except Exception as e:
          pass

create_training_data()

#nb d'elem = nb d'images total
print(len(training_data))

import random
#Mélange les données pour s'assurer que les données d'entraînement ne sont pas dans un ordre spécifique,
#Aide à éviter que le modèle ne s'entraîne sur des données biaisées.
random.shuffle(training_data)

for sample in training_data[:10]:
  print(sample[1])

X = []
y = []

#boucle décomposant chaque elem en image redimensionnée et en classe correspondante
for features, label in training_data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#permet de sérialiser (convertir en bytes) pour sauv dans des fichiers
import pickle

#ouvre fichier en mode écriture binaire et sérialise l'objet 'X'
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#ouvre fichier en mode lecture binaire et désérialise son contenu
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

#affiche 2nd elem du tableau 'X'
X[1]
