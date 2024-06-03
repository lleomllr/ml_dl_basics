import tensorflow as tf

mnist = tf.keras.datasets.mnist

#download et divise les données en données d'entraitement et de test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalisation des images en 0 à 1. axis=1 => Pour chaque image individuellement
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#couches du modèle empilées les unes après les autres
model = tf.keras.models.Sequential()
#couche Flatten qui transforme les 2D en vecteur 1D => préparer les données pour les couches denses suivantes
model.add(tf.keras.layers.Flatten())
#ajout couche dense avec 128 neurones et fonction activation ReLu
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#ajout 2eme couche dense avec 128 neurones et fonction activation ReLu => App caract + complexe
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#ajout couche dense finale avec 10 neurones (=10 classes de chiffres) et fonction activation softmax => distrib de proba sur les 10 classes
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#compilation du modèle avec opti adam, fction perte sparse et mesure de l'exactitude pendant l'entrainement
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#entraine le modèle sur 3 époques = Passe 3fois sur les données pour ajuster les poids des neurones et améliorer précision
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

import matplotlib.pyplot as plt
#affiche 1ere image du jeu d'entrainement en nuance de gris
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
#affiche valeur pixels de l'image (val entre 0 et 1)
print(x_train[0])

#enregistre model dans un fichier
model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')

#fait des predictions sur les données test
predictions = new_model.predict([x_test])

print(predictions)

import numpy as np
#affiche l'indice de la valeur max du tableau 'predictions'
print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
