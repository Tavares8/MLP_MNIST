# Professora: Tatiana S. Tavares - Pós Inteligência Artificial 2018/2
# Entrada: conjunto de dados MNIST. Entrada (N, 784) imagens vetorizadas
# MLPClassifier: uma camada oculta com 256 neurônios, taxa de aprendizagem de 0,01 e função de ativação 
# retificadora linear.
# Métricas: taxas de acertos (acurácia) no treinamento e na validação e matriz de confução.

# Intalar bibliotecas
import urllib
import _pickle as pickle
import os
import gzip 

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Ler banco de dados - shape (N,784)
def load_dataset():
	url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
	filename = 'mnist.pkl.gz'
	if not os.path.exists(filename):
		print('Downloading MNIST dataset...')
		urllib.request.urlretrieve(url, filename)
	with gzip.open(filename, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
        
	X_train, y_train = data[0]
	X_val, y_val = data[1]
	X_test, y_test = data[2]
    
	return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# Plota imagem como vetor e como matrix
X_train_vector = X_train.reshape((-1, 1, 1, 784))
plt.imshow(X_train_vector[5][0], cmap=cm.binary)
plt.show()

X_train_matrix = X_train.reshape((-1, 1, 28, 28))
plt.imshow(X_train_matrix[5][0], cmap=cm.binary)
plt.show()

# Definir arquitetura 
mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=4, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01)

# Treinar o modelo
mlp.fit(X_train, y_train)

# Métricas do treinamento
print('Métricas do treinamento')
print("Erro no final do treinamento: %f" % mlp.loss_)

loss_values =mlp.loss_curve_
plt.plot(loss_values)
plt.xlabel("n iterações")
plt.ylabel("Erro")
plt.show()

# Validar o modelo
preds_val = mlp.predict(X_val)

# Métricas da validação
print('Métricas da validação')
print("Acertos do conjunto de validação: %f" % mlp.score(X_val, y_val))

print(confusion_matrix(y_val,preds_val))
MC = confusion_matrix(y_val, preds_val)
plt.matshow(MC)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_val,preds_val))
#support = quantidade de exemplos de cada classe

