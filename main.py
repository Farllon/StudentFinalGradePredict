# Importações
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Obtendo o dataset
data = pd.read_csv("student-mat.csv", sep=";")

# Processando o dataset
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Defifnindo a variável de predição
predict = "G3"

# Definindo os arrays dos eixos de treinamento, onde
# a matriz X contem 5 eixos e o eixo Y é unário
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# Definindo as variáveis de treino e testes
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

# Definindo o modo de treino
linear = linear_model.LinearRegression()

# Treinando a rede e obtendo a precisão
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

# Fazendo a predição com os testes
predictions = linear.predict(x_test)

# Imprimindo as predições
for pred in range(len(predictions)):
  print(predictions[pred], x_test[pred], y_test[pred])
