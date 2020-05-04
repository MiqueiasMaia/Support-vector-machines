# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Support-vector-machines/credit_data/credit_dt.csv')

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

from sklearn.svm import SVC
classificador = SVC(kernel='linear', random_state=1, C=0.5)
# classificador = SVC(kernel='poly', random_state=1)
# classificador = SVC(kernel='sigmoid', random_state=1)
# classificador = SVC(kernel='rbf', random_state=1)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
print('precisao: ')
print(precisao)
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)
