import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.cluster import KMeans
from sklearn.decomposition import  PCA
from sklearn.model_selection import  train_test_split
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

#carrega dados
dataset = pd.read_csv('household_power_consumption.txt', delimiter=';',low_memory = False)

# visualiza as primeiras linhas
dataset.head()


#Dimensões do Dataset em linhas e colunas respectivamente
dataset.shape

#Verifica o tipo dos campos
dataset.dtypes

#Informações gerais do Dataset
dataset.info()

#checando se há valores missing
dataset.isnull().values.any()

#Checando onde há valores missing
dataset.isnull().sum()

#Remove os registros com valores NA e remove as duas primeiras colunas (não são necessárias)
dataset = dataset.iloc[0:, 2:9].dropna()

#verifica as primeiras linhas
dataset.head()

#checando se há valores missing
dataset.isnull().values.any()

#Checando onde há valores missing
dataset.isnull().sum()

#obtem os valores dos atributos. Obtem os valores de cada variável yn firnati de array
dataset_atrib = dataset.values

#imprime o array
dataset_atrib

#Coleta uma amostra de 1% dos dados para não comprometer a memoria do pc
dataset, amostra2 = train_test_split(dataset_atrib,train_size = 0.1)

dataset.shape

#Aplica redução de dimensionalidade no array das variaveis
pca = PCA(n_components= 2).fit_transform(dataset)

#determina um range do Hyperparâmetro "K" do Kmeans
k_range = range(1,12)
k_range

#Aplicando o modelo K-Means para cada valor de K (esta célula pode levar bastante tempo para ser encontrada)
k_means_var = [KMeans(n_clusters = k).fit(pca) for k in k_range]
#____________________________________________________________________________

#Curva de Elbow
centroids = [X.cluster_centers_ for X in k_means_var]

#Calculando a distancia euclidiana de cada ponto de dado para o centroide
k_euclid = [cdist(pca,cent, 'euclidean') for cent in centroids]
dist = [np.min(ke, axis = 1) for ke in k_euclid]

#soma dos quadrados das distancias dentro do cluster
soma_quadrados_intra_cluster = [sum(d**2) for d in dist]


#soma total dos quadrados
soma_total = sum(pdist(pca)**2)/pca.shape[0] #Deu erro por conta da quantidade de memoria alocada no PCA

#soma dos quadrados entre clusters
soma_quadrados_inter_cluster = soma_total - soma_quadrados_intra_cluster

#curva de Elbow
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range, soma_quadrados_inter_cluster/soma_total *100, 'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('N° de clusters')
plt.ylabel('% de variância explicada')
plt.title('variância explicada para cada valor de k')


#Criando um modelo com k = 8
modelo_v1 = KMeans(n_clusters = 8)
modelo_v1.fit(pca)

#Avaliação da máquina preditiva
x_min,x_max = pca[:,0].min() - 5, pca[:,0].max() -1
y_min, y_max = pca[:1].min() + 1, pca[:,1].max() +5
xx,yy = np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))
Z = modelo_v1.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

#Plot das áreas dos clusters
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation = 'nearest', 
                extent =(xx.min(),xx.max(),yy.min(),yy.max()),
                cmap= plt.cm.Paired,
                aspect = 'auto',
                origin = 'lower'
                )

#Métrica de avaliação para clusterização
#o melhor valor é 1 e o pior é -1
?silhouette_score#mostra a documentação

#Silhouette score
labels = modelo_v1.labels_
silhouette_score(pca,labels,metric='euclidean')#Resultado 0.7921586507527146


from sklearn.cluster import DBSCAN

dbscan_PCA = DBSCAN(eps=0.5,min_samples=20)
dbscan_PCA.fit(pca)


rotulos = dbscan_PCA.labels_
rotulos

dbscan_PCA = DBSCAN(eps=0.7,min_samples=8)
dbscan_PCA.fit(pca)

rotulos = dbscan_PCA.labels_
rotulos

from sklearn.metrics.cluster import davies_bouldin_score

labels = dbscan_PCA.labels_
silhouette_score(pca,labels,metric='euclidean')

davies_bouldin_score(pca,rotulos)
