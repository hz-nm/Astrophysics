from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1= pd.read_csv("June_Monthly_data.csv", delimiter= ',')
Check_NaNs=df1.isnull().values.any()
# df1=df.drop(df.columns[[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,27,28,30,31,32,33,34,35,36,37,38]], axis=1)
# df1=df1.drop(df1.columns[[0,4,5]], axis=1)
df1.applymap(np.isreal)
# df1['MEAN_MOTION'] = list(map(lambda x: x.isdigit(), df1['MEAN_MOTION']))
# df1 = df1.astype(float)
# df1 = df1[0:11]
# df1['MEAN_MOTION2'] = df1['MEAN_MOTION'].astype(float)

# CONVERSION TO FLOAT64
# con_s_fl = lambda x: float(x)
# df1= pd.read_csv("June_Monthly_data.csv", delimiter= ',')
# initial = 0
# for x in range(1,80):
#     if x == 79:
#         df1['MEAN_MOTION'] = df1['MEAN_MOTION'][initial : length].apply(con_s_fl)
        
#     df1['MEAN_MOTION'] = df1['MEAN_MOTION'][initial : x * 10000].apply(con_s_fl)
#     initial = x * 10000
#     print(x)

# DROP DUPLICATES
# df_k_newest = df_new.drop_duplicates(subset=['OBJECT_ID'])

#CONVERSION
# def remov(x):
#     if x in ['MEAN_MOTION', 'SEMIMAJOR_AXIS', 'PERIOD', 'AVG_ALTITUDE']:
#         return 0
#     else:
#         return float(x)
        
# for x in df_copy.columns:
#     df_copy[x] = df_copy[x].apply(remov)

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# label_encoder = LabelEncoder()
# true_labels = label_encoder.fit_transform(df1[["OBJECT_TYPE"]])
# # true_labels[:5]
# n_clusters = len(label_encoder.classes_)

kmeans = KMeans(
init="k-means++",
n_clusters=8,
n_init=10,
max_iter=300,
random_state=42)

kmeans.fit(df1[['MEAN_MOTION', 'SEMIMAJOR_AXIS', 'PERIOD','AVG_ALTITUDE' ]])
# colors_list = ['blue', 'black','red', 'green','yellow'] 
x = [i for i in range(len(df1))]
data_with_clusters = df1.copy()
data_with_clusters['Clusters'] = kmeans.labels_
plt.scatter(x,data_with_clusters['AVG_ALTITUDE'],c=data_with_clusters['Clusters'])

data_with_clusters = data_with_clusters.copy()
data_with_clusters['Clusters'] = kmeans.labels_
data = data_with_clusters.sort_values('Clusters')
data.to_csv('Sorted_Clusters_ID.csv', index=False)
# plt.scatter(x,data_with_clusters['APOAPSIS'],c=data_with_clusters['Clusters'],cmap='rainbow')

# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()