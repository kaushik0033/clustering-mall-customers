# --------------
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn import preprocessing 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

#Importing the mall dataset with pandas
data=pd.read_csv(path)
data.columns=['cid','gender','age','income','spend']
# Create an array
X=data.iloc[:,[3,4]]
km=KMeans(n_clusters=6, random_state=0)
km.fit(X)
print(km.cluster_centers_)
print(len(km.labels_))
#plt.scatter(X.income.values,X.spend.values,c=km.labels_,cmap='rainbow')
plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='black')
km=KMeans(n_clusters=6, random_state=0)
km.fit(X)
#print(km.cluster_centers_)
#print(len(km.labels_))
#plt.scatter(X.income.values,X.spend.values,c=km.labels_,cmap='rainbow')
plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='black')

#model=KMeans()
#scree_vis=KElbowVisualizer(model,k=10)
#scree_vis.fit(X)
#best_model=KMeans(n_clusters=6)
#best_model.fit(X)
#best_model.cluster_Centers_
# Using the elbow method to find the optimal number of clusters
la=preprocessing.LabelEncoder()
X.gender=la.fit_transform(data.gender)

linked = linkage(X, 'single')
fig,ax_1=plt.subplots(figsize=(10,10))
dendrogram(linked,ax=ax_1)
ax_1.set_title('Dendogram')
ax_1.set_xlabel('Customer')
ax_1.set_ylabel('euclidian')
plt.show()

# Plot the graph to visualize the Elbow Method to find the optimal number of cluster  



# Applying KMeans to the dataset with the optimal number of cluster



# Visualising the clusters



# Label encoding and plotting the dendogram




