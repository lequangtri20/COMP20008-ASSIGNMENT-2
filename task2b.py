import pandas as pd
import numpy as np

from sklearn import neighbors, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
np.random.seed(250)

#==============================================================================
#Linking data between world.csv and life.csv

world = pd.read_csv("world.csv", engine = 'python', na_values = ['..'])
life  = pd.read_csv("life.csv", engine = 'python', na_values = ['..'])

world = world.sort_values(by = 'Country Code', ascending = True)
life = life.sort_values(by = 'Country Code', ascending = True)

# Perform inner merge between two files    
final_set = world.merge(life,on = "Country Code", how = "inner")
final_set = final_set.sort_values(by = 'Country Code', ascending = True)
labels = []
for label in final_set["Life expectancy at birth (years)"]:
    labels.append(label)

#==============================================================================
# Splitting data set
data = final_set.drop(['Country Name', 'Time', 'Country Code' , 'Country',\
                       'Life expectancy at birth (years)',"Year"],\
                      axis = 1).astype(float)
X_train, X_test, y_train, y_test =\
train_test_split(data, labels, train_size=0.7, test_size=0.3, random_state=250)

#==============================================================================
# Imputing
imp_median_train = SimpleImputer(missing_values= np.nan, strategy='median')
imp_median_train = imp_median_train.fit(X_train)
X_train = imp_median_train.transform(X_train)

X_test = imp_median_train.transform(X_test)

#==============================================================================
# Scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scale=scaler.transform(X_train)
X_test_scale=scaler.transform(X_test)

#==============================FEATURE GENERATION===============================

# Interaction terms pairs
interaction = PolynomialFeatures(degree=2, include_bias=False,\
                                 interaction_only=True)
    
interaction = interaction.fit(X_train_scale, y_train)
X_train_inter = pd.DataFrame(interaction.transform(X_train_scale),\
                             columns=interaction.get_feature_names\
                                 (input_features=list(data.columns)))
    
X_test_inter =  pd.DataFrame(interaction.transform(X_test_scale),\
                             columns=interaction.get_feature_names\
                                 (input_features=list(data.columns)))

print('\033[1m'+"\nNames of 210 features after interaction term pairs"+'\033[0m')
print(pd.Series(list(X_train_inter.columns)))
#=====================================
#Clustering labels

# Elbow test
wcss = []
for i in range(1, 11):
    kmeans_train = KMeans(n_clusters=i, init='k-means++', max_iter=300,\
                          n_init=10, random_state=300)
    kmeans_train.fit(X_train_scale)
    wcss.append(kmeans_train.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
plt.xticks(range(1, 11))
ax.set_title("Elbow test")
plt.ylabel("WCSS")
plt.xlabel("Number of Clusters")
plt.tight_layout()
plt.savefig('task2bgraph1.png')

#=====================================
#Silhouette test
sse = []
for k in range(2,11):
    kmeans = KMeans(init="random",n_clusters=k,n_init=10,max_iter=300,\
                    random_state=300)
    kmeans.fit(X_train_inter)
    sse.append(silhouette_score(X_train_inter, kmeans.labels_,\
                                metric='euclidean'))
    
fig2, ax2 = plt.subplots()
ax2.plot(range(2, 11), sse)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
ax2.set_title("Silhouette Test")
plt.tight_layout()
plt.savefig('task2bgraph2.png')

print('\033[1m'+"\nElbow graph's bend & Global Maximum of Silhouette test is at\
 k=3 (task2bgraph1.png & task2bgraph2.png). Thus k = 3 is chosen." + '\033[0m')

#=====================================
# Based on Elbow test and Silhouette test, choose k = 3
# Creating f_cluster_label
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=250,\
                random_state=300)
    
kmeans.fit(X_train_scale)
X_train_inter['Cluster Label'] = kmeans.predict(X_train_scale)
X_test_inter["Cluster Label"] = kmeans.predict(X_test_scale) 

print("\n"+'\033[1m'+"Names of 211 features after clustering" + '\033[0m')
print(pd.Series(list(X_train_inter.columns)))

pd.set_option('max_rows', 10)

print("\n"+"\033[1m"+"Cluster Label for Train Set"+ '\033[0m')
print(X_train_inter['Cluster Label'])

print("\n"+"\033[1m"+"Cluster Label for Test Set"+ '\033[0m')
print(X_test_inter['Cluster Label'])


#=====================================
# Select 4 from 211 features, get accuracy by applying 3-nn classification

skb = SelectKBest(score_func=mutual_info_classif, k=4)
fit_train = skb.fit(X_train_inter, y_train)

mask = fit_train.get_support()
best_features = X_train_inter.columns[mask]
mi = mutual_info_classif(X_train_inter, y_train, random_state=250)[mask]

print("\n"+"\033[1m"+"Top 10 highest Mutual Information Values from 211 features\
      "+ '\033[0m')
print(pd.Series(sorted(mutual_info_classif(X_train_inter, y_train, \
                                    random_state=250), reverse = True)[:10]))
    
print("\n"+"\033[1m"+"Best 4 features with highest Mutual Information Values"+\
      '\033[0m')
df_4_best = pd.DataFrame({"Feature Name": best_features , 'MI score': mi}).\
    sort_values(by = 'MI score', ascending = False,ignore_index=True)
    
print(df_4_best, "\n")

features = fit_train.transform(X_train_inter)
features2 = fit_train.transform(X_test_inter)

knn_best4 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_best4 = knn_best4.fit(features , y_train)
y_pred_best4 = knn_best4.predict(features2)


#=====================================PCA======================================
pca = PCA(n_components=4, random_state = 250)
pca.fit(X_train_scale)

X_train_PCA = pca.transform(X_train_scale)
X_test_PCA = pca.transform(X_test_scale)

knn_PCA = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_PCA.fit(X_train_PCA, y_train)

y_pred_PCA=knn_PCA.predict(X_test_PCA)

print("\033[1m"+"First 4 Principal Components in Training Set"+'\033[0m')
print(pd.DataFrame(X_train_PCA))
print("\n"+"\033[1m"+"First 4 Principal Components in Test Set"+'\033[0m')
print(pd.DataFrame(X_test_PCA))
#==============================================================================
#First 4 columns 
X_train_4 = X_train_scale[:,:4]
X_test_4 = X_test_scale[:,:4]

knn_4 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_4.fit(X_train_4, y_train)

y_pred_4=knn_4.predict(X_test_4)

print("\n" +"\033[1m"+"First 4 Features D-G in Training Set"+'\033[0m')
print(pd.DataFrame(X_train_4,columns=pd.Series(list(X_train_inter.columns)[:4])))
print("\n"+"\033[1m"+"First 4 Features D-G in Test Set"+'\033[0m'+ "\n")
print(pd.DataFrame(X_test_4,columns=pd.Series(list(X_train_inter.columns)[:4])))
#==============================================================================
# Printing accuracy for 3 features sets
print("Accuracy of feature engineering: " ,round(accuracy_score(y_test,\
                                                            y_pred_best4),3))

print("Accuracy of PCA: " ,round(accuracy_score(y_test, y_pred_PCA),3))
print("Accuracy of first four features: ",\
      round(accuracy_score(y_test, y_pred_4),3))




