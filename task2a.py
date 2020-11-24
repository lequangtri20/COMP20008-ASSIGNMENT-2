import pandas as pd
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
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

#Splitting data set
data = final_set.drop(['Country Name', 'Time', 'Country Code' , 'Country',\
                       'Life expectancy at birth (years)',"Year"],\
                      axis = 1).astype(float)
    
X_train, X_test, y_train, y_test =\
train_test_split(data, labels, train_size=0.7, test_size=0.3, random_state=200)
#==============================================================================
#Imputing
imp_median_train = SimpleImputer(missing_values= np.nan, strategy='median')
imp_median_train = imp_median_train.fit(X_train)
X_train = imp_median_train.transform(X_train)

X_test = imp_median_train.transform(X_test)

# Taking the median values
median = imp_median_train.statistics_

#==============================================================================
#Scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#==============================================================================
#Decision tree
dt = DecisionTreeClassifier(random_state=200, max_depth=3)
dt.fit(X_train, y_train)
y_pred_dt=dt.predict(X_test)
print("Accuracy of decision tree: " ,round(accuracy_score(y_test, y_pred_dt),3))

#==============================================================================
#KNeighborsClassifier
# k = 3
knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
y_pred3=knn3.predict(X_test)
print("Accuracy of k-nn (k=3): " ,round(accuracy_score(y_test, y_pred3),3))

# k = 7
knn7 = neighbors.KNeighborsClassifier(n_neighbors=7)
knn7.fit(X_train, y_train)
y_pred7=knn7.predict(X_test)
print("Accuracy of k-nn (k=7): " ,round(accuracy_score(y_test, y_pred7),3))

#==============================================================================
# median, mean, variance for scaling
head = ["feature","median", "mean", "variance"]
mean = scaler.mean_
variance = scaler.var_
feature = data.columns

d = {'feature': feature,'median':median, 'mean':mean ,'variance': variance}
df_task2a = pd.DataFrame(d)
df_task2a.to_csv("task2a.csv", index=False)
