from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

path_train="/home/ali/Documents/Kaggle/ML1/train.csv"

# Let me read the data
data_=pd.read_csv(path_train,sep=",")

# Let me filter the data, and convert the categorical data to the numeric data
data_.drop(["Ticket","Cabin","Name","PassengerId"], inplace=True,axis=1)
data_["Embarked"].replace(["S","C","Q"],[0,1,2],inplace=True)
data_.loc[data_["Sex"]=="female","Sex"]=0
data_.loc[data_["Sex"]=="male","Sex"]=1


# Let me get the attributes and labels!


#
Attributes_=data_[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked']]
Labels_=data_["Survived"]


# Let me convert Attributes and Labels dataframes to numpy array and scale my data!



Attributes= np.array(Attributes_)
Attributes=scale(Attributes)
Labels_=np.array(Labels_)


from sklearn.model_selection import train_test_split

# Let  me now split my data to training set and test set!

X_train, X_test, y_train, y_test = train_test_split(Attributes,Labels_, test_size=0.3,random_state=109) # 70% training and 30% test data!

from sklearn.impute import SimpleImputer

# Let me replace the missing values with  the mean value using sklearn.impute function!

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)

X_train= imp.transform(X_train)
X_test= imp.transform(X_test)


'''
Now, I can generate whatever model I want! Let me try different models and evaluate each model. Because we have
binary labels, I can try binary classification algorithms! 3 of them might be logistic regression, kNN and SVM.
I will try each of them one by one and will evaluate my predictions!!
'''

# Let me start with SVM!

from sklearn import svm

# Let me create the SVM classifier

clf_kernel = svm.SVC(kernel='rbf') # Linear Kernel

#Let me train my model using the training datasets
clf_kernel.fit(X_train, y_train)

# Let me predict the results of the test datasets!

#Predict the response for test dataset
y_pred_kernel = clf_kernel.predict(X_test)


# Now, I can do the evaluation! I can employ various sklearn metrics modules to calculate the accuracy of my model!

from sklearn import metrics

print("My Accuracy is:",metrics.accuracy_score(y_test, y_pred_kernel))


# Let me also calculate the recall and precision of my model!

print("My Precision is:",metrics.precision_score(y_test, y_pred_kernel))
print("My Recall is:",metrics.recall_score(y_test, y_pred_kernel))


# The best kernel function was linear kernel, so we will now try to fit out data using kNN!

from sklearn.neighbors import KNeighborsClassifier

#
# Let me now fit out data with kNN algorithm

clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)

# Let me do our prediction!

y_pred_knn = clf_knn.predict(X_test)

# Let me look at the accuracy, precision and recall of my model!

print("My Accuracy is:",metrics.accuracy_score(y_test, y_pred_knn))
print("My Precision is:",metrics.precision_score(y_test, y_pred_knn))
print("My Recall is:",metrics.recall_score(y_test, y_pred_knn))

# Let me now do the logistic regression!

# Let me first fit out data!

from sklearn.linear_model import LogisticRegression as lr


# Let me now fit out data with Logistic regression algorithm

clf_lr = lr()
clf_lr.fit(X_train,y_train)

# Let me do our prediction!
y_pred_lr = clf_lr.predict(X_test)

# Let me look at the accuracy, precision and recall of my model!

print("My Accuracy is:",metrics.accuracy_score(y_test, y_pred_lr))
print("My Precision is:",metrics.precision_score(y_test, y_pred_lr))
print("My Recall is:",metrics.recall_score(y_test, y_pred_lr))


# Here, we saw that kNN worked better,so I will do my prediction with kNN, but will try others as well!

# Let me now import my testing data I want to predict!


# For this, similar to our training data, we need to filter our testing data!

data_test_path_="/home/ali/Documents/Kaggle/ML1/test.csv"

all_data=pd.read_csv(data_test_path_,sep=",")

# Let me filter the data, and convert the categorical data to the numeric data



data_test= all_data.drop(["Ticket","Cabin","Name","PassengerId"],axis=1)
data_test["Embarked"].replace(["S","C","Q"],[0,1,2],inplace=True)
data_test.loc[data_test["Sex"]=="female","Sex"]=0
data_test.loc[data_test["Sex"]=="male","Sex"]=1



# Let me get the attributes for the test data and convert it to the array and scale it!

Attributes_test=data_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
       'Embarked']]
Attributes_test= np.array(Attributes_test)
Attributes_test=scale(Attributes_test)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(Attributes_test)

Attributes_test= imp.transform(Attributes_test)
Attributes_test= imp.transform(Attributes_test)


# Let me now use my fitted models to do prediction on them!

# Let me first use all the models!!

y_tested_knn = clf_knn.predict(Attributes_test)
data_test["Predictions_knn"]=pd.Series(y_tested_knn)

y_tested_kernel = clf_kernel.predict(Attributes_test)
data_test["Predictions_kernel"]=pd.Series(y_tested_kernel)

y_tested_logistic = clf_lr.predict(Attributes_test)
data_test["Predictions_logistic"]=pd.Series(y_tested_logistic)

# Let me get at least 2 similar label similar to each  other as our prediction!!
data_test.to_excel("/home/ali/Documents/Kaggle/ML1/check.xlsx")
sum_=data_test[["Predictions_logistic","Predictions_kernel","Predictions_knn"]].sum(axis=1)

data_test["sum"]=pd.Series(sum_)


data_test.loc[data_test["sum"]>1,"Label"]=1
data_test.loc[data_test["sum"]<2,"Label"]=0

all_data["Label"]=data_test["Label"]

data_test.to_excel("/home/ali/Documents/Kaggle/ML1/control2_.xlsx")
all_data.to_excel("/home/ali/Documents/Kaggle/ML1/control1_.xlsx")

all_data=all_data[["PassengerId","Label"]]
all_data.columns= ["PassengerId","Survived"]


all_data.to_csv("/home/ali/Documents/Kaggle/ML1/predictions.csv",index=False,sep=",")
