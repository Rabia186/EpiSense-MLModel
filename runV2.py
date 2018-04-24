
#new one from scratch!!
import numpy as np
import pandas as pd
from sklearn import svm, neighbors, preprocessing
import csv
import time 
from time import gmtime, strftime
from pandas import DataFrame
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

#from IPython.display import Image, display


	
data = pd.read_csv("C:\Users\Rabia Rauf\Desktop\FINALTRAIN2.csv", )  #FOR TRAINING THE FILE THAT IS ON YOUR DESKTOP

data = data.drop("id",1)

#opening a file to write data to it


features = list(data.columns[3:16])
x = data[features]
y = data["diagnosis"]

#print(x,y)


# now!, we gotta train multiple files!
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.3, random_state=42)
#normalization
x_train_N = (x_train-x_train.mean())/(x_train.max()-x_train.min())
x_test_N = (x_test-x_test.mean())/(x_test.max()-x_test.min())

clf = svm.SVC()
fitme = clf.fit(x_train, y_train)

#print(fitme)

accuracy = clf.score(x_test,y_test)
print(accuracy)


#NOW LETS TRY TO USE A K-FOLD VALIDATION


#TEH ACCUARCY
#Kfold = KFold(len(data),n_folds=10,shuffle=False)
#print("SVM accuracy after using KFold is %s" %cross_val_score(fitme,x,y,cv=10).mean())




#prints the no. of rows and columns in the file
#print(data.shape) 
 

#testing with PCA
'''pca = PCA(n_components=2)# adjust yourself
pca.fit(x_train)
x_t_train = pca.transform(x_train)
x_t_test = pca.transform(x_test)
clf = SVC()
clf.fit(x_t_train, y_train)
print 'score', clf.score(x_t_test, y_test)'''




#NOW LETS CHANGE IT TO A FILE!






testdata = pd.read_csv("C:\Users\Rabia Rauf\Desktop\FINALTEST2.csv")
testdata = testdata.drop("id",1)
#pwid = list(testdada.columns[1:1],ypred)

testfeatures = list(testdata.columns[3:16])
x2 = testdata[testfeatures]
#y2 = testdata["diagnosis"]   #cause you dont have the y!

ypred2 = clf.predict(x2)


currenttime1 = strftime("%Y-%m-%d %H:%M:%S", gmtime())
print(ypred2)




   
#FOR PRINTING ONLY THE OUTPUT AND TIME STAMP IN THE FILEEE:
   
df = DataFrame({'Time': currenttime1, 'Output': ypred2})    

df.to_csv('FINALOUTPUT2.csv')
		
#f.close()








