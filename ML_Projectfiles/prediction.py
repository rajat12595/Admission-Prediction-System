# organize imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import io
import requests
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def fun1():

    #University_data_cobine_final_testing.csv
    dataset = pd.read_csv("University_data_cobine_final_testing - Copy.csv")

    # convert the dataframe into a matrix
    dataArray = dataset.values


    # split the input features and output variable
    X = dataArray[:, 1:3]
    y = dataArray[:, 0:1]


    # split training and testing dataset
    validation_size = 0.10
    seed = 9
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=validation_size, random_state=seed)

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    print("--------------------------------------------------------------------------------------------------------")

    # create the model
    model = LogisticRegression()

    # prepare the models - {LR, LDA, KNN, CART, RF, NB, SVM}
    num_trees = 200
    max_features = 2
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # fit the models and evaluate it
    results = []
    names = []
    scoring = 'accuracy'

    # evaluate each model using 10-FOLD cross validation
    #for name, model in models:
     #   kfold = KFold(n_splits=10, random_state=7)
      #  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
       # results.append(cv_results)
        #names.append(name)
       # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        #print(msg)

    # boxplots for ML algorithm comparison
    #fig = pyplot.figure()
    #fig.suptitle('Machine Learning algorithms comparison')
    #ax = fig.add_subplot(111)
    #pyplot.boxplot(results)
    #ax.set_xticklabels(names)
    #pyplot.show()

    # create the model
    model = LogisticRegression()

    # fit the model
    model.fit(X_train, Y_train)

    # make predictions on the test data
    predictions = model.predict(X_test)


    cm = confusion_matrix(Y_test, predictions)
    sns.heatmap(cm,
                annot=True,
                xticklabels=['reject', 'admit'],
                yticklabels=['reject', 'admit'])
    plt.figure(figsize=(3, 3))

    # compute the overall accuracy and display the classification report
    print("Model --> Logistic Regression")
    print("Overall Accuracy: {}").format(accuracy_score(Y_test, predictions) * 100)
    print(classification_report(Y_test, predictions))

    # plot confusion matrix and display the heatmap
    sns.plt.show()

    # make prediction on a new test data - (gre_score, gpa_grade, rank)
    new_data = [(150,150), (700,450), (550,550)]

    # convert the list of tuples to numpy array
    new_array = np.asarray(new_data)

    # the output labels
    labels = ["admit","reject"]

    # make prediction
    prediction = model.predict(new_array)

    # get the no.of.test cases used
    no_of_test_cases, cols = new_array.shape

    # show the result
    for i in range(no_of_test_cases):
        print("Status of STUDENT with GRE score= {}, GPA grade= {} will be --> {}".format(new_data[i][0],new_data[i][1],labels[int(prediction[i])]))
if __name__ == '__main__':
    fun1()

