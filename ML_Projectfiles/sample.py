import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import io
import requests
# seaborn and matplotlib are great libraries for Data Viz
import seaborn as sns

def fun2():

    dataset = pd.read_csv("University_data_cobine_final_testing.csv")

    # get the data type of dataset
    print(type(dataset))
    print("--------------------------------------------------------------------------------------------------------")

    # look into the first five rows of the dataset
    print(dataset.head())
    print("--------------------------------------------------------------------------------------------------------")

    # get the shape of dataset - {rows,columns}
    print(dataset.shape)
    print("--------------------------------------------------------------------------------------------------------")

    # get some information about the dataset
    print(dataset.info())
    print("--------------------------------------------------------------------------------------------------------")

    # count the number of non-NA values in the dataset
    print(dataset.count())
    print("--------------------------------------------------------------------------------------------------------")

    # get info about the columns/attributes of the dataset
    print(dataset.columns)
    print("--------------------------------------------------------------------------------------------------------")

    # get the sum of values for each column/attribute in the dataset
    print(dataset.sum())
    print("--------------------------------------------------------------------------------------------------------")

    # get the summary statistics of the dataset
    print(dataset.describe())
    print("--------------------------------------------------------------------------------------------------------")

    # get the mean values of each column/attribute in the dataset
    print(dataset.mean())
    print("--------------------------------------------------------------------------------------------------------")

    # get the median values of each column/attribute in the dataset
    print(dataset.median())
    print("--------------------------------------------------------------------------------------------------------")


    sns.set(style="white", context="talk")

    # two plots as subplots
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    # plot SAT Verbal scores
    sns.distplot(dataset.ix[:, 6], ax=ax1, color="r");

    # plot SAT Maths grades
    sns.distplot(dataset.ix[:, 7], ax=ax2, color="g");

    # display the plot
    sns.plt.show()

    print("--------------------------------------------------------------------------------------------------------")
    # multivariate analysis

    sns.pairplot(dataset, hue='admission', palette="husl",
                 x_vars=["SAT Verbal", "SAT Maths"],
                 y_vars=["SAT Verbal", "SAT Maths"], size=4)

    # display the plot
    sns.plt.show()


if __name__ == '__main__':
    fun2()
