import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# data beldelt
def get_df(dataset_name):
    if dataset_name == "Customer segmentation":
        df = pd.read_csv("train_data_customer_segmentation.csv")
    else:
        df = pd.read_csv("train_data_product_segmentation.csv")
    X = df.iloc[:, 1:4].values 
    y = df.iloc[:, 4].values
    return X,y,df

#Decision Tree
def dTree(x_train, y_train, x_test, y_test, st):
    model = DecisionTreeClassifier() 
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    dt_train_score = model.score(x_train, y_train)
    dt_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(dt_test_score*100,2),"%")

#Random Forest
def rForest(x_train, y_train, x_test, y_test, st):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    rf_train_score = model.score(x_train, y_train)
    rf_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(rf_test_score*100,2),"%")


#Logistic Regression
def lRegression(x_train, y_train, x_test, y_test, st):
    model = LogisticRegression()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    lr_train_score = model.score(x_train, y_train)
    lr_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(lr_test_score*100,2),"%")

#Support Vector Machine
def SVM(x_train, y_train, x_test, y_test, st):
    model = SVC()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    svm_train_score = model.score(x_train, y_train)
    svm_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл :**",round(svm_test_score*100,2),"%")

#Naive Bayes
def nBayes(x_train, y_train, x_test, y_test, st):
    model = GaussianNB()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    nb_train_score = model.score(x_train, y_train)
    nb_test_score = model.score(x_test, y_test)

    st.write("**Гүйцэтгэл** :",round(nb_test_score*100,2),"%")

#Kmeans
def kMeans(X, k, df, st):
    scaler = StandardScaler()
    scaler.fit(X)
    selected_data_std=scaler.transform(X)
    selected_data_std_df=pd.DataFrame(selected_data_std, columns=['Recency','Frequency','Monetary'])
    matrix = selected_data_std_df.to_numpy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(selected_data_std_df)
    plt.hist(kmeans.labels_)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    cluster_df = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=selected_data_std_df.columns)

    cluster_df_std = pd.DataFrame(kmeans.cluster_centers_, columns=selected_data_std_df.columns)
    
    selected_data_with_X = df.iloc[:,1:4] 
    selected_data_with_X['cluster'] = kmeans.labels_
    for i in range(k):
        selected_data_with_X.loc[selected_data_with_X['cluster'] == i].value_counts(normalize=True)


    pca = PCA(n_components=3)
    pca.fit(selected_data_std)
    pca.components_
    X1 = pd.DataFrame(pca.components_, columns =['Recency','Frequency','Monetary'])

    st.write('Selected data has', len(X1.columns), 'features')
    st.write('3 principal components has', np.sum(pca.explained_variance_ratio_), 'total variance explanation')

    selected_data_std_pca = pca.transform(selected_data_std)
    selected_data_std_pca = pd.DataFrame(selected_data_std_pca)
    selected_data_std_pca
    
    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[1], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    plt.scatter(selected_data_std_pca[1], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired', alpha=0.8)
    st.pyplot()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(selected_data_std_pca[0], selected_data_std_pca[1], selected_data_std_pca[2], c=kmeans.labels_, cmap='Paired')
    st.pyplot()
    