# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:51:22 2020

@author: Mandar
"""


#import all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
import matplotlib

#load the dataset
df=pd.read_csv("F:\\Data Science\\Python\\ChirantanPythonSpyder\\ProjectFinal\\OnlineRetail.csv",encoding='latin1')

#understand the dataset
df.info()
df_describe=df.describe()
df.head()
df.shape

#find the null values
df.isnull().sum()
sns.heatmap(df.isnull())

#remove the null values
df=df.dropna()

#calculate the total amount
TotalAmount = df['Quantity'] * df['UnitPrice']
df.insert(loc=6, column = 'TotalAmount', value=TotalAmount)

#correlation matrix
df_corr=df.corr()
plt.figure(figsize = (12,10))
sns.heatmap(df_corr,square=True,annot=True,linewidths=4,linecolor='k')

#convert datatype of the required columns
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

#split date and time
df['month']= pd.to_datetime(df['InvoiceDate']).dt.month
#df['day'] = pd.to_datetime(df['InvoiceDate']).dt.dayofweek
df['day_number'] = pd.to_datetime(df['InvoiceDate']).dt.day
df['year'] = pd.to_datetime(df['InvoiceDate']).dt.year
df['hour'] = pd.to_datetime(df['InvoiceDate']).dt.hour
df['m_y']=pd.to_datetime(df['InvoiceDate']).dt.to_period('M')

df['m_y']=df['m_y'].astype(str)

#finding unique values of certain columns
df['year'].unique()
df['month'].unique()

#create a new column showing a specific number with respect to each country
df['Country_Category']=df["Country"].astype('category').cat.codes
df['InvoiceNo']=df["InvoiceNo"].astype('category').cat.codes
df['StockCode']=df["StockCode"].astype('category').cat.codes
df['Description']=df["Description"].astype('category').cat.codes
df['Description']=df["Description"].astype('category').cat.codes

#create a dataframe showing the number represented by each country
a=df['Country'].unique()
b=df['Country_Category'].unique()
data = a,b
Countries = pd.DataFrame(data=data)

#boxplot of Quantity
plt.figure(figsize = (12,6))
sns.boxplot(df['Quantity'],data=df)
plt.xlabel("Quantity")
plt.grid(linestyle='-.',linewidth = .5)

#violin plot of Quantity
plt.figure(figsize = (12,6))
sns.violinplot(df['Quantity'],data=df)
plt.xlabel("Quantity")
plt.grid(linestyle='-.',linewidth = .5)

#scatter plot of quantity vs month
plt.scatter(df['month'],df['Quantity'],color="blue")
plt.show()

#removing outliers of Quantity 
max_quantity=df['Quantity'].quantile(0.90)
a1=df[df['Quantity']>max_quantity]
min_quantity=df['Quantity'].quantile(0.10)
b1=df[df['Quantity']<min_quantity]
df=df[(df['Quantity']<max_quantity) & (df['Quantity']>min_quantity)]

#boxplot of TotalAmount
plt.figure(figsize = (12,6))
sns.boxplot(df['TotalAmount'],data=df)
plt.xlabel("TotalAmount")
plt.grid(linestyle='-.',linewidth = .5)

#violin plot of TotalAmount
plt.figure(figsize = (12,6))
sns.violinplot(df['TotalAmount'],data=df)
plt.xlabel("TotalAmount")
plt.grid(linestyle='-.',linewidth = .5)

#scatter plot of TotalAmount vs month
plt.scatter(df['month'],df['TotalAmount'],color="blue")
plt.show()

#removing outliers of TotalAmount
max_TotalAmount=df['TotalAmount'].quantile(0.92)
a2=df[df['TotalAmount']>max_TotalAmount]
min_TotalAmount=df['TotalAmount'].quantile(0.08)
b2=df[df['TotalAmount']<min_TotalAmount]
df=df[(df['TotalAmount']<max_TotalAmount) & (df['TotalAmount']>min_TotalAmount)]

#boxplot of UnitPrice
plt.figure(figsize = (12,6))
sns.boxplot(df['UnitPrice'],data=df)
plt.xlabel("UnitPrice")
plt.grid(linestyle='-.',linewidth = .5)

#violin plot of UnitPrice
plt.figure(figsize = (12,6))
sns.violinplot(df['UnitPrice'],data=df)
plt.xlabel("UnitPrice")
plt.grid(linestyle='-.',linewidth = .5)

#scatter plot of UnitPrice vs month
plt.scatter(df['month'],df['UnitPrice'],color="blue")
plt.show()

#scatter plot of quantity vs unit price
plt.scatter(df['Quantity'],df['UnitPrice'],color="blue")
plt.show()

#removing outliers of UnitPrice
max_UnitPrice=df['UnitPrice'].quantile(0.85)
a2=df[df['UnitPrice']>max_UnitPrice]
min_UnitPrice=df['UnitPrice'].quantile(0.15)
b2=df[df['UnitPrice']<min_UnitPrice]
df=df[(df['UnitPrice']<max_UnitPrice) & (df['UnitPrice']>min_UnitPrice)]

#bargraph of month wise quantity
df.groupby(['m_y'])['Quantity'].mean().plot(kind='bar')
plt.ylabel('Quantity')

#bargraph of month wise TotalAmount
df.groupby(['m_y'])['TotalAmount'].mean().plot(kind='bar')
plt.ylabel('UnitPrice')

#bargraph of country wise quantity
df.groupby(['Country'])['Quantity'].mean().plot(kind='bar')
plt.ylabel('Quantity')

#bargraph of country wise hourly orders
df.groupby(['Country_Category'])['hour'].mean().plot(kind='bar')
plt.ylabel('Quantity')

#bargraph of country wise TotalAmount
df.groupby(['Country'])['TotalAmount'].mean().plot(kind='bar')
plt.ylabel('UnitPrice')

#boxplot of Quantity for monthly analysis
plt.figure(figsize=(12,7))
sns.boxplot(x='m_y', y = 'Quantity', data = df)
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.title("Monthly analysis of Quantity")
plt.grid(linestyle='-.',linewidth = .2)

#boxplot of TotalAmount for monthly analysis
plt.figure(figsize=(12,7))
sns.boxplot(x='m_y', y = 'TotalAmount', data = df)
plt.xlabel('Month')
plt.ylabel('TotalAmount')
plt.title("Monthly analysis of TotalAmount")
plt.grid(linestyle='-.',linewidth = .2)

#linegraph of month vs quantity
fig, ax = plt.subplots(figsize=(10,7))
df.groupby(['m_y'])['Quantity'].mean().plot(ax=ax)
plt.show(fig,ax)

#linegraph of month vs TotalAmount
fig, ax = plt.subplots(figsize=(10,7))
df.groupby(['m_y'])['TotalAmount'].mean().plot(ax=ax)
plt.show(fig,ax)

#histogram of quantity
plt.figure(figsize=(10,7))
plt.hist(df['Quantity'],color='orange', bins=20)
plt.show()

#histogram of TotalAmount
plt.figure(figsize=(10,7))
plt.hist(df['TotalAmount'],color='orange', bins=20)
plt.show()

#distplot of quantity
plt.figure(figsize = (12,6))
sns.distplot(df['Quantity'],kde=True,bins=20)
plt.xlabel("Quantity")
plt.title("Destribution of frequency")
plt.grid(linestyle='-.',linewidth = .5)

#distplot of TotalAmount
plt.figure(figsize = (12,6))
sns.distplot(df['TotalAmount'],kde=True,bins=20)
plt.xlabel("TotalAmount")
plt.title("Destribution of frequency")
plt.grid(linestyle='-.',linewidth = .5)

#regplot of country vs total amount
sns.regplot(df['TotalAmount'],df['Country_Category'],color="blue")
plt.show()

#bargraph of country wise customers
df.groupby(['Country_Category'])['CustomerID'].plot(kind='bar')
plt.ylabel('Customers')

#bargraph of customer wise quantity
plt.figure(figsize = (15,6))
df.groupby(['CustomerID'])['Quantity'].mean().plot(kind='bar')
plt.ylabel('Quantity')

#scatter plot of Country_Category vs CustomerID
plt.scatter(df['CustomerID'],df['Country_Category'],color="blue")
plt.show()

#lmplot to check the relation of all the columns
df=df.drop(['InvoiceDate','Country'],axis=1)
for i in df.columns.tolist():
    sns.lmplot(x=i,y='Country_Category',data=df,markers='.')
    
sns.lmplot(x='CustomerID',y='Country_Category',data=df,markers='.')

#find the number of unique values in columns
len(pd.unique(df['Description']))
len(pd.unique(df['Country']))

#create the dependant and independant variables
X=df.drop(['Country_Category','InvoiceDate','m_y','Country'],axis=1)
Y=df['Country_Category'].values.reshape(-1,1)

#Perform lasso regression on the data
names=X.columns
lasso=Lasso(alpha=0.2)
lasso_coef=lasso.fit(X,Y).coef_

#plot a graph showing the coefficients of each independant variable
figure=plt.plot(range(len(names)),lasso_coef)
figure=plt.xticks(range(len(names)),names,rotation=60)
figure=plt.ylabel('Coefficient')
plt.show()

#create a function to find out the root mean squared error
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model,X,Y,scoring="neg_mean_squared_error",cv = 5))
    return(rmse)

#perform lasso regression
model_lasso = LassoCV(alphas = [1, 0.1 , 0.001, 0.0005], selection = 'random', max_iter=15000).fit(X,Y)
res= rmse_cv(model_lasso)

#find out the min and mean root mean squared error
print("Mean:",res.mean())
print("Min:",res.min())

#print the variables that lasso picked
coeff = pd.Series(model_lasso.coef_,index = X.columns)
print("Lasso picked"+ str(sum(coeff!=0))+"Variables and eliminated and the other"+ str(sum(coeff == 0))+" variables")

#coefficients of all the columns
imp_coef = pd.concat([coeff.sort_values().head(10),coeff.sort_values().tail(10)])
  
#plot the graph of lasso regression 
matplotlib.rcParams['figure.figsize'] =(8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

#Perform ridge regression on the data
ridge=Ridge(alpha=0.2)
ridge_coef=ridge.fit(X,Y).coef_

#perform ridge regression
model_ridge = RidgeCV(alphas = [1, 0.1 , 0.001, 0.0005]).fit(X,Y)
res = rmse_cv(model_ridge)

#find out the min and mean root mean squared error
model_ridge.coef_
print("Mean:",res.mean())
print("Min:",res.min())

#print the coefficient that ridge picked
coeff = pd.Series(model_ridge.coef_,index = X.columns)
print("Ridge Regression picked"+ str(sum(coeff!=0))+"Variables and eliminated and the other"+ str(sum(coeff == 0))+" variables")

#coefficients of all the columns
imp_coef = pd.concat([coeff.sort_values().head(10),coeff.sort_values().tail(10)])
   
#plot the graph of ridge regression 
matplotlib.rcParams['figure.figsize'] =(8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Ridge Regression Model")

#copy df into a new dataset and drop unwanted columns
df1=df.copy()
df1.info()
df1=df1.drop(['InvoiceDate','m_y','Country'],axis=1)

#scale down the values of the dataset
scaler=StandardScaler()
scaler.fit(df1)
scaled_data=scaler.transform(df1)

#perform PCA
pca=PCA(n_components=2)
pca.fit(df1)
x_pca=pca.transform(df1)

#the dimensionality is reduced to 2
x_pca.shape

#plot the x_pca
plt.scatter(x_pca[:,0],x_pca[:,1],c=df1['CustomerID'])

#drop unwanted columns and create independant and dependant variables for non linear algorithms
X1=X.drop(['InvoiceNo','Description','UnitPrice'],axis=1)
Y1=Y

#split train and test data
x_train,x_test,y_train,y_test=train_test_split(X1,Y1,test_size=0.3)

#perform decision tree algorithm
dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)

#print the accuracy score for decision tree
print(accuracy_score(y_test,y_pred))

#classification report of decision tree
cr_dt=classification_report(y_test, y_pred)

#using regplot for decision tree
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)

#perform support vector machine algorithm
svm1 = SVC()
svm1.fit(x_train,y_train)
y_pred = svm1.predict(x_test)

#print accuracy score for support vector machine
print(accuracy_score(y_test,y_pred))

#classification report of support vector machine
cr_svm1=classification_report(y_test, y_pred)

#using regplot for support vector machine
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)

#perform random forest algorithm
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

#print accuracy score for random forest
print(accuracy_score(y_test,y_pred))

#classification report of random forest
cr_rfc=classification_report(y_test, y_pred)

#using regplot for random forest
sns.regplot(y_test,y_pred,order=1, ci=None, scatter_kws={'color':'black', 's':10})
plt.scatter(y_test,y_pred,s=2)

#create a suitable range of clusters
k_range=range(1,11)
sse=[]
for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(df[['CustomerID','Country_Category']])
    sse.append(km.inertia_)

#plot a graph to finalise cluster number by using the elbow rule
plt.xlabel('k')
plt.ylabel('sum of squared error')
plt.plot(k_range,sse)

#perform KMeans clustering
km1=KMeans(n_clusters=3)
y_predicted=km1.fit_predict(df[['CustomerID','Country_Category']])
df['cluster']=y_predicted

#create datasets with different specific cluster values
cluster1=df[df.cluster==0]
cluster2=df[df.cluster==1]
cluster3=df[df.cluster==2]

#find the centroids of each clusters
centroids=km1.cluster_centers_

#plot the cluster graph
plt.scatter(cluster1.CustomerID,cluster1['Country_Category'],color='green')
plt.scatter(cluster2.CustomerID,cluster2['Country_Category'],color='red')
plt.scatter(cluster3.CustomerID,cluster3['Country_Category'],color='black')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],color='purple',marker='*',label='centroids')
plt.xlabel('CustomerID')
plt.ylabel('Country_Category')
plt.legend()
plt.show()

#copy data from df into new dataset for hierarchical clustering
h_data=df[['Country_Category','CustomerID']].copy()
h_data.info()
h_data['CustomerID']=h_data["CustomerID"].astype('int')

#using dendogram to find optimal number of clusters
dendrogram=sch.dendrogram(sch.linkage(h_data,method='ward'))

#perform hierarchical clustering
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(h_data)

