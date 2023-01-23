#!/usr/bin/env python
# coding: utf-8

# In[8]:


#What are the top two to four factors that people should base their decisions 
#on when buying a property in Melbourne within the next three months? 

import pandas as pd 
import numpy as np
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set_style("darkgrid")
mpl.rcParams['figure.figsize'] = (20,5)

Melbourne= pd.read_csv("Melbourne.csv")

Melbourne.count()
Melbourne.countblank()


# In[2]:


Melbourne1=Melbourne.dropna(inplace=False)
print(Melbourne1)


# In[3]:


print(Melbourne.describe())
print(Melbourne.info())


# In[4]:


Melbourne1.columns


# In[5]:


Melbourne2=Melbourne1.select_dtypes(include=np.number)
print(Melbourne2)
Melbourne2.plot(kind="box")


# In[6]:


#Calculating interquartile range

Q1=Melbourne2.quantile(.25,axis=0)
Q3=Melbourne2.quantile(.75,axis=0)
IQR=Q3-Q1
print(IQR)
IQR_2Decimal=round(IQR,2)
print(IQR_2Decimal)


# In[7]:


#Boxplot that only shows outliers

Lower_Limit=Q1-1.5*IQR_2Decimal
Upper_Limit=Q3+1.5*IQR_2Decimal
Outliers=Melbourne2[((Melbourne2<Lower_Limit)|(Melbourne2>Upper_Limit)).any(axis=1)]
Outliers.plot(kind="box")
plt.title("Outliers Included")


# In[8]:


#Data with No outliers calculated and its boxplot 

No_Outliers=Melbourne2[~((Melbourne2<Lower_Limit)|(Melbourne2>Upper_Limit)).any(axis=1)]
print(No_Outliers)
No_Outliers.plot(kind="box")
plt.title("Outliers Excluded")


# In[9]:


No_Outliers.columns


# In[10]:


#Boxplots without outliers

No_Outliers.plot(kind="box")
plt.title("All Variables")
plt.show()

#It doesn't make sense to keep Longitude, Lattitude, YearBuilt and Postcode in boxplots becauese they are not numerical.
#So, I removed them.I also separated Price, Landsize, BuildingArea and Properycount from the rest of the variables
#because they have a lot more information within them.

No_Outliers[["Rooms","Distance", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "Propertycount"]].plot(kind="box")
plt.title("Price Excluded")
plt.show()

No_Outliers[["Rooms","Distance", "Bedroom2", "Bathroom", "Car"]].plot(kind="box")
plt.title("Five Variables")
plt.show()

No_Outliers[["BuildingArea","Landsize"]].plot(kind="box")
plt.title("Two Variables")
plt.show()

No_Outliers["Price"].plot(kind="box")
plt.title("Price")
plt.show()

No_Outliers["Propertycount"].plot(kind="box")
plt.title("Price")
plt.show()





# In[11]:


#Getting my data's mean
#"Rooms","Bedroom2","Bathroom","Car"
# "Price","Landsize","BuildingArea","YearBuilt","Propertycount"

A=round(No_Outliers.Price.mean(),2)
B=round(No_Outliers.Rooms.mean(),2)
C=round(No_Outliers.Bedroom2.mean(),2)
D=round(No_Outliers.Bathroom.mean(),2)
E=round(No_Outliers.Car.mean(),2)
F=round(No_Outliers.Landsize.mean(),2)
G=round(No_Outliers.BuildingArea.mean(),2)
H=round(No_Outliers.Propertycount.mean(),2)

list=[A,B,C,D,E,F,G,H]

print(list)


# In[12]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["Rooms"], color="blue",alpha=.2, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("Rooms")
plt.show()


# In[13]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["Bedroom2"], color="blue",alpha=.2, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("Bedroom2")
plt.show()


# In[14]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["Bathroom"], color="blue",alpha=.1, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("Bathroom")
plt.show()


# In[15]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["Car"], color="blue",alpha=.1, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("Car")
plt.show()


# In[16]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["Landsize"], color="blue", alpha=.2, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("Landsize")
plt.show()


# In[17]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["BuildingArea"], color="blue", alpha=.2, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("BuildingArea")
plt.show()


# In[18]:


fig, ax = plt.subplots()
ax.scatter(No_Outliers["Price"],No_Outliers["Propertycount"], color="blue", alpha=.3, label="Variables")
ax.legend()
ax.set_xlabel("Price")
ax.set_ylabel("Propertycount")
plt.show()


# In[19]:


#Please ignore this cell

import pandas as pd 

A=round(No_Outliers.Price.mean(),2)
B=round(No_Outliers.Rooms.mean(),2)
C=round(No_Outliers.Bedroom2.mean(),2)
D=round(No_Outliers.Bathroom.mean(),2)
E=round(No_Outliers.Car.mean(),2)
F=round(No_Outliers.Landsize.mean(),2)
G=round(No_Outliers.BuildingArea.mean(),2)
H=round(No_Outliers.Propertycount.mean(),2)

list=[A,B,C,D,E,F,G,H]

data={'list':['A','B','C','D','E','F','G','H']}

df=pd.DataFrame(data)

print(df)


# In[23]:


df=No_Outliers.drop(['Lattitude','Longtitude','Distance','YearBuilt'],axis=1).corr()
df["Price"].sort_values().plot(kind="bar")


# In[26]:


df2=No_Outliers.drop(['Lattitude','Longtitude','Distance','YearBuilt'],axis=1).corr()
sns.heatmap(df2.corr(),cmap="YlGnBu")


# In[99]:


#Multivariate Regression Model/should be without outliers 

independent_variable=No_Outliers.drop(['Price','Lattitude', 'Postcode','YearBuilt','Longtitude','Propertycount'],axis=1)
independent_variable=sm.add_constant(independent_variable)
dependent_variable=No_Outliers['Price']
regression_model=sm.OLS(dependent_variable,independent_variable).fit()
Melbourne3=regression_model.summary()
print(Melbourne3)

