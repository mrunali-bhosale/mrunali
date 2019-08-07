#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
data = [train_df,test_df]

#checking for null values
train_df.isnull().sum()
test_df.isnull().sum()

#percentage of null values
null_values = pd.DataFrame(train_df.isnull().sum(),columns=['total_count'])
round(null_values.loc[null_values['total_count'] > 0 , : ]['total_count']['Gender'] / len(train_df),2) * 100

#dropping column containing null values
train_df.drop(['Gender'],axis=1,inplace = True)
test_df.drop(['Gender'],axis=1,inplace = True)

#Exploratory data analysis
def bivariate_analysis_categorical(dataframe,target):
    cols = list(train_df)
    for col in cols:
        if col in dataframe.select_dtypes(exclude=np.number).columns:
            sns.countplot(x=dataframe[col],hue=target,data=dataframe)
            plt.xticks(rotation='vertical')
            plt.show()
bivariate_analysis_categorical(train_df,train_df['Claim']) 

sns.kdeplot(train_df.loc[(train_df['Claim']==0),'Net Sales'],color='r',shade='True',Label='Not Claimed',legend = True)
sns.kdeplot(train_df.loc[(train_df['Claim']==1),'Net Sales'],color='b',shade='True',Label='Claimed',legend = True)
          

train_df['Agency'].unique()
train_df['Agency'].nunique()
train_df['Agency Type'].unique()
train_df['Distribution Channel'].unique()
train_df['Product Name'].nunique()
train_df['Product Name'].value_counts()
train_df['Destination'].nunique()
x=train_df['Destination'].value_counts()
train_df.groupby('Claim').mean()
train_df['Age'].max()
train_df['Duration'].value_counts()

for dataset in data:
    dataset.loc[(dataset['Age']>0)&(dataset['Age']<=20),'Age']=0
    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=40),'Age']=1
    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=60),'Age']=2
    dataset.loc[(dataset['Age']>60)&(dataset['Age']<=80),'Age']=3
    dataset.loc[(dataset['Age']>80)&(dataset['Age']<=100),'Age']=4
    dataset.loc[(dataset['Age']>100)&(dataset['Age']<=120),'Age']=5

def duration_months(size):
    a = ' '
    if (size >5)&(size <=90):
        a = '1-3 months'
    elif(size>90)&(size<=180):
        a = '4-6 months'
    elif(size>180)&(size<=270):
        a = '7-9 months'
    elif(size>270)&(size<=360):
        a = '10-12 months'
    elif(size>360)&(size<=450):
        a = '13-15 months'        
    elif(size>450)&(size<=540):
        a = '16-19months'
    elif(size>540)&(size<=630):
        a = '20-23 months'
    else:
        a = 'more than 2 years'
    return a

for dataset in data:
    dataset['Duration']=dataset.Duration.map(duration_months)

train_df['Destination'].value_counts()
#to detect outliers
sns.boxplot(x='Claim',y='Duration',data=train_df)
sns.boxplot(x='Claim',y='Age',data=train_df)

#top 10 entries of Destination
n = 10
train_df['Destination'].value_counts().keys()[:n].tolist()
train_df['Destination'].value_counts().idxmax()

#mapping top 10 entries of destination
dest_mapping={'SINGAPORE':'A','MALAYSIA':'B','THAILAND':'C','CHINA':'D','AUSTRALIA':'E','INDONESIA':'F','UNITED STATES':'G','PHILIPPINES':'H','HONG KONG':'I','INDIA':'J'}
train_df['Destination']=train_df['Destination'].map(dest_mapping)
train_df.Destination.fillna('K',inplace = True)

#checking class imbalace
train_df.Claim.value_counts()

#splitting training dataset
X=train_df.drop('Claim',axis=1)
y=train_df['Claim']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=0)

#class imbalace - oversampling of majority class
from sklearn.utils import resample
trained = pd.concat([X_train,y_train],axis=1)
not_claimed_majority = pd.DataFrame(trained[trained.Claim == 0])
claimed_minority = pd.DataFrame(trained[trained.Claim == 1])
claim_0, claim_1 = trained.Claim.value_counts()
claim_upsampled = resample(claimed_minority,replace = True ,n_samples = claim_0,random_state = 27)
upsampled = pd.concat([not_claimed_majority,claim_upsampled])
upsampled.Claim.value_counts()

#encoding
upsampled= pd.get_dummies(data=upsampled,columns=['Agency','Agency Type','Distribution Channel','Product Name','Destination','Duration'])
up_sampled=pd.DataFrame(upsampled)
up_sampled.to_csv('clean.csv',index=False)
upsampled= pd.read_csv('clean.csv')
X=upsampled.drop('Claim',axis=1)
y=upsampled['Claim']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=0)


#undersampling majority class
not_claimed_downsampled = resample(not_claimed_majority,replace = False,n_samples = claim_1,random_state=27)
downsampled = pd.concat([claimed_minority,not_claimed_downsampled])
downsampled.Claim.value_counts()

#encoding
downsampled= pd.get_dummies(data=downsampled,columns=['Agency','Agency Type','Distribution Channel','Product Name','Destination','Duration'])
X=downsampled.drop('Claim',axis=1)
y=downsampled['Claim']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
seed = 7
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f " % (name, cv_results.mean())
	print(msg)
    
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import precision_score
precision_score(y_test,y_pred)


