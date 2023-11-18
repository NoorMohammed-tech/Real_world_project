#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARY

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ![image.png](attachment:image.png)

# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\Lenovo\\Downloads')


# In[4]:


data = pd.read_csv("full_bank_data.csv")


# ## EDA ##

# In[5]:


data


# In[6]:


data.shape


# In[8]:


data.isnull().sum() #there is no nan value in data set


# In[10]:


(data == "unknown").sum()


# In[13]:


target_no = (data['y']=="no").sum()
print("percentage_of_No==",target_no/45211*100,"%")


# In[14]:


target_yes = (data['y']=="yes").sum()
print("percentage_of_Yes==",target_yes/45211*100,"%")


# In[15]:


data.describe(include='all').style.background_gradient(cmap='Blues')


# In[16]:


data.info()


# In[17]:


data.duplicated().sum() #there no duplication in data_set


# # DATA VISUVILAIZATION

# In[18]:


data.hist(bins=20,color='darkgreen',figsize=[20,20])


# In[19]:


def pie_plot(data, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        data[col].value_counts().plot(ax=ax, kind='pie', figsize=(25,10), fontsize=12, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize = 12)
    plt.show()


# In[20]:


pie_plot(data,[ 'job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact', 'month', 'poutcome', 'y'],2,5)


# In[21]:


def scatter_features(l):
    g = sns.PairGrid(data,y_vars="y",x_vars=data[l].columns, height=5)
    g.map(plt.scatter,color='darkgreen',alpha=0.2)
    plt.show()


# In[22]:


scatter_features(['age','balance','day','duration'])


# In[23]:


scatter_features(['campaign','pdays','previous'])


# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['y'] = le.fit_transform(data['y'])


# In[25]:


category_mean_diff = data.groupby('education')['y'].mean()

# Create a bar plot
category_mean_diff.plot(kind='bar', figsize=(12, 6),color='g')
plt.xlabel('EDUCATION')
plt.ylabel('TARGET VARIABLE')
plt.title('MEAN_VALUE_OF_DATA')
plt.xticks(rotation=90) 
plt.show()


# In[26]:


category_mean_diff = data.groupby('job')['y'].mean()

# Create a bar plot
category_mean_diff.plot(kind='bar', figsize=(12, 6),color='PINK')
plt.xlabel('JOB')
plt.ylabel('TARGET_VARIABLE')
plt.title('Mean_VALUE_OF_JOB_TARGET_VARIABLE')
plt.xticks(rotation=90) 
plt.show()


# In[27]:


plt.figure(figsize=[10, 6])
sns.barplot(x = data['month'].value_counts().values, y = data['month'].value_counts().index, palette= 'Set2')
plt.title('Top_month')
plt.xticks(rotation = 45)
sns.despine()


# In[28]:


plt.figure(figsize=[5,5])
plt.pie(data['poutcome'].value_counts().values,
        explode=(0.05, 0.05, 0.05, 0.05),
        labels= data['poutcome'].value_counts().index,
        autopct='%1.1f%%',
        shadow=False,
        startangle=90
       );


# In[30]:


catocary_job = data.groupby('job')[['balance']].sum().sort_values(by='balance', ascending=False)
catocary_job.reset_index(inplace=True)
# plot Top 10 cities with highest revenue
plt.figure(figsize=[15, 8])
sns.barplot(x = catocary_job.job[:10], y= catocary_job.balance[:10], palette= 'Set2')
plt.title('Top 10 Cities with highest Revenue', fontsize= 15)
plt.xlabel('Customer City', fontsize= 12)
plt.ylabel('Total Payments in Millions',fontsize= 12)
sns.despine()


# In[32]:


plt.figure(figsize=(8, 6))
sns.lineplot(data['day'])
plt.title('Sample Line Plot')
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.grid(True)  # Display grid
plt.show()


# In[33]:


def box_plot(num_cols):
    plt.figure(figsize=(20, 15))
    for i in range(len(num_cols)):
        if i == 16:
            break
        else:
            plt.subplot(4,4, i+1)
            l = num_cols[i]
            sns.boxplot(data[l], palette="flare")


# In[34]:


box_plot(['age','balance','day','duration','campaign','pdays','previous'])
#there huge outliers in dataset


# # LABEL_ENCODING

# In[35]:


df = data.copy()


# In[36]:


from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
ce_t= ce.TargetEncoder()
le = LabelEncoder()


# In[37]:


df = pd.get_dummies(df,columns=['loan','housing','default'])


# In[38]:


df['job'] = ce_t.fit_transform(df['job'],df['y'])
df['poutcome']= ce_t.fit_transform(df['poutcome'],df['y'])


# In[39]:


df['marital'] = le.fit_transform(df['marital'])
df['education']= le.fit_transform(df['education'])
df['contact'] = le.fit_transform(df['contact'])
df['month'] = le.fit_transform(df['month'])


# In[40]:


df


# # DATA NORMALIZATION

# In[41]:


dc = df.copy()


# In[42]:


from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
from sklearn import preprocessing


# In[43]:


ds = dc[['age','balance','day','duration','campaign','pdays','previous']].copy()


# In[44]:


scaler = preprocessing.RobustScaler()
robust = scaler.fit_transform(ds)
robust = pd.DataFrame(robust)


# In[45]:


scaler_m  = preprocessing.MinMaxScaler()
min_max = scaler_m.fit_transform(ds)
min_max = pd.DataFrame(min_max)


# In[46]:


scaler_s = preprocessing.StandardScaler()
standard= scaler_s.fit_transform(ds)
standard = pd.DataFrame(standard)
standard


# In[47]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols = 4, figsize =(20, 10))
ax1.set_title('Before Scaling')
sns.kdeplot(ds, ax = ax1, color ='b')
ax2.set_title('After Robust')

sns.kdeplot(robust, ax = ax2, color ='g')
ax3.set_title('After Standard Scaling')

sns.kdeplot(standard, ax = ax3, color ='b')
ax4.set_title('After Min-Max Scaling')

sns.kdeplot(min_max, ax = ax4, color ='g')
plt.show()
# after visualizing that min-max Scaling is Better


# In[48]:


def min_maxScaling(df_num, cols):
    scaler = preprocessing.MinMaxScaler()
    min_max = scaler_s.fit_transform(ds)
    min_max = pd.DataFrame(min_max, columns =cols)
    return min_max


# In[49]:


min_max_s = min_maxScaling(ds,['age','balance','day','duration','campaign','pdays','previous'])


# In[51]:


clean_ds = dc.copy()
clean_ds.drop(labels=['age','balance','day','duration','campaign','pdays','previous'],axis=1,inplace=True)
clean_ds[['age','balance','day','duration','campaign','pdays','previous']]=min_max_s[['age','balance','day','duration','campaign','pdays','previous']]


# In[52]:


clean_ds


# # MODEL SAMPLING

# In[53]:


ds1 = clean_ds[0:6459]
ds2 = clean_ds[6460:12918]
ds3 = clean_ds[12919:19376]
ds4 = clean_ds[19377:25834]
ds5 = clean_ds[25835:32292]
ds6 =  clean_ds[32293:38750]
ds7 = clean_ds[38751:45211]
ds7


# In[54]:


x1 = ds1.copy()
x1.drop(columns=['y'],axis=1,inplace=True)


# In[55]:


y1 = ds1.iloc[:,[6]]


# In[56]:


x2 = ds2.copy()
x2.drop(columns=['y'],axis=1,inplace=True)
y2 = ds2.iloc[:,[6]]


# In[57]:


x3 = ds3.copy()
x3.drop(columns=['y'],axis=1,inplace=True)
y3 = ds3.iloc[:,[6]]


# In[58]:


x4 = ds4.copy()
x4.drop(columns=['y'],axis=1,inplace=True)
y4 = ds4.iloc[:,[6]]


# In[59]:


x5 = ds5.copy()
x5.drop(columns=['y'],axis=1,inplace=True)
y5 = ds5.iloc[:,[6]]


# In[60]:


x6 = ds6.copy()
x6.drop(columns=['y'],axis=1,inplace=True)
y6 = ds6.iloc[:,[6]]


# In[61]:


x7 = ds7.copy()
x7.drop(columns=['y'],axis=1,inplace=True)
y7 = ds7.iloc[:,[6]]


# # MODELING

# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,precision_score,recall_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import multilabel_confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[63]:


x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.2,random_state=9)


# In[64]:


def model_evaluation(model,name,x_test,y_test):
    report = classification_report(y_test,model.predict(x_test),zero_division=1)
    accuracy_scor = accuracy_score(model.predict(x_test),y_test)
    print(report)
    print(name,"accuracy_score=",accuracy_scor*100)


# In[65]:


def conf_matrixs(y_test,model,x_test,cmap,normalize=None,plot=True,encoded_labels=True):
    y_pred = model.predict(x_test)
    conf_mat = confusion_matrix(y_test,y_pred,normalize=None,labels=[0,1])   
    ax = sns.heatmap(conf_mat, cmap=cmap, square=True, cbar=False, annot=True, fmt='g')
    ax.set_title('confution_matrics')
    ax.set_xlabel('prediction')
    ax.set_ylabel('Actual')
    return conf_mat


# In[66]:


def plot_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    df = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
    plt.figure(figsize=(7,5))
    plt.plot(df[:33])
    plt.legend(['Actual' , 'Predicted'])


# In[67]:


from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

def custom_roc_curve(model_name, x_train_label, x_test_label, y_train_label, y_test_label):
    # Assuming you have already trained and fit the logistic regression model (model_kt)
    score = {}
    
    # Predict probabilities on the training and test sets
    y_probs_train = model_name.predict_proba(x_train_label)[:, 1]
    y_probs_test = model_name.predict_proba(x_test_label)[:, 1]

    # Predictions on the training and test sets
    y_predicted_train = model_name.predict(x_train_label)
    y_predicted_test = model_name.predict(x_test_label)

    # Calculate AUC and Accuracy
    train_auc = roc_auc_score(y_train_label, y_probs_train)
    test_auc = roc_auc_score(y_test_label, y_probs_test)
    train_acc = accuracy_score(y_train_label, y_predicted_train)
    test_acc = accuracy_score(y_test_label, y_predicted_test)

    print('*' * 50)
    print('Train AUC: %.3f' % train_auc)
    print('Test AUC: %.3f' % test_auc)
    print('*' * 50)
    print('Train Accuracy: %.3f' % train_acc)
    print('Test Accuracy: %.3f' % test_acc)

    # Store the results in a dictionary or any other data structure as needed
    score['Logistic Regression'] = [test_auc, test_acc]

    # Calculate ROC curve
    train_fpr, train_tpr, train_thresholds = roc_curve(y_train_label, y_probs_train)
    test_fpr, test_tpr, test_thresholds = roc_curve(y_test_label, y_probs_test)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(train_fpr, train_tpr, marker='.', label='Train AUC')
    plt.plot(test_fpr, test_tpr, marker='.', label='Test AUC')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()


# # K-NEAREST NEABOUR

# ITERATION 1

# In[68]:


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x_train,y_train)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[69]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


# In[74]:


y2 = y2.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x2,y2)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[80]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


# In[75]:


y3 = y3.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x3,y3)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[81]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


# In[76]:


y4 = y4.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x4,y4)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[82]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


# In[77]:


y5 = y5.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x5,y5)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[83]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


# In[78]:


y6 = y6.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x6,y6)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[84]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


# In[79]:


y7 = y7.values.ravel()
model_k1 = KNeighborsClassifier(n_neighbors=5,p=2)
model_k1.fit(x7,y7)
model_evaluation(model_k1,"KNeighborsClassifier",x_test,y_test)
conf_matrixs(y_test,model_k1,x_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[85]:


custom_roc_curve(model_k1,x_train,x_test,y_train,y_test)


#  # NAIVE BAYES   

# ITERATION 1

# In[86]:


model_g1 = GaussianNB()
model_g1.fit(x_train,y_train)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[71]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# ITERATION 2

# In[89]:


model_g1 = GaussianNB()
model_g1.fit(x2,y2)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[90]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# ITERATION 3

# In[91]:


model_g1 = GaussianNB()
model_g1.fit(x3,y3)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[92]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# ITERATION 4

# In[93]:


model_g1 = GaussianNB()
model_g1.fit(x4,y4)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[94]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# ITERATION 5

# In[95]:


model_g1 = GaussianNB()
model_g1.fit(x5,y5)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[96]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# ITERATION 6

# In[97]:


model_g1 = GaussianNB()
model_g1.fit(x6,y6)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[98]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# ITERATION 7

# In[99]:


model_g1 = GaussianNB()
model_g1.fit(x7,y7)
model_evaluation(model_g1," GaussianNB",x_test,y_test)
conf_matrixs(y_test,model_g1,x_test,cmap='Blues',normalize=None,plot=True,encoded_labels=True)


# In[100]:


custom_roc_curve(model_g1,x_train,x_test,y_train,y_test)


# # SUPPORT VECTOR MACHINE

# ITERATION 1

# In[101]:


model_s1 = SVC()
model_s1.fit(x_train,y_train)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# ITERATION 2

# In[104]:


model_s1 = SVC()
model_s1.fit(x2,y2)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# ITERATION 3

# In[105]:


model_s1 = SVC()
model_s1.fit(x3,y3)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# ITERATION 4

# In[106]:


model_s1 = SVC()
model_s1.fit(x4,y4)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# ITERATION 5

# In[107]:


model_s1 = SVC()
model_s1.fit(x5,y5)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# ITERATION 6

# In[108]:


model_s1 = SVC()
model_s1.fit(x6,y6)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# ITERATION 7

# In[109]:


model_s1 = SVC()
model_s1.fit(x7,y7)
model_evaluation(model_s1,"SVM",x_test,y_test)
conf_matrixs(y_test,model_s1,x_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# # LOGESTIC REGRESSION

# ITERATION 1

# In[111]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x_train,y_train)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[112]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# ITERATION 2

# In[114]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x2,y2)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[115]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# ITERATION 3

# In[116]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x3,y3)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[117]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# ITERATION 4

# In[118]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x4,y4)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[119]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# ITERATION 5

# In[120]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x5,y5)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[121]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# ITERATION 6

# In[122]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x6,y6)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[123]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# ITERATION 7

# In[124]:


model_l1 =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_l1.fit(x7,y7)
model_evaluation(model_l1,"LOGESTIC REGRESSION",x_test,y_test)
conf_matrixs(y_test,model_l1,x_test,cmap="Greens",normalize=None,plot=True,encoded_labels=True)


# In[125]:


custom_roc_curve(model_l1,x_train,x_test,y_train,y_test)


# # RANDOM SAMPLE TESTING

# In[127]:


test_x = clean_ds.copy()


# In[128]:


test_x.drop(columns=['y'],axis=1,inplace=True)


# In[129]:


test_y = clean_ds.iloc[:,[6]]


# In[130]:


x_train_t,x_test_t,y_train_t,y_test_t = train_test_split(test_x,test_y,test_size=0.2,random_state=127)


# # ITERATION MODEL TESTING¶

# KNN

# In[131]:


model_evaluation(model_k1,"KNN",x_test_t,y_test_t)
conf_matrixs(y_test_t,model_k1,x_test_t,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[132]:


custom_roc_curve(model_k1,x_train_t,x_test_t,y_train_t,y_test_t)


# NAIVE BAYES

# In[133]:


model_evaluation(model_g1," GaussianNB",x_test_t,y_test_t)
conf_matrixs(y_test_t,model_g1,x_test_t,cmap='Greens',normalize=None,plot=True,encoded_labels=True)


# In[134]:


custom_roc_curve(model_g1,x_train_t,x_test_t,y_train_t,y_test_t)


# SVM

# In[135]:


model_evaluation(model_s1,"SVM",x_test_t,y_test_t)
conf_matrixs(y_test_t,model_s1,x_test_t,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# LOGESTIC REGRESSION

# In[137]:


model_evaluation(model_l1,"LOGESTIC_REGRESSION_TESTING",x_test_t,y_test_t)
conf_matrixs(y_test_t,model_l1,x_test_t,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# In[138]:


custom_roc_curve(model_l1,x_train_t,x_test_t,y_train_t,y_test_t)


# # TOTAL SAMPLE TESTING¶

# In[139]:


dt  = clean_ds.copy()


# In[140]:


dt.drop(columns=['y'],axis= 1,inplace=True)
y_t = clean_ds.iloc[:,[6]]


# In[141]:


X_train,X_test,Y_train,Y_test = train_test_split(dt,y_t,test_size=0.2,random_state=127)


# KNN

# In[142]:


Y_train = Y_train.values.ravel()
Y_test = Y_test.values.ravel()
model_kt = KNeighborsClassifier(n_neighbors=5,p=2)
model_kt.fit(X_train,Y_train)
model_evaluation(model_kt,"KNeighborsClassifier",X_test,Y_test)
conf_matrixs(Y_test,model_kt,X_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[143]:


custom_roc_curve(model_kt,X_train,X_test,Y_train,Y_test)


# NAIVE BAYES

# In[144]:


model_gt = GaussianNB()
model_gt.fit(X_train,Y_train)
model_evaluation(model_gt," GaussianNB",X_test,Y_test)
conf_matrixs(Y_test,model_gt,X_test,cmap='Reds',normalize=None,plot=True,encoded_labels=True)


# In[145]:


custom_roc_curve(model_gt,X_train,X_test,Y_train,Y_test)


# SVM

# In[146]:


model_st = SVC()
model_st.fit(X_train,Y_train)
model_evaluation(model_st,"SVM",X_test,Y_test)
conf_matrixs(Y_test,model_st,X_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# LOGESTIC REGRESSION

# In[147]:


model_lt =  LogisticRegression(solver='lbfgs',max_iter=1000)
model_lt.fit(X_train,Y_train)
model_evaluation(model_lt,"LOGESTIC REGRESSION",X_test,Y_test)
conf_matrixs(Y_test,model_lt,X_test,cmap='GnBu',normalize=None,plot=True,encoded_labels=True)


# In[148]:


custom_roc_curve(model_lt,X_train,X_test,Y_train,Y_test)


# # CONCLUTION
# ## 1.Summary of Findings:
# Begin by summarizing the key findings from your data analysis. Highlight any significant trends, patterns, or insights that you discovered through your visualizations
# 
# ## 2.Visualization Impact:
# Discuss the impact of your visualizations on the overall understanding of the data. Emphasize how visual representations helped in conveying complex information and making the findings more accessible
# 
# ## 3.Rationale for Sample Separation:
# Explain why you chose to separate your data into seven samples. Was it based on specific criteria, such as time periods, geographical locations, or other relevant factors? Clarify the rationale behind this decision
# 
# ## 4.Key Findings Across Samples:
# Summarize the key findings from each of the seven samples. Highlight any commonalities or differences observed across the samples. This can provide insights into the variability or consistency of your results
# 
# ## 5.Comparative Analysis:
# Discuss how the separation into seven samples allowed you to perform a comparative analysis. Did you identify trends or patterns that were consistent across multiple samples? Did certain samples exhibit unique characteristics
# 
# ## V6.Statistical Significance:
# If applicable, discuss the statistical significance of your findings. Are the patterns observed in each sample statistically significant, and what does this mean for the overall reliability of your results
# 
# ## 7.Iterative Modeling Approach:
# Explain the rationale behind the iterative fitting of your model. Did you refine parameters, features, or other aspects of the model in each iteration? Detail any adjustments made to enhance model performance.## 8.Evolution of ROC-AUC Curves: Discuss how the ROC-AUC curves evolved throughout the iterations. Highlight any improvements or changes in model performance. Did you observe convergence or stabilization of the curves over time
# 
# ## 9.Performance Metrics:
# Besides ROC-AUC, mention other relevant performance metrics you considered during the iterative process. This could include accuracy, precision, recall, or F1 score. Discuss how these metrics influenced your decisions in each iteration
# 
# ## 10.Decision-Making Criteria:
# Explain the criteria you used to make decisions about model adjustments. Were there specific thresholds or benchmarks you aimed to achieve in terms of ROC-AUC or other metrics? Clarify how these criteria guided your model refinement
# 
# ## .Challenges and Learnings:
# Discuss any challenges encountered during the iterative process. This could include issues related to overfitting, underfitting, or data-specific challenges. Highlight what you learned from each iteration and how it influenced subsequent model adjustments.## 12.Future Model Refinement: Provide insights into potential future directions for model refinement. Are there additional features or data preprocessing steps that could further enhance performance? Consider any avenues for ongoing improvement. ..?? ??....
# 
# 

# In[ ]:




