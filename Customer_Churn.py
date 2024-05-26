# Importing python's pandas library
# Importing python's numpy library
# Importing python's submodule from matplotlib library

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  

df = pd.read_csv('Customer_Churn.csv')
print(df.sample(5)) #It will return random 5 rows from a data frame.
print(df.shape) #It will return the no. of rows and columns present in a DF.

df.drop('customerID',axis='columns') # Here 'customerID' is being deleted as it's of no use
print(df.dtypes) #Here we see MonthlyCharges is float64 & TotalCharges is object, and we need to fix this.

print(df.TotalCharges.values) # It will return all the values from column TotalCharges in the form of an array.

print(df.MonthlyCharges.values) # It will return all the values from column MonthlyCharges in the form of an array.

print(pd.to_numeric(df.TotalCharges,errors="coerce").isnull()) # pd.to_numeric will convert all the rows to numbers by ignoring errors if any with the help of errors="coerce".
                                                               #This will return a numpy series for each row by checking if the value is null or not.

print(df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()]) # return a DF with all those rows where total charges are blank

print(df[pd.to_numeric(df.TotalCharges,errors="coerce").isnull()].shape) # checking how many rows are there where TotalCharges is null, here there are 11 such cases.

df1 = df[df.TotalCharges != ' '] # here we are making a new data frame(df1) by removing those rows where the space is not present.
print(df1.shape) 

print(df1.dtypes)

print(pd.to_numeric(df1.TotalCharges))

print(df1.shape)

print(df1.TotalCharges.dtype)

Tenure_Churn_No = df1[df1.Churn == 'No'].tenure 
print(Tenure_Churn_No)

Tenure_Churn_Yes = df1[df1.Churn == 'Yes'].tenure
print(Tenure_Churn_Yes)

plt.hist([Tenure_Churn_Yes,Tenure_Churn_No], color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend()
plt.xlabel('Tenure in months')
plt.ylabel('Number of customers')
plt.title('Customer churn prediction')
plt.show()


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges
mc_churn_no
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges
mc_churn_yes
plt.hist([mc_churn_yes,mc_churn_no], color=['green','red'], label=['Churn=Yes','Churn=No'])
plt.legend()
plt.xlabel('Monthy charges')
plt.ylabel('Number of customers')
plt.title('Customer churn prediction')
plt.show()


for columns in df:
  if df[columns].dtypes == 'object':
    print(df[columns].unique()) # prints unique value in all the columns
    print(f'{columns} : {df[columns].unique()}') 

for columns in df:
  if df[columns].dtypes == 'object':
    print(f'{columns} : {df[columns].unique()}') # prints unique value in along with columns names    

print(df1.head(100))

df2 = df1.replace(['No internet service'],['No'])
df2 = df1.replace(['No phone service'],['No'])

for columns in df1:
    if df1[columns].dtypes == 'object':
        print(f'{columns} : {df1[columns].unique()}') # prints unique value in along with columns names

def print_unique_col_values(df):
  for column in df:
    if df[column].dtypes=='object':
      print(f'{column}: {df[column].unique()}')

print_unique_col_values(df1)

yes_no_columns = ["Partner","Dependents","PhoneService","MultipleLines",
                 "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                 "StreamingMovies","PaperlessBilling","Churn"]


for col in yes_no_columns:
  df1[col].replace({"Yes":1,"No":0}) # "yes" is changed to 1, "No" is changed to 0

print(df1['gender'].replace({"Female":1,"Male":0}))  

print(df1['gender'].unique())

df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
print(df2.columns)

print(df2)

yes_no_to_one_and_zero = ["Contract_Month-to-month","Contract_One year","Contract_Two year","PaymentMethod_Bank transfer (automatic)","PaymentMethod_Credit card (automatic)","PaymentMethod_Electronic check","PaymentMethod_Mailed check","InternetService_DSL","InternetService_Fiber optic","InternetService_No"]

for col in yes_no_to_one_and_zero:
  df2[col].replace({"True":1,"False":0},inplace=True) # "True" is changed to 1, "False" is changed to 0

df2['Contract_Month-to-month'].replace({'True':1,'False':0})
print(df2.dtypes)