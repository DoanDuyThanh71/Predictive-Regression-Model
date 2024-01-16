import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
# import data 
df = pd.read_csv('data.csv', encoding='latin1')
# print(df.head(), df.tail())

# Truc quan hoa describe
des = df.describe()
# print(des)

# Loại bỏ các cột không cần thiết cho phân tích, bao gồm STATUS', 'CITY','CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER', 'PRODUCTCODE', "ADDRESSLINE2","STATE","TERRITORY", "PHONE","ADDRESSLINE1","POSTALCODE", "ORDERDATE"
drops = ['STATUS', 'CITY','CONTACTFIRSTNAME','CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER', 'PRODUCTCODE', "ADDRESSLINE2","STATE","TERRITORY", "PHONE","ADDRESSLINE1","POSTALCODE", "ORDERDATE"]
df = df.drop(drops,axis=1)

# chuan hoa du lieu

# Chuẩn hóa PRODUCTLINE
le = LabelEncoder()
df['PRODUCTLINE'] = le.fit_transform(df['PRODUCTLINE'])

# Chuẩn hóa COUNTRY
df['COUNTRY'] = le.fit_transform(df['COUNTRY'])

# Chuẩn hóa DEALSIZE
df['DEALSIZE'] = le.fit_transform(df['DEALSIZE'])

# Chuẩn hóa về cùng độ đo cho các cột
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

correlation_matrix = df_scaled.corr()

plt.figure(figsize=(10, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Danh sách cột có thể sử dụng
significant_columns = []

# Danh sách cột cần loại bỏ
columns_to_remove = []

for column in df_scaled.columns:
    slope, intercept, r_value, p_value, std_err = linregress(df_scaled[column], df_scaled['SALES'])
    if p_value < 0.05:
        significant_columns.append(column)
    else:
      columns_to_remove.append(column)


df_clear = df.drop(columns=columns_to_remove, axis=1)
df_scaled = df_scaled.drop(columns=columns_to_remove, axis=1)
# print(significant_columns)
df_clear.to_csv('cleaned_data.csv')
print(df_clear.info())
