from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

df_scaled = pd.read_csv("cleaned_data.csv", encoding='latin1')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = df_scaled.drop('SALES', axis=1)
y = df_scaled['SALES']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test.to_csv('test.csv')

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

import joblib

# Lưu mô hình vào file
model_filename = 'linear_model.joblib'
joblib.dump(model, model_filename)

# Load mô hình từ file
loaded_model = joblib.load(model_filename)

# Đường dẫn đến file model.joblib
model_path = "linear_model.joblib"

# Tải mô hình từ file
loaded_model = joblib.load(model_path)

# Hiển thị thông tin của mô hình
print("Model Information:")
print(loaded_model)

# Lấy hệ số của mô hình
coefficients = loaded_model.coef_

# Lấy chệch của mô hình
intercept = loaded_model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)