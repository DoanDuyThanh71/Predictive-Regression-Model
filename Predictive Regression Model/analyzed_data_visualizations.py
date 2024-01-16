import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_scaled = pd.read_csv("cleaned_data.csv", encoding='latin1')
# Thiết lập kiểu cho Seaborn
sns.set(style="whitegrid")

# Kích thước của biểu đồ
plt.figure(figsize=(15, 12))


columns_to_plot = df_scaled.columns[df_scaled.columns != 'SALES']

# Duyệt qua từng cột và vẽ biểu đồ phân tán so với 'SALES' và in phương trình đường tuyến tính
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=column, y='SALES', data=df_scaled)

    # Tính hệ số tỷ lệ (slope) và chệch (intercept) của đường tuyến tính
    slope, intercept = np.polyfit(df_scaled[column], df_scaled['SALES'], 1)

    # In phương trình đường tuyến tính lên biểu đồ
    plt.plot(df_scaled[column], slope * df_scaled[column] + intercept, color='red', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
    plt.legend()

    plt.title(f'Scatterplot: {column} vs SALES')

# Tự động điều chỉnh khoảng cách giữa subplot
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()



# Thiết lập kiểu cho Seaborn
sns.set(style="whitegrid")

# Số lượng cột
num_cols = len(df_scaled.columns)

# Kích thước của biểu đồ
plt.figure(figsize=(15, 12))

# Duyệt qua từng cột và vẽ biểu đồ boxplot
for i, column in enumerate(df_scaled.columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=df_scaled[column])
    plt.title(f'Boxplot: {column}')

# Tự động điều chỉnh khoảng cách giữa subplot
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()
