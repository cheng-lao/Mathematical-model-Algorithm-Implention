# 使用python 对excel中的时间序列做季节性分解
import pandas as pd          
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


#对时间序列 分解的理解掌握的可以看看这个博客如何理解 http://t.csdnimg.cn/TYVPY

data = pd.read_excel(r"D:\LearningFiles\MathModel-Qingfeng\往年例题\CUMCM2023Problems\C题\6个品种的每月销售额计算.xlsx",sheet_name="茄类")
# dates = data.iloc[:,0]

dates = pd.to_datetime(data.iloc[:, 0])
amount = data.iloc[:,1].astype(float)
print(dates.head())
print(amount.head())

amounts = []
date = []
for i in range(data.shape[0]):
    amounts.append(data.iloc[i, 1])
    date.append(data.iloc[i, 0])

time_series = pd.Series(amounts, index=date)

print(time_series.head())

result = seasonal_decompose(time_series, model='additive')  #multiplicative

# 打印分解结果
print("Observed:\n", result.observed)
print("Trend:\n", result.trend)
print("Seasonal:\n", result.seasonal)
print("Residual:\n", result.resid)

result.observed.fillna(0)
result.trend.fillna(0)
result.seasonal.fillna(0)
result.resid.fillna(0)

sum = result.observed + result.trend + result.seasonal + result.resid

plt.figure(figsize=(10, 6))
plt.plot(sum, label='Sum', color='blue')
plt.plot(time_series, label='Amount', color='red')
plt.legend()
plt.title('Sum and Amount')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

result.plot()
plt.show()

