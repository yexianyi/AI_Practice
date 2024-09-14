import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

# 加载数据
df = pd.read_csv("./AssociationRule/shuangseqiu.csv")
# values = df['O1'].values  
# unique_values, counts = np.unique(values, return_counts=True)  
  
# 初始化一个空的DataFrame来存储结果  
result_df = pd.DataFrame(index=range(1, 34))  
  
# 对每一列进行处理  
for col in df.columns:  
    # 获取该列的计数  
    counts = df[col].value_counts().reindex(range(1, 34), fill_value=0)  
    # 将计数结果赋值给结果DataFrame的对应列  
    result_df[col] = counts  

print("每个号码的红色和蓝球分别出现在第1位到第6位的次数：")
result_df = result_df[['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'B1']]
print(result_df)