import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


warnings.filterwarnings("ignore")
df = pd.read_csv("shuangseqiu.csv")
rows_count = df.shape[0]

# 初始化用于求频繁项集的数据表
data = [[0 for _ in range(34)] for _ in range(rows_count)]

for index, row in df.iterrows():
    data[index][row[8]] = 1
    data[index][row[9]] = 1
    data[index][row[10]] = 1
    data[index][row[11]] = 1
    data[index][row[12]] = 1
    data[index][row[13]] = 1

# print(data)

df = pd.DataFrame(data, columns=['0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 
                                 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19',
                                 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R66', 'R27', 'R28', 'R29',
                                 'R30', 'R31', 'R32', 'R33']) 

# 求出频繁项集
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
print("======频繁项集======\n")
print(frequent_itemsets)

# 求出关联规则
# 默认用置信度来算，阈值是0.8，小于0.8的不要，此处修改为lift，小于lift为1的删除。
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("======关联规则======\n")
print(rules)
# 查看一下支持度和置信度的关系,
print("======支持度和置信度的关系======\n")
plt.scatter(rules.support, rules.confidence)
plt.title("Association Rules")
plt.xlabel("support")
plt.ylabel("confidence")
