import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import mplcursors  

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

def find_XY(df, x, y):
    filtered_df = df[(df['support'] == x) & (df['confidence'] == y)]
    rows_as_str = filtered_df[['support', 'confidence', 'antecedents', 'consequents']].to_string(index = False)
    return rows_as_str


df = pd.read_csv("AssociationRule\\daletou.csv")
rows_count = df.shape[0]

# 初始化用于求频繁项集的数据表
data = [[0 for _ in range(60)] for _ in range(rows_count)]

for index, row in df.iterrows():
    data[index][row[7]] = 1
    data[index][row[8]] = 1
    data[index][row[9]] = 1
    data[index][row[10]] = 1
    data[index][row[11]] = 1

    blue_1_transf_num = row[12] + 35     # 用[36,47]表示第1个蓝球[1,12]
    data[index][blue_1_transf_num] = 1
    blue_2_transf_num = row[13] + 47     # 用[48,59]表示第2个蓝球[1,12]
    data[index][blue_2_transf_num] = 1

# print(data)

df = pd.DataFrame(data, columns=['0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 
                                 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19',
                                 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29',
                                 'R30', 'R31', 'R32', 'R33', 'R34', 'R35', 
                                 'B1-1',  'B1-2',  'B1-3', 'B1-4', 'B1-5', 'B1-6', 'B1-7', 'B1-8',  'B1-9',  'B1-10', 'B1-11', 'B1-12', 
                                 'B2-1',  'B2-2',  'B2-3', 'B2-4', 'B2-5', 'B2-6', 'B2-7', 'B2-8',  'B2-9',  'B2-10', 'B2-11', 'B2-12']) 

# 求出频繁项集
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)  
print("======频繁项集======\n")
print(frequent_itemsets)

# 求出关联规则
# 默认用置信度来算，阈值是0.8，小于0.8的不要，此处修改为lift，小于lift为1的删除。
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
print("======关联规则======\n")
sorted_rules = rules.sort_values(by=['confidence', 'support', 'antecedents', 'consequents'], ascending=False)  
selected_columns = sorted_rules[['confidence', 'support', 'antecedents', 'consequents']]  
print(selected_columns)

# 查看一下支持度和置信度的关系,
print("======支持度和置信度的关系======\n")
plt.scatter(rules.support, rules.confidence)
plt.title("Association Rules")
plt.xlabel("support")
plt.ylabel("confidence")

# 自定义悬停时显示的信息  
cursor = mplcursors.cursor(hover=True)  
cursor.connect("add", lambda sel: sel.annotation.set_text(find_XY(rules, sel.target[0], sel.target[1])))

plt.show()


