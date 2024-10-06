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

df = pd.read_csv("AssociationRule\shuangseqiu.csv")


# # 求R1~R33作为第一个被抓取到红球的频率
# red_values = [f'R{i+1}' for i in range(33)]  
# count_values = [0] * 33  
# data = {'Red': red_values, 'Count': count_values} 
# red_fre_stat = pd.DataFrame(data) 
# red_counts = df['O1'].value_counts(ascending=True)  
# print("======第一个被抓取到红球的频率(从小到大):======")
# print(red_counts)


rows_count = df.shape[0]
# 初始化用于求频繁项集的数据表
data = [[0 for _ in range(50)] for _ in range(rows_count)]

for index, row in df.iterrows():
    data[index][row[8]] = 1
    data[index][row[9]] = 1
    data[index][row[10]] = 1
    data[index][row[11]] = 1
    data[index][row[12]] = 1
    data[index][row[13]] = 1

    blue_transf_num = row[14] + 33     # 用[34,49]表示蓝球[1,16]
    data[index][blue_transf_num] = 1

# print(data)

df = pd.DataFrame(data, columns=['0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 
                                 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19',
                                 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29',
                                 'R30', 'R31', 'R32', 'R33', 'B1',  'B2',  'B3', 'B4', 'B5', 'B6', 'B7',
                                 'B8',  'B9',  'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16']) 

# 求出频繁项集
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
print("======频繁项集======")
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)  
frequent_itemsets['Rank'] = frequent_itemsets['support'].rank(method='min', ascending=False)
print(frequent_itemsets)

rows_of_top_10_single_num = frequent_itemsets.nlargest(10, 'support')  
print("======支持度最高的前10个数字, 作为初始的红球预选号码======")
print(rows_of_top_10_single_num)


red_values = [f'R{i+1}' for i in range(33)]  
count_values = [0] * 33  
data = {'Red': red_values, 'Count': count_values} 
red_fre_stat = pd.DataFrame(data) 

for i in range(1, 34):
    red = 'R' + str(i)
    for idx, row in frequent_itemsets.iterrows():  
        if red in row['itemsets']:  
            red_fre_stat.at[i-1, 'Count'] += 1

print("======红色球在频繁项集中出现的频次：======")
print(red_fre_stat.sort_values(by=['Count'], ascending=False) )




# 求出关联规则
# 默认用置信度来算，阈值是0.8，小于0.8的不要，此处修改为lift，小于lift为1的删除。
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("======关联规则======")
sorted_rules = rules.sort_values(by=['confidence', 'support', 'antecedents', 'consequents'], ascending=False)  
selected_columns = sorted_rules[['confidence', 'support', 'antecedents', 'consequents']]  
print(selected_columns)


red_list = []
blue_list = []
for items in rows_of_top_10_single_num['itemsets']: 
    for item in items: 
        # 将球自身加入到red_list或blue_list中。
        if item.startswith("R"):
            red_list.append(item)
        else:
            blue_list.append(item)
        # 从生成的关联规则中检索与该红球号码相关联的所有后驱号码，并将这些后驱号码按颜色分别放到red_list和blue_list中。
        consequents = rules.loc[rules['antecedents'].apply(lambda x: item in x), 'consequents']
        if not consequents.empty: 
            for ball in consequents: 
                ball_num = next(iter(ball))
                if ball_num.startswith("R"): # 放入红球list中
                    red_list.append(ball_num)
                else:   # 放入蓝球list中
                    blue_list.append(ball_num)

print("======候选红球：======")
print(red_list)
print("======候选蓝球：======")
print(blue_list)


red_df = pd.DataFrame(red_list, columns=['Red'])
blue_df = pd.DataFrame(blue_list, columns=['Blue'])

red_counts = red_df['Red'].value_counts()  
print("======候选红球出现频次：======")
print(red_counts)

blue_counts = blue_df['Blue'].value_counts()  
print("======候选蓝球出现频次：======")
print(blue_counts)

#查看一下支持度和置信度的关系,
print("======支持度和置信度的关系======\n")
plt.scatter(rules.support, rules.confidence)
plt.title("Association Rules")
plt.xlabel("support")
plt.ylabel("confidence")

# 自定义悬停时显示的信息  
cursor = mplcursors.cursor(hover=True)  
cursor.connect("add", lambda sel: sel.annotation.set_text(find_XY(rules, sel.target[0], sel.target[1])))

plt.show()