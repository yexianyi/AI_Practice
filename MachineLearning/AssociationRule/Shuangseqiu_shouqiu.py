import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import mplcursors  

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

max_combination = None  
max_sum = -1

def find_XY(df, x, y):
    filtered_df = df[(df['support'] == x) & (df['confidence'] == y)]
    rows_as_str = filtered_df[['support', 'confidence', 'antecedents', 'consequents']].to_string(index = False)
    return rows_as_str

def find_max_value_combination(df, start, k=6, current_combination=[], current_sum=0):  
    global max_combination  
    global max_sum  

    # 基本情况：如果已找到k个元素，返回当前组合和总和  
    if len(current_combination) == k:  
        # print("return:" + " ,".join(current_combination))
        return current_combination.copy(), current_sum  
      
    # 遍历DataFrame的剩余行  
    filtered_df = df[df['antecedents'].apply(lambda x: start in x)] 
    for index, row in filtered_df.iterrows():  
            # 尝试添加当前行到组合中  
            consequent = next(iter(row['consequents']))
            if consequent.startswith('B'): # 暂时忽略掉后驱球是蓝球的情况，以减轻计算量
                 continue
            current_combination.append(consequent)  
            new_sum = current_sum + row['support']
    
            next_combination, next_sum = find_max_value_combination(  
                df, consequent, k, current_combination, new_sum  
            )  

            current_combination.pop()
            new_sum = current_sum - row['support']
              
            # 更新最大组合和总和  
            if next_sum > max_sum:  
                max_combination = next_combination  
                max_sum = next_sum  
                # print("max_combination:" + " , ".join(max_combination))
      
    return max_combination, max_sum  



df = pd.read_csv("AssociationRule\shuangseqiu.csv")


rows_count = df.shape[0]
# 1. 初始化用于求频繁项集的数据表
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

df2 = pd.DataFrame(data, columns=['0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 
                                 'R10', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19',
                                 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29',
                                 'R30', 'R31', 'R32', 'R33', 'B1',  'B2',  'B3', 'B4', 'B5', 'B6', 'B7',
                                 'B8',  'B9',  'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16']) 

# 2.求出频繁项集
frequent_itemsets = apriori(df2, min_support=0.01, use_colnames=True)
print("======频繁项集======")
frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)  
frequent_itemsets['Rank'] = frequent_itemsets['support'].rank(method='min', ascending=False)
print(frequent_itemsets)


# 求红色球在频繁项集中出现的频次：
# red_values = [f'R{i+1}' for i in range(33)]  
# count_values = [0] * 33  
# data = {'Red': red_values, 'Count': count_values} 
# red_fre_stat = pd.DataFrame(data) 

# for i in range(1, 34):
#     red = 'R' + str(i)
#     for idx, row in frequent_itemsets.iterrows():  
#         if red in row['itemsets']:  
#             red_fre_stat.at[i-1, 'Count'] += 1

# print("======红色球在频繁项集中出现的频次：======")
# print(red_fre_stat.sort_values(by=['Count'], ascending=False) )



# rows_of_top_10_single_num = frequent_itemsets.nlargest(10, 'support')  
# print("======频繁出现的前10个数字, 作为初始的红球预选号码======")
# print(rows_of_top_10_single_num)

# 3. 求出关联规则
# 默认用置信度来算，阈值是0.8，小于0.8的不要，此处修改为lift，小于lift为1的删除。
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print("======关联规则======")
sorted_rules = rules.sort_values(by=['confidence', 'support', 'antecedents', 'consequents'], ascending=False)  
selected_columns = sorted_rules[['confidence', 'support', 'antecedents', 'consequents']]  
print(selected_columns)



# 4. 求R1~R33作为第一个被抓取到红球的频率
red_values = [f'R{i+1}' for i in range(33)]  
count_values = [0] * 33  
data = {'Red': red_values, 'Count': count_values} 
red_fre_stat = pd.DataFrame(data) 
red_counts = df['O1'].value_counts(ascending=True)  
print("======第一个被抓取到红球的频率(从小到大):======")
print(red_counts)

# 5. 以出现频率最低的红球作为初始备选红球

# 6. 以初始备选红球作为首个被抽取的球，依次从关联规则中寻找该球支持度和置信度最高的后驱球，作为下个可能被抽取的球
# for num, count in red_counts.items(): 
#     result = df.loc[df['antecedents'] == 'R' + str(num), 'consequents'].iloc[0]



  
# 调用函数  
for i in range(1, 34):
    red = 'R' + str(i)
    max_combination, max_sum = find_max_value_combination(rules, start=red, current_combination=[red])   
    print("Max Combination for " + red + ": ", max_combination)  
    print("Max Sum:", max_sum) 
    max_combination = None  
    max_sum = -1