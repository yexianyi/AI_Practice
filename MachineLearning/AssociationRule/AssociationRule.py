import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings("ignore")
df = pd.DataFrame(
    [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
    ],
    columns=["News", "Finance", "Sports", "Arts", "Entertainment"],
    index=range(1, 7),
)
print(df)
# 求出频繁项集
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print(frequent_itemsets)
# 求出关联规则
# 默认用置信度来算，阈值是0.8，小于0.8的不要，此处修改为lift，小于lift为1的删除。
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)
# 查看一下支持度和置信度的关系,
plt.scatter(rules.support, rules.confidence)
plt.title("Association Rules")
plt.xlabel("support")
plt.ylabel("confidence")
