import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_src = "train.csv"
df = pd.read_csv(data_src, header=0)

# check dataset basic info
'''
<bound method DataFrame.info of      PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0              1         0       3  ...   7.2500   NaN         S
1              2         1       1  ...  71.2833   C85         C
2              3         1       3  ...   7.9250   NaN         S
3              4         1       1  ...  53.1000  C123         S
4              5         0       3  ...   8.0500   NaN         S
..           ...       ...     ...  ...      ...   ...       ...
886          887         0       2  ...  13.0000   NaN         S
887          888         1       1  ...  30.0000   B42         S
888          889         0       3  ...  23.4500   NaN         S
889          890         1       1  ...  30.0000  C148         C
890          891         0       3  ...   7.7500   NaN         Q

[891 rows x 12 columns]>
'''
print(df.info)

# check data set abstract info
'''
       PassengerId    Survived      Pclass  ...       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  ...  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642  ...    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071  ...    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000  ...    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000  ...    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000  ...    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000  ...    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000  ...    8.000000    6.000000  512.329200
'''
print(df.describe())

# check first several rows
'''
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S

[5 rows x 12 columns]
'''
print(df.head())

# check total survived rate
survived_rate = df['Survived'].sum() / df['PassengerId'].count()
print(survived_rate)

print("PClass VS Survive rate", ":")
x = [df[(df.Pclass == 1)]['Pclass'].size,
     df[(df.Pclass == 2)]['Pclass'].size,
     df[(df.Pclass == 3)]['Pclass'].size
     ]
y = [df[(df.Pclass == 1) & (df.Survived == 1)]['Pclass'].size,
     df[(df.Pclass == 2) & (df.Survived == 1)]['Pclass'].size,
     df[(df.Pclass == 3) & (df.Survived == 1)]['Pclass'].size
     ]
print('1 Pclass number:'+str(x[0])+' '+' 1 Pclass survive:'+str(y[0])+' '+'1 Pclass survive rat:', float(y[0]) / x[0])
print('2 Pclass number:'+str(x[1])+' '+' 2 Pclass survive:'+str(y[1])+' '+'2 Pclass survive rat:', float(y[1]) / x[1])
print('3 Pclass number:'+str(x[2])+' '+' 3 Pclass survive:'+str(y[2])+' '+'3 Pclass survice rat:', float(y[2]) / x[2])


print("Gender VS Survive rate", ":")
male_survived = df[(df.Sex == 'male')]['Sex'].size
female_survived = df[(df.Sex == 'female')]['Sex'].size
print('male survive:', male_survived)
print('female survive:', female_survived)
Sex_survived_rate = (df.groupby(['Sex']).sum()/df.groupby(['Sex']).count())['Survived']
Sex_survived_rate.plot(kind='bar')
plt.title('Sex_survived_rate')
plt.show()


print("Age VS Survive rate", ":")
age_clean_date = df[~np.isnan(df['Age'])] # remove NaN

ages = np.arange(0, 81, 5)  # age scope from 0~80, each group contains 5 years
age_cut = pd.cut(age_clean_date.Age, ages)  # Quantilize data set
print("age_cut = ", age_cut)
'''
age_cut =  
0      (20, 25]
1      (35, 40]
2      (25, 30]
3      (30, 35]
4      (30, 35]
         ...   
885    (35, 40]
886    (25, 30]
887    (15, 20]
889    (25, 30]
890    (30, 35]
'''
age_cut_grouped = age_clean_date.groupby(age_cut)
age_Survival_Rate = (age_cut_grouped.sum() / age_cut_grouped.count())['Survived']  # 计算每年龄的幸存率

age_Survival_Rate.plot(kind='bar')
plt.title('Age_group_Survived_rate')
plt.show()


print("Survive Rate VS Composite Variable")
# Combine Pclass and Gender together to analyze
Pclass_Sex_Surivived_rate = (df.groupby(['Sex', 'Pclass']).sum() / df.groupby(['Sex', 'Pclass']).count())['Survived']
Pclass_Sex_Surivived_rate.plot(kind='bar')
plt.title('Pclass_Sex_Surivived_rate')
plt.show()
