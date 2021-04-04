import pandas as pd
from sklearn.preprocessing import StandardScaler

def stand_demo():
    data = pd.read_csv("dating.txt")
    print(data)

    transfer = StandardScaler()
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("Standardization result: \n", data)
    print("Mean of each figure: \n", transfer.mean_)
    print("Variance of each figure: \n", transfer.mean_)

    return None

stand_demo()
