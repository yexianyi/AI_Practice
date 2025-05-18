import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


class DoubleColorBallPredictor:
    def __init__(self):
        # 初始化模型
        self.red_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
        self.blue_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        self.feature_columns = [
            '和值', '平均值', '尾数和值', '奇号个数', '偶号个数', '奇偶偏差',
            '奇号连续', '偶号连续', '大号个数', '小号个数', '大小偏差', '尾号组数',
            'AC值', '连号个数', '连号组数', '首尾差', '最大间距', '同位相同', '重号个数', '斜号个数'
        ]
        self.red_columns = [f'红色球{i}' for i in range(1, 7)]
        self.blue_column = '蓝色球'

    def load_data(self, file_path):
        """加载并预处理数据"""
        data = pd.read_csv(file_path)

        # 确保列名正确
        data.columns = [
            '期号', '开奖日期',
            *self.red_columns,
            *[f'摇出红色球{i}' for i in range(1, 7)],
            self.blue_column,
            *self.feature_columns
        ]

        # 添加一些衍生特征
        data['红球最大值'] = data[self.red_columns].max(axis=1)
        data['红球最小值'] = data[self.red_columns].min(axis=1)
        data['红球范围'] = data['红球最大值'] - data['红球最小值']

        return data

    def prepare_features_labels(self, data):
        """准备特征和标签"""
        # 特征矩阵
        X = data[self.feature_columns + ['红球最大值', '红球最小值', '红球范围']]

        # 标签 - 预测下一个红球和蓝球
        # 对于红球，我们预测每个位置的可能号码
        y_red = data[self.red_columns].values
        # 对于蓝球，直接预测号码
        y_blue = data[self.blue_column].values

        return X, y_red, y_blue

    def train(self, X, y_red, y_blue):
        """训练模型"""
        # 训练红球模型 - 为每个位置训练一个模型
        self.red_models = []
        for i in range(6):
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ])
            model.fit(X, y_red[:, i])
            self.red_models.append(model)

        # 训练蓝球模型
        self.blue_model.fit(X, y_blue)

    def predict(self, X, top_n=5):
        """进行预测"""
        # 预测红球 - 每个位置预测top_n个最可能的号码
        red_preds = []
        for model in self.red_models:
            probas = model.predict_proba(X)[:, 1]
            top_indices = np.argsort(probas)[-top_n:]
            red_preds.append(top_indices + 1)  # 号码从1开始

        # 预测蓝球
        blue_probas = self.blue_model.predict_proba(X)[:, 1]
        blue_top_indices = np.argsort(blue_probas)[-top_n:]
        blue_preds = blue_top_indices + 1

        return red_preds, blue_preds

    def generate_recommendation(self, red_preds, blue_preds):
        """生成推荐号码组合"""
        # 获取所有可能的红球组合
        from itertools import product
        red_combinations = list(product(*red_preds))

        # 简单的组合选择策略 - 选择频率最高的组合
        from collections import Counter
        counter = Counter(red_combinations)
        most_common = counter.most_common(1)[0][0]

        # 蓝球选择最可能的
        most_common_blue = blue_preds[0] if len(set(blue_preds)) == 1 else np.random.choice(blue_preds)

        # 确保红球不重复且排序
        final_red = sorted(list(set(most_common)))[:6]

        return final_red, most_common_blue


def main():
    # 初始化预测器
    predictor = DoubleColorBallPredictor()

    # 加载数据 (替换为您的实际文件路径)
    data = predictor.load_data('shuangseqiu.csv')

    # 准备特征和标签
    X, y_red, y_blue = predictor.prepare_features_labels(data)

    # 划分训练测试集
    X_train, X_test, y_red_train, y_red_test, y_blue_train, y_blue_test = train_test_split(
        X, y_red, y_blue, test_size=0.2, random_state=42
    )

    # 训练模型
    predictor.train(X_train, y_red_train, y_blue_train)

    # 预测下一期
    latest_features = X.iloc[-1:].values
    red_preds, blue_preds = predictor.predict(latest_features)

    # 生成推荐号码
    final_red, final_blue = predictor.generate_recommendation(red_preds, blue_preds)

    # 输出结果
    print("\n=== 双色球预测推荐 ===")
    print("红球推荐号码:", sorted(final_red))
    print("蓝球推荐号码:", final_blue)
    print("\n提示: 彩票本质是随机游戏，此预测仅供娱乐参考")


if __name__ == '__main__':
    main()