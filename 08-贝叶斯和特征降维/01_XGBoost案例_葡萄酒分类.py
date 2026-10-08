"""
案例:
    演示葡萄酒分类, 最后1列是 标签, 其它是 特征.

目的:
    给大家演示 XGBoost算法(极限梯度提升树, Extreme Gradient Boosting), 底层通过 打分函数 决定是否划分分支.
    分数 = 分裂前 - 分裂后的(左子树 + 右子树)
    如果 分数 > 0, 可以考虑分裂.
    如果 分数 < 0, 不分裂.
"""

# 导包
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb                   # 引入 xgboost包, 必须通过 xgb. 的方式调用.
from collections import Counter         # 引入 统计类
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold # 引入 训练集和测试集的拆分, 网格搜索+交叉验证, 分层采样
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold # 引入 分层K折
from sklearn.utils import class_weight  # 引入 类权重

# from xgboost import XGBClassifier     # 这种方式, 可以直接用: XGBClassifier


# 1. 定义函数, 实现数据的加载和预处理.
def dm01_数据预处理():
    # 1. 加载数据.
    data = pd.read_csv('./data/红酒品质分类.csv')
    # print(data.head(10))

    # 2. 抽取特征和标签.
    x = data.iloc[:, :-1]       # 除了最后1行, 都是: 特征.
    y = data.iloc[:, -1] - 3    # 最后1行: 标签, 原始数据范围[3, 8] -> [0, 5], 刚好6个分类, 多分类法.

    # 3. 打印结果.
    print(x.head(10))
    print(y.head(10))

    # 4. 查看数据分布情况, 确实: 标签不均衡.
    print(Counter(y))   # Counter({2: 681, 3: 638, 4: 199, 1: 53, 5: 18, 0: 10})

    # 5. 拆分训练集和测试集.
    # 参数: 要拆分的特征, 标签, 测试集比例, 随机种子, stratify(拆分数据时, 参考原始数据的样本比例)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22, stratify=y)
    # 打印下结果.
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # 打印拆分后, 训练集 和 测试集 标签分布情况.
    print(Counter(y_train)) # Counter({2: 545, 3: 510, 4: 159, 1: 42, 5: 15, 0: 8})
    print(Counter(y_test))  # Counter({2: 136, 3: 128, 4: 40, 1: 11, 5: 3, 0: 2})

    # 6. 为了避免以后重复的做上述的操作, 我们可以把结果保存到文件中.
    # 按照列拼接(axis=1), 分别把 训练集, 测试集的  特征和标签 各自拼接到一起, 保存到本地 csv文件中.
    # print(pd.concat([x_train, y_train], axis=1))
    pd.concat([x_train, y_train], axis=1).to_csv('./data/红酒品质分类_train.csv', index=False)
    pd.concat([x_test, y_test], axis=1).to_csv('./data/红酒品质分类_test.csv', index=False)


# 2. 定义函数, 实现: XGBoost模型训练.
def dm02_模型训练():
    # 1. 加载数据.
    train_data = pd.read_csv('./data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./data/红酒品质分类_test.csv')

    # 2. 数据预处理, 抽取特征和标签.
    # print(train_data.head())
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # 扩展: 查看数据分布情况, 确实: 标签不均衡.
    print(Counter(y_train))
    # 平衡样本的权重问题, balanced: 默认是: 平衡.
    # clazz_weight= class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    class_weight.compute_sample_weight('balanced', y_train)

    # 3.创建XGBoost模型对象: 极限梯度提升树.
    # 参数:
    #   objective: 损失函数, 默认是: 二分类, 我们的标签是: 6个分类[0, 5], 所以是: multi:softmax -> 多分类.
    #   learning_rate: 学习率(理解为权重), 默认是: 0.3, 防止过拟合.
    #   max_depth: 树的深度, 默认是: 6.
    #   n_estimators: 树的数量, 默认是: 100.
    estimator = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, objective='multi:softmax')
    # 4. 模型训练.
    estimator.fit(x_train, y_train)

    # 5. 模型预测.
    y_predict = estimator.predict(x_test)
    print(f'预测结果为: {y_predict}')

    # 6. 模型评估
    print(f'正确率: {estimator.score(x_test, y_test)}')    # 正确率: 0.65

    # 7. 模型保存.
    joblib.dump(estimator, './model/红酒品质分类.pkl')    # .pkl, .pth 都可以, 都是 pickle的扩展名.


# 3. 定义函数, 实现: XGBoost模型预测.
def dm03_模型预测():
    # 1. 加载数据.
    train_data = pd.read_csv('./data/红酒品质分类_train.csv')
    test_data = pd.read_csv('./data/红酒品质分类_test.csv')

    # 2. 数据预处理, 抽取特征和标签.
    # print(train_data.head())
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # 扩展: 查看数据分布情况, 确实: 标签不均衡.
    # print(Counter(y_train))

    # 3.分层采样(一般结合 网格搜索+交叉验证一起使用), 目的: 避免过拟合, 采样时 会让采样数据的各分类比例 保持 和原始数据的 各分类比例 保持近似一致.
    # 参1: 采样的次数(类似于折数), shuffle: 是否打乱, random_state: 随机种子.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)

    # 4.读取文件,获取模型对象 -> XGBoost模型对象: 极限梯度提升树.
    estimator = joblib.load('./model/红酒品质分类.pkl')

    # 5. 定义超参字典, 记录超参可取值.
    param_dict = {'n_estimators': [60, 90, 130, 150, 200], 'max_depth':[2, 3, 5, 7, 9], 'learning_rate':[0.1, 0.2, 0.3, 1, 1.3]}

    # 6. 通过网格搜索 + 交叉验证, 寻找最优的参数组合.
    grid_search = GridSearchCV(estimator, param_grid=param_dict, cv=skf)
    # 7. 模型训练.
    grid_search.fit(x_train, y_train)
    # 8. 模型预测.
    y_predict = grid_search.predict(x_test)
    print(f'预测值: {y_predict}')

    # 9. 评估, 打印最优组合.
    print(f'最优参数组合: {grid_search.best_estimator_}')
    print(f'最优准确率: {grid_search.best_score_}')
    print(f'模型准确率: {grid_search.score(x_test, y_test)}')


# 4. 在main函数中测试.
if __name__ == '__main__':
    # dm01_数据预处理()
    # dm02_模型训练()
    dm03_模型预测()