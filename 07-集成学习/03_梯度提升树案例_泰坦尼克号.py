"""
案例:
    通过梯度提升树来演示 泰坦尼克号的数据集.

梯度提升树介绍:
    回顾: 提升树(BDT, Boosting Decision Tree), 采用 残差思路, 来不断地逼近 正确值.
        残差:
            真实值 - 预测值
        流程:
            真实值     预测值     残差
            100(岁)    80(岁)    20(岁)
            20(岁)     16(岁)    4(岁)
            4(岁)      3.2(岁)   0.8(岁)
            0.8(岁)    ......

    梯度提升树(GBDT, Gradient Boosting Decision Tree), 采用 负梯度 思路, 来不断地逼近 正确值.
        负梯度:
            经过公式推导, 负梯度 = 真实值 - 预测值, 所以可以把 负梯度当做 类似于 BDT树的残差来用.
        梯度提升树:
            采用了类似于 梯度下降算法, 不断地比较 正确值, 把多个弱学习器组合成1个强学习器, 属于 Boosting思想, 也是集成学习的一种.
"""

# 导包
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


# 1.读取数据.
data = pd.read_csv('./data/train.csv')

# 2. 数据的预处理.
# 2.1 提取特征 和 标签.
x = data[['Pclass', 'Age', 'Sex']].copy()
y = data['Survived'].copy()
# print(x.head())
# print(y.head())

# 2.2 处理缺失值.
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3 对数据sex(字符串形式)做 热编码处理.
x = pd.get_dummies(x)

# 2.4 查看数据集.
# print(len(x), len(y))
# print(x.head())
# print(y.head())

# 2.5 划分训练集和测试集.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 3. 构建模型.
# 3.1 构建 GBDT(梯度提升树, Gradient Boosting Decision Tree集成学习), 采用 模型默认值.
estimator = GradientBoostingClassifier()
# 模型训练
estimator.fit(x_train, y_train)
# 模型预测
y_predict = estimator.predict(x_test)
# 模型评分.
print(f'GBDT(梯度提升树) 正确率: {estimator.score(x_test, y_test)}')        # 0.75
print(f'GBDT(梯度提升树) 预测结果为: {accuracy_score(y_test, y_predict)}')
print('-' * 22)

# 4. 模型调优.
# 4.1 构建 GBDT(梯度提升树, Gradient Boosting Decision Tree集成学习), 采用 模型默认值.
gbc = GradientBoostingClassifier()
# 4.2 定义参数字典, 记录: 超参可选项.
param_dict = {'n_estimators': [70, 80, 90, 100, 130, 150, 200], 'max_depth':[2, 3, 4, 5, 6], 'random_state':[22]}
# 4.3 构建 GridSearchCV, 采用 网格搜索 + 交叉验证提升模型效果.
gbc_model = GridSearchCV(gbc, param_grid=param_dict, cv=3)
# 4.4 模型训练
gbc_model.fit(x_train, y_train)
# 4.5 模型预测.
y_predict = gbc_model.predict(x_test)
# 4.6 模型评分.
print(f'GBDT(梯度提升树) 正确率: {gbc_model.score(x_test, y_test)}')        #  0.76
print(f'GBDT(梯度提升树) 预测结果为: {accuracy_score(y_test, y_predict)}')
