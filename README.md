# iris_flower
KNN解决鸢尾花分类问题
K近邻算法： 又称KNN
是指在输入的没有标签的新数据后，在训练集中找到与新数据最近邻的k个实例，如果这k个实例多数属于某个类别，那么新数据就属于这个类别，可以简单理解为：由离X最近的k个点来投票决定X归于哪一类

核心数学公式：距离公式  =  
也可以用三个或者四个坐标如 = 

数据归一化处理：
 =  
KNN步骤：
(1)样本的所有特征都要做可比较的量化
(2)样本特征要做归一化处理
(3)需要一个距离函数以计算两个样本之间的距离
#todo  KNN实现鸢尾花的分类

import pandas as pd  # 导入pandas库
data = pd.read_csv('F:\S\iris.txt',header=None) #读入txt数据文件
‘’里可以是路径也可以是名字
m = 0.7  #测试机和训练集的划分比例
n = int(data.shape[0]*m) #训练集的个数

data_1 = data.sample(frac=1).reset_index(drop=True) #随机打乱并且重新设置索引
随机打乱有三种
#切分数据集
train = data_1.iloc[:n,:]
Iloc是按数字划分，loc是按’名词’划分
test = data_1.iloc[n:,:].reset_index(drop=True)
reset_index 索引重新排列
print(test)
k = 5 # 设置分类
result = [] # 通过knn分类得到的标签
for i in range(test.shape[0]):
    # 算出来每个测试的元素距离所有训练集元素的距离
    dist = list(((train.iloc[:,0:4]-test.iloc[i,0:4])**2).sum(1)**0.5)
	Sum（1）是指将一行进行相加
    # 将距离和标签设置成为dataframe格式 方便下面处理
    dist_1 = pd.DataFrame({'dist':dist,'labels':(train.iloc[:,4])})
    # 找到k值内出现最多的标签即为该测试集的预测结果
    dr = dist_1.sort_values(by='dist')[:k]
	取距离最近的k个元素
    re = dr.loc[:,'labels'].value_counts()
	取距离最近的k个元素，计算哪个标签出现的最多
    result.append(re.index[0])
	出现最多的就是该花的预测种类
test['predict']=result # 将预测的标签放入原测试集中进行对比
print(test)

acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()
# 求出来预测的准确率
print(acc)



K-近邻：
功能：KNN多用于分类问题（核心）和回归问题
类型：有监督学习，惰性学习，距离类模型
数据输入：包含数据标签，至少包含k个数据训练样本，量纲统一，不统一进行归一化处理
模型输出：
分类中：输出的是某个类别
回归中：输出的是距离输入的数据最近的k个训练样本的平均值。
优点：
简单好用，容易理解，精度高，无需估计参数，无需训练。既可以做分类也可以做回归
可用于数值型数据和离散型数据
无数据输入假定
适合对稀有事件进行分类
缺点：
计算复杂性大，空间复杂度高
计算量大，且样本不能太少
样本不平衡问题
