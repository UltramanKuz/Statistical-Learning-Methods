import time
import numpy as np
import math
import random
import tqdm

# 使用AdaBoost算法处理二分类问题，基函数选择二叉决策树桩（单层决策树), 返回实现二分类问题的提升树.
# 关于二叉决策树桩： 
# 总共有n个特征，n=784
# 特征取值为2，每个特征有三种划分方式，划分点为[-0.5, 0.5, 1.5]
# 划分规则有两种，划分点左边为正例还是右边为正例
# 故基函数一共有n * 3 * 2种
# 最终选择误分率最低的决策树桩

def load_data(file_name):
    """
    加载Mnist数据集
    return: data: 图片数据, label: 标签
    """
    data = []
    label = []

    with open(file_name) as f:
        flag = 0
        for line in f:
            line = line.strip().split(',')
            # 减少特征数量，否则运行时间太长
            data.append([int(int(i) >= 128) for i in line[1:]])
            # 二分类问题，只判断是否为0类
            label.append(1 if int(line[0]) == 0 else -1)
    return data, label

def classify_method(rule):
    """
    根据分类规则返回L和R的值
    """
    if rule == 'L1':
        # 左节点为正例
        L = 1
        R = -1
    else: 
        # 右节点为正例
        L = -1
        R = 1
    return L, R

def calculate_e(train_data, train_label, feature, division_point, rule, weight):
    """
    计算单个决策树桩的误分率
    """
    e = 0
    # 只需要当前特征的数据
    feature_col = train_data[:, feature]
    # 确定分类规则
    L, R = classify_method(rule)
    for i in range(len(train_data)):
        if feature_col[i] < division_point:
            if train_label[i] != L:
                # 误分类时，误分类率要加上该样本的权重
                e += weight[i]
        else:
            if train_label[i] != R:
                e += weight[i]
    return e

def get_error_index(train_data, train_label, tree):
    """
    计算误分类的样本索引，用于更新权重
    """
    feature = tree['feature']
    division_point = tree['division_point']
    rule = tree['rule']
    error_index = []
    feature_col = train_data[:, feature]
    L, R = classify_method(rule)
    for i in range(len(train_data)):
        if feature_col[i] < division_point:
            if train_label[i] != L:
                error_index.append(i)
        else:
            if train_label[i] != R:
                error_index.append(i)
    return error_index


def update_weight(weight, alpha, error_index):
    """
    更新样本权重
    """
    for i in range(len(weight)):
        if i in error_index:
            weight[i] *= math.exp(alpha)
        else:
            weight[i] *= math.exp(-alpha)
    # 归一化，使其表示权重
    return np.array(weight) / sum(weight)

def create_single_boosting_tree(train_data, train_label, weight):
    """
    创建单棵提升树
    """
    tree = {}
    feature_num = train_data.shape[1]
    e = 1
    for feature in range(feature_num):
        for division_point in [-0.5, 0.5, 1.5]:
            for rule in ['L1', 'R1']:
                tmp_e = calculate_e(train_data, train_label, feature, division_point, rule, weight)
                # 选择误分率最低的决策树桩，保存其信息
                if tmp_e < e:
                    e = tmp_e
                    tree['e'] = e
                    tree['feature'] = feature
                    tree['division_point'] = division_point
                    tree['rule'] = rule
                    tree['alpha'] = 0.5 * math.log((1 - e) / e)
    return tree

def create_boosting_tree(train_data, train_label, tree_num=50):
    """
    创建提升树
    """
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    m = train_data.shape[0]
    # 初始化的权重为相等权重
    weight = [1 / m] * m
    boosting_tree = []

    for i in tqdm.tqdm(range(tree_num)):
        # 创建单棵提升树
        tree = create_single_boosting_tree(train_data, train_label, weight)
        boosting_tree.append(tree)
        # 更新权重
        error_index = get_error_index(train_data, train_label, tree)
        alpha = tree['alpha']
        weight = update_weight(weight, alpha, error_index)
        boosting_tree.append(tree)
    return boosting_tree

def predict(tree, x):
    """
    预测单个数据的类别
    """
    L, R = classify_method(tree['rule'])
    if x[tree['feature']] < tree['division_point']:
        return L
    else:
        return R
    

def test(test_data, test_label, boosting_tree):
    """
    测试准确率
    """
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    m = test_data.shape[0]
    error_cnt = 0
    for i in range(m):
        value = 0
        for tree in boosting_tree:
            value += tree['alpha'] * predict(tree, test_data[i])
        # value 为正则为正例，反之为负例
        if np.sign(value) != test_label[i]:
            error_cnt += 1
    return 1 - error_cnt / m

def main():
    # 开始时间
    start = time.time()

    # 获取训练集
    train_data, train_label = load_data('../mnist_train.csv')
    # 获取测试集
    test_data, test_label = load_data('../mnist_test.csv')

    # 创建提升树
    print('Start creating a boosting tree')
    tree = create_boosting_tree(train_data[:1000], train_label[:1000], 10)

    #测试准确率
    print('Start testing')
    accuracy = test(test_data[:100], test_label[:100], tree)
    print('The accuracy is:', accuracy)

    #结束时间
    end = time.time()
    print('Time span:', end - start)

if __name__ == '__main__':
    main()

        







