import tqdm
import numpy as np
import time

def load_data(filename):
    """
    加载 Mnist 数据集, 输出数据集和标签集
    """
    data = []
    label = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            # 为了简化信息增益的计算，精确点说是条件熵的计算，
            # 故减少特征取值的数量，将特征值大于等于128的设置为1，小于128的设置为0，这样条件熵只包含两部分
            data.append([1 if int(i) >= 128 else 0 for i in line[1:]])
            label.append(int(line[0]))
    return data, label

def max_class(label):
    """
    根据标签中的最多数类别作为结果返回 
    """
    class_dict = {}
    for i in range(len(label)):
        class_dict[label[i]] = class_dict.get(label[i], 0) + 1
    class_sort = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
    return class_sort[0][0]

def calculate_H_D(label):
    """
    计算标签集的信息熵，即将当前样本的每一类的经验概率乘以负对数概率并进行求和
    """
    H_D = 0
    # 计算类别，这样当特定类别数量为0时就不需要特殊处理了，否则还需要指定0 * log2(0) ：= 0
    class_set = set([l for l in label])
    # 对每一个类别计算经验概率乘以负对数概率并求和，实际使用减法即可
    for i in class_set:
        # 此处的写法为
        p = label[label == i].size / label.size
        H_D -= p * np.log2(p)
    return H_D

def calculate_H_D_A(data, label):
    """
    计算标签集的条件熵
    data: 某一特征的数据，即原数据中的一列，共有784列
    """

    H_D_A = 0
    # 根据选定特征，计算该特征不同取值时的经验熵
    # 本实验的样本为了简便，只有两类，即0和1，所以只需要计算两类的经验熵即可
    class_set = set([d for d in data])
    for i in class_set:
        H_D_A += data[data == i].size / data.size  * calculate_H_D(label[data == i])
    return H_D_A    

def choose_best_feature(data, label):
    """ 
    选择最佳特征，即信息增益最大的特征
    data: 全部样本数据
    """
    max_gda = 0
    # 转为np数组，方便调用size属性
    data = np.array(data)
    label = np.array(label)
    # 可供选择的特征为当前样本的长度，一开始为784，后面递减到1
    feature_num = data.shape[1]
    H_D = calculate_H_D(label)
    for feature in range(feature_num):
        gda = H_D - calculate_H_D_A(data[:, feature], label)
        if gda > max_gda:
            max_gda = gda
            best_feature = feature
    return best_feature, max_gda

def get_sub_data(data, label, feature, value):
    """
    确定选择的特征后，需要切割数据集，删除选择特征的这一列数据，并根据特征取值分为两个子数据集
    data:当前样本数据集
    feature: 当前选择的特征
    value: 当前选择特征的取值
    """
    ret_data = []
    ret_label = []
    for i in range(len(label)):
        # 只要特征为特定取值的数据
        if data[i][feature] == value:
            ret_data.append(data[i][:feature] + data[i][feature+1:])
            ret_label.append(label[i])
    return ret_data , ret_label


def create_tree(*dataset, epsilon=0.1):
    """
    递归创建决策树
    epsilon: 信息增益的阈值
    dataset: 元组形式的数据集，包含data和label，这样写的目的是直接将函数返回值递归调用，而函数返回值是一个元组
    return: 未剪枝的决策树
    """
    data = dataset[0][0]
    label = dataset[0][1]

    # 递归第一步：设置递归终止条件
    # 获得当前标签集中的类别, 得到类别set
    class_dic = {i for i in label}

    # 若当前只有一个类别，无需再分类，返回该类别即可
    if len(class_dic) == 1:
        return label[0]

    # 若当前特征集为空，代表所有特征都已用完，已分到最后的叶子节点，无法再划分，故返回当前标签集中最多的类别作为结果
    if len(data[0]) == 0:
        return max_class(label)

    # 若尚未终止，代表当前样本还需要继续划分，选择最佳特征
    best_feature, max_gda = choose_best_feature(data, label)
    # 若最大信息增益小于阈值便不再细分，认为剩余所有特征都没有分辨能力，故返回当前标签集中最多的类别作为结果
    if max_gda < epsilon:
        return max_class(label)
    
    # 否则继续递归，创建子树
    tree = {best_feature:{}}
    # 划分当前数据集，递归调用并创建左右子树，
    tree[best_feature][0] = create_tree(get_sub_data(data, label, best_feature, 0))
    tree[best_feature][1] = create_tree(get_sub_data(data, label, best_feature, 1))
    # 返回值有两种可能，一是子树，二是类别
    return tree

def predict(testDataList, tree):
    '''
    预测标签
    :param testDataList:样本
    :param tree: 决策树
    :return: 预测结果
    '''

    #死循环，直到找到一个有效的分类方法
    while True:
        # 这行代码表示的是字典的unpacking操作，因为只有一个键值对，所以只会返回一个元素
        # 因此(key, value)后要＋,表示(key, value)为第一个键值对
        # 若不加, 那么key就是第一个元素（键值对），value没有值分配，会报错
        # 当然，也可以使用tree.keys()和tree.values()来获取键和值, 但是这样写更简洁，不然还需要对keys和values进行list化
        (key, value), = tree.items()
        #如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            # 因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
            # 所以在获取完目前所在节点的feature值后，需要在样本中删除该feature, 保证训练集和测试集索引位置是一致的
            # 此处的key是feature值，即当前的索引值，dataVal是该索引值对应的特征取值（本实验下为0 或者 1）
            dataVal = testDataList[key]
            # 删除样本的第key个特征，保证接下来选择的特征索引是正确的
            del testDataList[key]
            # 根据dataVal将tree更新为其子节点的字典
            tree = value[dataVal]
            # 如果当前节点的子节点的值是int，就直接返回该int值，不需要再遍历下去了，即已经确定了样本的预测类别了
            # 否则继续按照特征取值分下去，直到找到一个int值作为结果
            if type(tree).__name__ == 'int':
                #返回该节点值，也就是分类值
                return tree
        else:
            #如果当前value不是字典，那就返回分类值
            return value

def test(data, label, tree):
    '''
    测试准确率
    data: 测试数据集
    tree: 训练集生成的树
    return: 准确率
    '''
    #错误次数计数
    error_cnt = 0
    #遍历测试集中每一个测试样本
    for i in range(len(data)):
        #判断预测与标签中结果是否一致
        if label[i] != predict(data[i], tree):
            error_cnt += 1
    #返回准确率
    return 1 - error_cnt / len(data)

def main():
        #开始时间
    start = time.time()

    # 获取训练集
    trainDataList, trainLabelList = load_data('../mnist_train.csv')
    # 获取测试集
    testDataList, testLabelList = load_data('../mnist_test.csv')

    #创建决策树
    print('Start creating a tree')
    tree = create_tree((trainDataList, trainLabelList))
    print('The tree is:', tree)

    #测试准确率
    print('Start testing')
    accur = test(testDataList, testLabelList, tree)
    print('The accuracy is:', accur)

    #结束时间
    end = time.time()
    print('Time span:', end - start)

if __name__ == '__main__':
    main()
        





    
    
    
    
    