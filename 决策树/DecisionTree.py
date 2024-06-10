import tqdm
import numpy as np
import time

def load_data(filename):
    data = []
    label = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            data.append([1 if int(i) >= 128 else 0 for i in line[1:]])
            label.append(int(line[0]))
    return data, label

def max_class(label):
    class_dict = {}
    for i in range(len(label)):
        class_dict[label[i]] = class_dict.get(label[i], 0) + 1
    class_sort = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
    return class_sort[0][0]

def calculate_H_D(label):
    H_D = 0

    class_set = set([l for l in label])
    for i in class_set:
        p = label[label == i].size / label.size
        H_D -= p * np.log2(p)
    return H_D

def calculate_H_D_A(data, label):
    H_D_A = 0

    class_set = set([d for d in data])
    for i in class_set:
        H_D_A += data[data == i].size / data.size  * calculate_H_D(label[data == i])
    return H_D_A    

def choose_best_feature(data, label):
    max_gda = 0
    data = np.array(data)
    label = np.array(label)
    feature_num = data.shape[1]
    H_D = calculate_H_D(label)
    for feature in range(feature_num):
        gda = H_D - calculate_H_D_A(np.array(data[:, feature].flat), label)
        if gda > max_gda:
            max_gda = gda
            best_feature = feature
    return best_feature, max_gda

def get_sub_data(data, label, feature, value):
    ret_data = []
    ret_label = []
    for i in range(len(label)):
        if data[i][feature] == value:
            ret_data.append(data[i][:feature] + data[i][feature+1:])
            ret_label.append(label[i])
    return ret_data , ret_label


def create_tree(*dataset, epsilon=0.1):
    data = dataset[0][0]
    label = dataset[0][1]

    class_dic = {i for i in label}

    if len(class_dic) == 1:
        return label[0]

    if len(data[0]) == 0:
        return max_class(label)

    best_feature, max_gda = choose_best_feature(data, label)
    if max_gda < epsilon:
        return max_class(label)
    
    tree = {best_feature:{}}
    tree[best_feature][0] = create_tree(get_sub_data(data, label, best_feature, 0))
    tree[best_feature][1] = create_tree(get_sub_data(data, label, best_feature, 1))

    return tree

def predict(testDataList, tree):
    '''
    预测标签
    :param testDataList:样本
    :param tree: 决策树
    :return: 预测结果
    '''
    # treeDict = copy.deepcopy(tree)

    #死循环，直到找到一个有效地分类
    while True:
        #因为有时候当前字典只有一个节点
        #例如{73: {0: {74:6}}}看起来节点很多，但是对于字典的最顶层来说，只有73一个key，其余都是value
        #若还是采用for来读取的话不太合适，所以使用下行这种方式读取key和value
        (key, value), = tree.items()
        #如果当前的value是字典，说明还需要遍历下去
        if type(tree[key]).__name__ == 'dict':
            #获取目前所在节点的feature值，需要在样本中删除该feature
            #因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
            #所以需要不断地删除已经用掉的特征，保证索引相对位置的一致性
            dataVal = testDataList[key]
            del testDataList[key]
            #将tree更新为其子节点的字典
            tree = value[dataVal]
            #如果当前节点的子节点的值是int，就直接返回该int值
            #例如{403: {0: 7, 1: {297:7}}，dataVal=0
            #此时上一行tree = value[dataVal]，将tree定位到了7，而7不再是一个字典了，
            #这里就可以直接返回7了，如果tree = value[1]，那就是一个新的子节点，需要继续遍历下去
            if type(tree).__name__ == 'int':
                #返回该节点值，也就是分类值
                return tree
        else:
            #如果当前value不是字典，那就返回分类值
            return value
        
def model_test(testDataList, testLabelList, tree):
    '''
    测试准确率
    :param testDataList:待测试数据集
    :param testLabelList: 待测试标签集
    :param tree: 训练集生成的树
    :return: 准确率
    '''
    #错误次数计数
    errorCnt = 0
    #遍历测试集中每一个测试样本
    for i in range(len(testDataList)):
        #判断预测与标签中结果是否一致
        if testLabelList[i] != predict(testDataList[i], tree):
            errorCnt += 1
    #返回准确率
    return 1 - errorCnt / len(testDataList)


def test(data, label, tree):
    '''
    测试准确率
    :param testDataList:待测试数据集
    :param testLabelList: 待测试标签集
    :param tree: 训练集生成的树
    :return: 准确率
    '''
    #错误次数计数
    errorCnt = 0
    #遍历测试集中每一个测试样本
    for i in range(len(data)):
        #判断预测与标签中结果是否一致
        if label[i] != predict(label[i], tree):
            errorCnt += 1
    #返回准确率
    return 1 - errorCnt / len(data)

def main():
        #开始时间
    start = time.time()

    # 获取训练集
    trainDataList, trainLabelList = load_data('../mnist_train.csv')
    # 获取测试集
    testDataList, testLabelList = load_data('../mnist_test.csv')

    #创建决策树
    print('start create tree')
    tree = create_tree((trainDataList, trainLabelList))
    print('tree is:', tree)

    #测试准确率
    print('start test')
    accur = model_test(testDataList, testLabelList, tree)
    print('the accur is:', accur)

    #结束时间
    end = time.time()
    print('time span:', end - start)

if __name__ == '__main__':
    main()
        





    
    
    
    
    