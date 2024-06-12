import time
import numpy as np
import tqdm

def load_data(file_name):
    """
    加载Mnist数据集
    return: data: 图片数据, label: 标签
    """
    data = []
    label = []

    with open(file_name) as f:
        for line in f:
            line = line.strip().split(',')
            # 归一化数据集
            data.append([int(i) / 255 for i in line[1:]])
            # 二分类问题，只判断是否为0，若按照大于等于5判断，则精确率会降低(原博主调试结果)
            label.append(1 if int(line[0]) == 0 else 0)
    return data, label

def predict(w, x):
    """
    预测样本类别，根据逻辑回归的思想，sigmoid函数：p = 1 / (1 + e^(-wx))
    当p大于0.5时为正例，否则为负例
    
    """
    wx = np.dot(w, x)
    p = np.exp(wx) / (1 + np.exp(wx))
    return 1 if p >= 0.5 else 0

def logistic_regression(data, label, lr=0.001, iterations=100):
    """
    训练逻辑回归模型，使用随机梯度法, 此处的目标函数为极大似然函数，故使用梯度上升方法
    """
    for i in range(len(data)):
        data[i].append(1)
    data = np.array(data)
    w = np.zeros(len(data[0]))

    for i in tqdm.tqdm(range(iterations)):
        for x, y in zip(data, label):
            wx = np.dot(w, x)
            p = np.exp(wx) / (1 + np.exp(wx))
            # 计算关于每个算例的梯度，分别求导，然后累加
            w += lr * (y * x - (np.exp(wx) * x) / (1 + np.exp(wx)))
    return w

def test(data, label, w):
    """
    计算测试数据集的精确率
    """
    error_cnt = 0
    for i in range(len(data)):
        data[i].append(1)
    for i in range(len(data)):
        if predict(w, data[i]) != label[i]:
            error_cnt += 1
    return 1 - error_cnt / len(data)

def main():
    #开始时间
    start = time.time()

    # 获取训练集
    trainDataList, trainLabelList = load_data('../mnist_train.csv')
    # 获取测试集
    testDataList, testLabelList = load_data('../mnist_test.csv')

    #在训练集上学习模型参数w
    print('Start training')
    w = logistic_regression(trainDataList, trainLabelList, lr=0.001, iterations=100)

    #测试准确率
    print('Start testing')
    accur = test(testDataList, testLabelList, w)
    print('The accuracy is:', accur)

    #结束时间
    end = time.time()
    print('Time span:', end - start)

if __name__ == '__main__':
    main()

