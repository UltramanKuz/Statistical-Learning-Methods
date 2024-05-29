import numpy as np
import time as time
import tqdm as tqdm


def load_data(file_path):
    """
    加载Mnist数据集
    """
    data = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            cur_line = line.strip().split(',')
            label.append(int(cur_line[0]))
            # 将[0, 255]转为[0, 1], 减少参数量
            data.append([1 if int(i) >= 128 else 0 for i in cur_line[1:]])
    return np.array(data), np.array(label)

def getProbability(train_data, train_label):
    """
    通过极大似然估计计算训练数据的先验概率和条件概率
    P(y=c_k)：类别c_k的先验概率
    P(x^(j)=a^(j)|y=c_k)：类别c_k的条件概率
    """
    # 每条数据共有28 * 28个特征
    feature_num = 28 * 28
    # 数据共分为10类
    class_num = 10

    # 初始化先验概率和条件概率
    Py = np.zeros((class_num, 1))
    Px_y = np.zeros((class_num, feature_num, 2))

    # 计算先验概率, 使用Laplace平滑，避免出现概率为0的情况
    for i in range(class_num):
        Py[i] = (np.sum(train_label == i) + 1) / (len(train_label) + class_num)
    
    # 计算条件概率，第一步先记录数据的类别和所有特征的出现次数
    for i in range(len(train_data)):
        label = train_label[i]
        x = train_data[i]
        for j in range(len(x)):
            # 第j个特征在类别label下取值为x[j]的次数
            Px_y[label][j][x[j]] += 1
    
    # 第二步计算条件概率，再次使用Laplace平滑, 避免出现概率为0的情况
    for i in range(class_num):
        for j in range(feature_num):
            Px_y0 = Px_y[i][j][0]
            Px_y1 = Px_y[i][j][1]
            Px_y[i][j][0] = (Px_y0 + 1) / (Px_y0 + Px_y1 + 2)
            Px_y[i][j][1] = (Px_y1 + 1) / (Px_y0 + Px_y1 + 2)

    # 最终返回对数概率，目的是将极大似然估计变成对数极大似然估计，避免概率连乘导致下溢出
    # 因此最终计算类别时，要求最大的对数概率和，而不是最大的概率积
    return np.log(Py), np.log(Px_y)



def NaiveBayes(Py, Px_y, x):
    """
    朴素贝叶斯分类器：根据极大似然估计得到的参数估计量来估计样本类别
    """
    P = np.zeros((10, 1))
    for i in range(10):
        P[i] = Py[i]
        for j in range(len(x)):
            # 计算对数概率和
            P[i] += Px_y[i][j][x[j]]
    # 返回最大概率的类别
    return np.argmax(P)

def test(test_data, test_label, Py, Px_y):
    """
    测试分类器的准确率
    """
    correct = 0
    for i in tqdm.tqdm(range(len(test_data))):
        label = NaiveBayes(Py, Px_y, test_data[i])
        correct += (label == test_label[i])
    return correct / len(test_data)

def main():
    # 加载数据
    start_time = time.time()
    print('Start loading training data...')
    train_data, train_label = load_data('../mnist_train.csv')
    print('Start loading testing data...')
    test_data, test_label = load_data('../mnist_test.csv')

    # 训练模型
    print('Start training...')
    Py, Px_y = getProbability(train_data, train_label)

    # 评估模型
    print('Start testing...')
    acc = test(test_data, test_label, Py, Px_y)
    end_time = time.time()
    print('Accuracy: {:.2f}'.format(acc))
    print('Running time:', end_time - start_time, 's')

if __name__ == '__main__':
    main()






