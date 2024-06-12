import numpy as np
import time
import tqdm



def loadData(file_path):
    """
    加载Mnist数据，返回数据集和标签
    """
    data = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            cur_line = line.strip().split(',')
            # 因为感知机是二分类，故要将数据二分，策略是将大于等于5的数据标为1，小于5的标为-1
            if int(cur_line[0]) >= 5:  
                label.append(-1)
            else:
                label.append(1)
            # 数据归一化, 0-255 -> 0-1
            data.append([int(num) / 255 for num in cur_line[1:]]) 
    return data, label

def train(data, label, epoch=50, lr = 0.0001):
    """
    感知器训练过程
    data: 训练集的数据
    label: 训练集的标签
    epoch: 迭代次数，默认50
    lr: 学习率，默认0.0001 
    """
    print('Start to train:')
    # 用标准正态分布初始化权重
    w = np.random.randn(dimension) 
    # 偏置初始化为0
    b = 0  
    
    for _ in tqdm.tqdm(range(epoch)):
        for i in range(len(data)):
            x = np.array(data[i])
            y = label[i]
            if y * (np.dot(w, x) + b) <= 0:
                # list 和 turple 不能乘以浮点数，所以要转换成np.array（x）
                w += lr * y * x
                b += lr * y
    end_time = time.time()
    print('Training cost %.2f seconds' % (end_time - start_time))
    return w, b

def test(data, label, w, b):
    print('Start to test:')
    start_time = time.time()
    error_count = 0
    for i in range(len(data)):
        x = data[i]
        y = label[i]
        if y * (np.dot(w, x) + b) <= 0:
            error_count += 1
    error_rate = error_count / len(data)
    end_time = time.time()
    print('Testing cost %.2f seconds' % (end_time - start_time))
    return 1 - error_rate

if __name__ == '__main__':
    train_data, train_label = loadData('../mnist_train.csv')
    test_data, test_label = loadData('../mnist_test.csv')
    w, b = train(train_data, train_label)
    acc = test(test_data, test_label, w, b)
    print('The accuracy is %.2f' % acc)

