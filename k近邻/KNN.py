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
            data.append([int(i) for i in cur_line[1:]])
    return np.array(data), np.array(label)

def EuclideanDistance(x, y):
    """
    计算欧几里得距离
    """
    return np.sqrt(np.sum(np.square(x - y)))

def ManhattanDistance(x, y):
    """
    计算曼哈顿距离
    """
    return np.sqrt(np.sum(np.abs(x - y)))

def getClosest(train_data, train_label, x, topk, method):
    """
    根据训练数据得到最近的k个样本，根据样本类别数设定x的类别
    """
    distance = []
    for sample in train_data:
        # 两种计算方法，一般选择欧几里得距离
        if method == 'Euclidean':
            distance.append(EuclideanDistance(sample, x))
        else:
            distance.append(ManhattanDistance(sample, x))
    index = np.argsort(np.array(distance))[:topk]
    label = np.array(train_label)[index]
    return np.argmax(np.bincount(label))

def test(train_data, train_label, test_data, test_label, topk, method):
    """
    根据训练数据对测试数据进行测试，得到准确率
    """
    print('Start testing...')
    correct = 0
    for i in tqdm.tqdm(range(len(test_data))):
        if getClosest(train_data, train_label, test_data[i], topk, method) == test_label[i]:
            correct += 1
    return correct / len(test_data)

def main():
    # 加载训练和测试数据
    print('Start loading training data...')
    train_data, train_label = load_data('../mnist_train.csv')
    print('Start loading testing data...')
    test_data, test_label = load_data('../mnist_test.csv')

    # 为了节约时间，选择前两百个数据进行测试
    test_data = test_data[:200]
    test_label = test_label[:200]

    # 选择了欧几里得距离，k的值设定为25
    topk = 25
    method = 'Euclidean'

    # 打印结果
    start_time = time.time()
    accuracy = test(train_data, train_label, test_data, test_label, topk, method)
    end_time = time.time()
    print('Metric method: {}'.format(method))
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Testing time: {}'.format(end_time - start_time))

if __name__ == '__main__':
    main()