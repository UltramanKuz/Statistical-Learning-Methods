import numpy as np
import time as time
import tqdm as tqdm


def load_data(file_path):
    data = []
    label = []
    with open(file_path, 'r') as f:
        for line in f:
            cur_line = line.strip().split(',')
            label.append(int(cur_line[0]))
            data.append([int(i) for i in cur_line[1:]])
    return np.array(data), np.array(label)

def EuclideanDistance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def ManhattanDistance(x, y):
    return np.sqrt(np.sum(np.abs(x - y)))

def getClosest(train_data, train_label, x, topk, method):
    distance = []
    for sample in train_data:
        if method == 'Euclidean':
            distance.append(EuclideanDistance(sample, x))
        else:
            distance.append(ManhattanDistance(sample, x))
    index = np.argsort(np.array(distance))[:topk]
    label = np.array(train_label)[index]
    return np.argmax(np.bincount(label))

def test(train_data, train_label, test_data, test_label, topk, method):
    print('Start testing...')
    correct = 0
    for i in tqdm.tqdm(range(len(test_data))):
        if getClosest(train_data, train_label, test_data[i], topk, method) == test_label[i]:
            correct += 1
    return correct / len(test_data)

def main():
    print('Start loading training data...')
    train_data, train_label = load_data('../mnist_train.csv')
    print('Start loading testing data...')
    test_data, test_label = load_data('../mnist_test.csv')
    test_data = test_data[:200]
    test_label = test_label[:200]
    topk = 25
    method = 'Euclidean'
    start_time = time.time()
    accuracy = test(train_data, train_label, test_data, test_label, topk, method)
    end_time = time.time()
    print('Metric method: {}'.format(method))
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Testing time: {}'.format(end_time - start_time))

if __name__ == '__main__':
    main()