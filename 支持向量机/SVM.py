import time
import numpy as np
import math
import random
import tqdm
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
            # 归一化数据集
            data.append([int(i) / 255 for i in line[1:]])
            # 二分类问题，只判断是否为0类
            label.append(1 if int(line[0]) == 0 else -1)
    return data, label

class SVM:
    def __init__(self, train_data, train_label, C=200, sigma=10, toler=0.001):
        """
        初始化SVM模型，采用Gaussian核函数和SMO算法进行计算
        train_data: 训练数据
        train_label: 训练标签
        sigma: 高斯核函数的参数σ
        toler: 同时作为alpha的容许误差和ksi的松弛变量
        C: 软间隔惩罚参数

        """
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
        self.m, self.n = self.train_data.shape

        self.sigma = sigma
        self.toler = toler
        # 初始化alpha和b为0
        self.alpha = [0] * self.m
        self.b = 0

        self.C = C
        # 计算高斯核矩阵并保存
        self.Gaussian_kernel = self.calculate_kernel()
        # 初始化E为0
        # 此处很重要，详情见select_j函数
        self.E = [0] * self.m
        # 保存支持向量的索引列表
        self.support_vectors_index = []
    
    def calculate_kernel(self):
        """
        计算Gaussian核矩阵
        """
        kernel = np.zeros((self.m, self.m))
        for i in range(self.m):
            X = self.train_data[i]
            for j in range(i, self.m):
                Z = self.train_data[j]
                res = np.dot(X - Z, X - Z)  / (-2 * self.sigma ** 2)
                kernel[i, j] = np.exp(res)
                kernel[j, i] = np.exp(res)
        return kernel
    
    def calculate_g(self, i):
        """
        计算g(x_i)
        """
        # 只计算alpha不为0的样本
        index = [i for i, alpha_i in enumerate(self.alpha) if alpha_i != 0]
        g_x_i = 0
        for id in index:
            g_x_i += self.alpha[id] * self.train_label[id] * self.Gaussian_kernel[id][i]
        g_x_i += self.b
        return g_x_i
    
    def satisfy_KKT(self, i):
        """
        判断alpha_i是否满足KKT条件
        """
        g_x_i = self.calculate_g(i)
        y_i = self.train_label[i]
        alpha_i = self.alpha[i]
        # KKT条件：
        # 1、目标函数L对w求导 --> w = sum(alpha_i * y_i * x_i)
        # 2、目标函数L对b求导 --> b = y_j - sum(alpha_i * y_i * K(x_i, x_j))
        # 3、目标函数L对ksi求导 --> alpha_i + nu_i = C
        # 4、alpha_i * (y_i * g(x_i) - 1 + ksi_i) = 0
        # 5、y_i * g(x_i) >= 1 - ksi_i
        # 6、nu_i * ksi_i = 0
        # 7、ksi_i >= 0
        # 8、alpha_i, nu_i >= 0

        # 由KKT条件，a_i有以下三种情形满足条件
        # 第一种情形：alpha_i = 0 --> nu_i = C --> ksi_i = 0 -> y_i * g(x_i) >= 1
        # 第二种情形：alpha_i = C -->  y_i * g(x_i) - 1 + ksi_i = 0 --> y_i * g(x_i) = 1 - ksi_i <= 1
        # 第三种情形: 0 < alpha_i < C --> nu_i != 0 --> ksi_i = 0 --> y_i * g(x_i) = 1
        if (np.abs(alpha_i) < self.toler) and (y_i * g_x_i >= 1):
            return True
        elif (np.abs(alpha_i - self.C) < self.toler) and (y_i * g_x_i <= 1):
            return True
        elif (-self.toler < alpha_i < self.C + self.toler)  and (np.abs(y_i * g_x_i - 1) < self.toler):
            return True
        else:
            return False
        
    def calculate_E(self, i):
        """
        计算E_i
        """
        return self.calculate_g(i) - self.train_label[i]
    
    def select_j(self, i):
        """
        SMO算法第二步, 选择第二个变量alpha_j
        """
        # i为第一个变量，j为第二个变量
        E1 = self.calculate_E(i)
        E2 = 0
        # max_diff为|E1 - E2|的最大值
        max_diff = -1
        j = -1

        # 这一步是一个优化性的算法
        # 实际上书上算法中初始时每一个Ei应当都为-yi（因为g(xi)由于初始α为0，必然为0）
        # 然后每次按照书中第二步去计算不同的E2来使得|E1-E2|最大，但是时间耗费太长了
        # 原作者最初是全部按照书中缩写，但是测试下来函数在需要3秒左右，本人尝试了以下确实时间很长，所以进行了一些优化措施
        # 优化策略为初始化Ei为0，然后在挑选第二个变量时只考虑不为0的Ei，这样可以大大减少候选的index，从而减少时间

        # 如此以来相当于只考虑更新过Ei的变量，但是存在如下问题
        # 1.当程序刚开始运行时，所有Ei都是0，那挑谁呢？
        # 策略： 当程序检测到并没有Ei为非0时，将会使用随机函数随机挑选一个
        # 2.怎么保证能和书中的方法保持一样的有效性呢？
        # 策略：在挑选第一个变量时是有一个大循环的，它能保证遍历到每一个xi，并更新xi的值，
        # 在程序运行后期后其实绝大部分Ei都已经更新完毕了。下方优化算法只不过是在程序运行的前半程进行了时间的加速，在程序后期其实与未优化的情况无异

        # 只考虑更新过的Ei
        index = [i for i, E_i in enumerate(self.E) if E_i != 0]
        # 遍历候选项，选择其中使|E1 - E2|最大的j
        for k in index:
            tmp = self.calculate_E(k)
            if np.abs(E1 - tmp) > max_diff:
                max_diff = np.abs(E1 - tmp)
                E2 = tmp
                j = k
        # 开始阶段没有更新过的Ei时，随机选择j，这也是每次运行程序后精确率不同的原因
        if j == -1:
            j = i
            while j == i:
                j = random.randint(0, self.m - 1)
            E2 = self.calculate_E(j)
        return E2, j
    

    def train(self, max_iter=100):
        parameter_changed = 1

        for k in tqdm.tqdm(range(max_iter)):
            # 每次迭代时若没有变量发生变化，则认为迭代已然收敛，结束迭代
            if parameter_changed == 0:
                break
            parameter_changed = 0

            for i in range(self.m):
                if not self.satisfy_KKT(i):
                    # 选定两个变量i, j 以及它们对应的标签和alpha值
                    E1 = self.calculate_E(i)
                    E2, j = self.select_j(i)
                    y1 = self.train_label[i]
                    y2 = self.train_label[j]
                    alpha1 = self.alpha[i]
                    alpha2 = self.alpha[j]

                    # 计算L和H, 确定约束范围
                    if y1 != y2:
                        L = max(0, alpha2 - alpha1)
                        H = min(self.C, self.C + alpha2 - alpha1)
                    else:
                        L = max(0, alpha2 + alpha1 - self.C)
                        H = min(self.C, alpha2 + alpha1)
                    
                    if L == H: continue

                    # 调取存好的高斯核矩阵
                    K11 = self.Gaussian_kernel[i][i]
                    K22 = self.Gaussian_kernel[j][j]
                    K12 = self.Gaussian_kernel[i][j]
                    eta = K11 + K22 - 2 * K12

                    # 计算无约束下二次规划问题的最优解
                    alpha2_new = alpha2 + y2 * (E1 - E2) / eta
                    # 计算约束条件下的最优解
                    if alpha2_new > H:
                        alpha2_new = H
                    elif alpha2_new < L:
                        alpha2_new = L
                    
                    # 计算alpha1_new
                    alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)
                    # 更新alpha列表
                    self.alpha[i] = alpha1_new
                    self.alpha[j] = alpha2_new
                    
                    # 计算b_new
                    b1_new = -E1 - y1 * K11 * (alpha1_new - alpha1) - y2 * K12 * (alpha2_new - alpha2) + self.b
                    b2_new = -E2 - y1 * K12 * (alpha1_new - alpha1) - y2 * K22 * (alpha2_new - alpha2) + self.b
                    
                    # 若i或者j为支持向量，则b_new = b
                    if 0 < alpha1_new < self.C:
                        self.b = b1_new
                    elif 0 < alpha2_new < self.C:
                        self.b = b2_new
                    # 否则b取两者之间的值都可以，此处选定为均值
                    else:
                        self.b = (b1_new + b2_new) / 2

                    self.E[i] = self.calculate_E(i)
                    self.E[j] = self.calculate_E(j)

                    # 计算当前的alpha是否发生了足够大的变化，若是则记录变化参数数量
                    if np.abs(alpha2_new - alpha2) >= 0.00001:
                        parameter_changed += 1

        # 全部迭代完计算此时的支持向量
        for i in range(self.m):
            if self.alpha[i] > 0:
                self.support_vectors_index.append(i)
    
    def calculate_Gaussian_kernel(self, X, Z):
        """
        计算高斯核函数，用于判定分类
        """
        return np.exp(np.dot((X - Z), (X - Z))  / (-2 * self.sigma ** 2))

    def predict(self, x):
        """
        计算单个样本的预测值
        """
        # 计算预测值，为正则为正例，为负则为负例
        res = 0
        for i in self.support_vectors_index:
            res += self.alpha[i] * self.train_label[i] * self.calculate_Gaussian_kernel(self.train_data[i], x)
        res += self.b
        return 1 if res > 0 else -1
    
    def test(self, test_data, test_label):
        """
        在测试集上测试模型
        """
        error_cnt = 0
        for i in range(len(test_data)):
            if self.predict(test_data[i]) != test_label[i]:
                error_cnt += 1
        return 1 -  error_cnt/ len(test_data)


def main():         
    # 开始时间
    start = time.time()

    # 获取训练集
    trainDataList, trainLabelList = load_data('../mnist_train.csv')

    # 获取测试集
    testDataList, testLabelList = load_data('../mnist_test.csv')

    # 初始化SVM模型
    svm = SVM(trainDataList[:1000], trainLabelList[:1000])

    # 开始训练SVM模型
    print('Start training:')
    svm.train()
    print('Training is Done.')

    # 测试准确率
    print('Start testing:')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('The accuracy is:', accuracy)

    # 结束时间
    end = time.time()
    print('Time span:', end - start)
    print('Supporting vectors are:')
    print(svm.support_vectors_index)

if __name__ == '__main__':
    main()




            

