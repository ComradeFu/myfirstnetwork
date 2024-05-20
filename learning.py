import numpy as np
import gen
import copy
import math

BATCH_SIZE = 30
LEARNING_RATE = 0.003
NETWORK_SHAPE = [2, 100, 20, 15, 2]

def create_weights(n_inputs, n_neurons):
    return np.random.randn(n_inputs, n_neurons)

def create_biases(n_neurons):
    return np.random.randn(n_neurons)

# sigmoid 可替换，ReLU 不适用于输出层
def activation_ReLU(inputs):
    return np.maximum(0, inputs)

# 适用于输出层，原理是利用 f(y) = e^x  函数，等距离 x 的 y 比值一样，来压缩指数爆炸问题。
# 同时利用上矩阵的运算加速
def activation_softmax(inputs):
    max_value = np.max(inputs, axis=1, keepdims=True)
    biase_value = inputs - max_value # 这里用到的是 np 自己的广播运算，不是真的矩阵规则，这一步让所有的值都 < 0
    exp_value = np.exp(biase_value) # 这一步全部压缩成 (0 - 1] 的值
    normal_base = np.sum(exp_value, axis=1, keepdims=True)
    normal_value = exp_value / normal_base # 同样是 np 的广播

    return normal_value

# 标准化（也可以用上面的 exp 方法）
# 目的是将可能很小的值或者很大的值，等比例缩放到 0-1，这样不会连乘导致梯度消失或者梯度爆炸
def normalize(inputs):
    max_values = np.max(np.absolute(inputs), axis=1, keepdims=True)
    scale_factor = np.where(max_values == 0, 0, 1 / max_values)
    return inputs * scale_factor

def normalize_vector(inputs):
    max_values = np.max(np.absolute(inputs))
    scale_factor = np.where(max_values == 0, 0, 1 / max_values)
    return inputs * scale_factor

# 分类函数
def classify(probabilities):
    # 用第二列数据做判断，rount int 四舍五入，> 0.5 就是 1， < 0.5 就是 0
    return np.rint(probabilities[:, 1]) 

# 交叉熵损失函数（ 0 - 1 向量的点乘）, precise 是精准含义
def precise_loss_function(predicted, real):
    real_matrix = np.zeros((len(real), 2)) # 填充一列，从 [0, 1] 变成 [[1, 0], [0, 1]] 好比较
    real_matrix[:, 1] = real
    real_matrix[:, 0] = 1 - real

    return 1 - np.sum(real_matrix * predicted, axis=1)

# 粗略损失函数，用来不这么严格进行评估，进行四舍五入。两个函数有改进就能用。
# 不是必须，但无坏处
def loss_function(predicted, real):
    condition = (predicted > 0.5)
    binary_predicted = np.where(condition, 1, 0)

    return precise_loss_function(binary_predicted, real)


# 最后一层输出层的需求函数（需求激活函数前怎么变）
def get_final_layer_preAct_demands(predicted_values, target_vector):
    target = np.zeros((len(target_vector), 2))
    target[:, 1] = target_vector
    target[:, 0] = 1 - target_vector

    for i in range(len(target_vector)):
        if target[i] @ predicted_values[i] > 0.5 :
            target[i] = np.array([0, 0]) # 不用变，准
        else:
            target[i] = (target[i] - 0.5) * 2 # 对应变 [-1, 1] 或者 [1, -1] 分辨拉大或者降低
    return target
        

# 层
class Layer:
    def __init__(self, n_inputs, n_neurons, is_output_layer = False):
        self.weights = create_weights(n_inputs, n_neurons)
        self.biases = create_biases(n_neurons)
        self.is_output_layer = is_output_layer

    # 向前传播
    def forward(self, inputs):
        self.output = output = inputs @ self.weights  + self.biases
        if(not self.is_output_layer):
            return normalize(activation_ReLU(output)) # ReLu > 0 可能会爆炸，所以标准化一下
        else:
            return activation_softmax(output)
    
    # 向后传播
    def backward(self, pre_values, demands):
        # ReLU 导数，< 0 = 0, > 0 = 1
        condition = (pre_values > 0)
        value_derivatives = np.where(condition, 1, 0)
        
        # 反过来
        pre_output = demands @ self.weights.T

        pre_demands = pre_output * value_derivatives # 链式求导，花乘
        # 多条 batch ，每一条神经元的 demand 归一化防止爆炸
        normal_pre_demands = normalize(pre_demands)

        # 同时也标准化一下调整的权重
        adjust_weight_martix = self.get_weight_adjust_matrix(pre_values, demands)
        normal_adjust_weight_martix = normalize(adjust_weight_martix)

        return (normal_pre_demands, normal_adjust_weight_martix)

    def get_weight_adjust_matrix(self, pre_values, demands):
        weight_adjust_matrix = np.zeros(self.weights.shape)
        # 多结果直接取平均，不等待链式
        for i in range(len(pre_values)):
            weight_adjust_matrix += pre_values[i, :].reshape(-1, 1) @ demands[i, :].reshape(1, -1)
        weight_adjust_matrix /= len(pre_values)

        return weight_adjust_matrix

# 网
class Network:
    def __init__(self, network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape) - 1):
            is_output_layer = True if i == (len(network_shape) - 2) else False
            layer = Layer(network_shape[i], network_shape[i + 1], is_output_layer)
            self.layers.append(layer)

    def forward(self, inputs):
        outputs = [inputs]
        last_inputs = inputs
        for layer in self.layers:
            last_inputs = layer.forward(last_inputs)
            outputs.append(last_inputs)
        
        return outputs
    
    # 反向传播
    def backward(self, layer_outputs, target_vector):
        new_network = copy.deepcopy(self)
        demands = get_final_layer_preAct_demands(layer_outputs[-1], target_vector)
        for i in range(len(new_network.layers)):
            # 倒序
            layer = new_network.layers[-i - 1]
            # 输出层不能调整 biases ，不然当样本较多同类时，极其容易过拟合
            if(i != 0):
                layer.biases += LEARNING_RATE * np.mean(demands, axis=0)
                layer.biases = normalize_vector(layer.biases)

            result = layer.backward(layer_outputs[-i - 2], demands)
            demands = result[0]
            weights_adjust_matrix = result[1]
            layer.weights += LEARNING_RATE * weights_adjust_matrix
            layer.weights = normalize(layer.weights)

        return new_network
    
    # 批次训练
    def batch_train(self, data):
        inputs = data[:, (0, 1)]
        tags = copy.deepcopy(data[:, 2]) # python always reference
        outputs = self.forward(inputs)
        precise_loss = precise_loss_function(outputs[-1], tags) # 结果更准 
        loss = loss_function(outputs[-1], tags) # 意义更准

        if(np.mean(precise_loss < 0.05)):
            print("No need to train.")
        else:
            # 只是一个优化，其实不一定必须
            new_network = self.backward(outputs, tags)
            new_outputs = new_network.forward(inputs)

            new_precise_loss = precise_loss_function(new_outputs[-1], tags)
            new_loss = loss_function(new_outputs[-1], tags)

            print(np.mean(precise_loss), np.mean(new_precise_loss), np.mean(loss), np.mean(new_loss))
            if(np.mean(precise_loss) > np.mean(new_precise_loss) or np.mean(loss) > np.mean(new_loss)):
                for i in range(len(self.layers)):
                    self.layers[i].weights = new_network.layers[i].weights.copy()
                    self.layers[i].biases = new_network.layers[i].biases.copy()
                print('Improved!')
            else:
                print('Unimproved!')
    
    def train(self, data):
        batch_times = math.ceil(len(data) / BATCH_SIZE)
        for i in range(batch_times):
            batch_data = data[(i * BATCH_SIZE, min(len(data) - 1, (i + 1) * BATCH_SIZE - 1)), : ]
            self.batch_train(batch_data)
    
data = gen.creat_data(10001)
network = Network(NETWORK_SHAPE)
network.train(data)

data_test = gen.creat_data(100)
input_test = data_test[:, (0, 1)]
gen.plot_data(data_test, "tag data")

outputs = network.forward(input_test)
classification = classify(outputs[-1])
data_test[:, 2] = classification
gen.plot_data(data_test, "train data")
