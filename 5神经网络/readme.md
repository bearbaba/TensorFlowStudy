# 神经网络

我们在之前的内容提到过，因为线性模型的表达能力比较弱，我们可以采用为他们嵌套多层，将每一层的输出作为下一层的输出，那么总的输出就可以表示为：

```latex
out = ReLU { ReLU{ReLU[X@W_1 + b_1]@W_2 + b_2}@W_3 + b_3}
```
ReLU 是激活函数的表示，上述是一个前向传播的算法。

从总体上说，一直都在使用的却是$z=w^Tx + b$算法，它实际上是一个线性的感知机模型，变量$b$被称为感知机的偏置，$w$被称为感知机的权值，$z$被称为感知机的净活性值。再添加上激活函数后：

$$
\alpha = \sigma(z) = \sigma(w^Tx + b)
$$

## 全连接层

我们通过堆叠多个网络层来增强网络的表达能力。在多个网络层由于每个输出节点与全部的输入节点相连接，因此这种网络层被称为全连接层，或者稠密连接层，$W$矩阵被叫做全连接层的权值矩阵，$b$向量叫做全连接层的偏置向量。

在 TensorFlow 中有`layers.Dense(units, activation)`，通过指定输出节点数`Units`和激活函数类型`activation`可以实现层的表示。

```python
from tensorflow.keras import layers
import tensorflow as tf

x = tf.random.normal([4, 28 * 28])
fc = layers.Dense(512, activation=tf.nn.relu)
h1 = fc(x)
print(fc.kernel)
print(fc.bias)
print(h1)
```

上述指定了输出节点为512的一层全连接层`fc`，我们可以通过类内部的成员名`kernel`和`bias`来获取权值张量$W$和偏置张量$b$对象。

## 层方式实现神经网络

我们通过层层堆叠全连接层，并保证前一层的输出节点数与当前层的输入节点数相匹配，可以堆叠出任意层数的网络，这种由神经元相互连接而成的网络就叫做神经网络。

我们下面通过堆叠4个全连接层，获得层数为4的神经网络，其中第1~3个全连接层在网络中间，称为隐藏层1、2、3，最后一个全连接层的输出作为网络输出，称为输出层。

```python
from tensorflow.keras import layers, Sequential
import tensorflow as tf

fc1 = layers.Dense(256, activation=tf.nn.relu)  # 隐藏层1
fc2 = layers.Dense(128, activation=tf.nn.relu)  # 隐藏层2
fc3 = layers.Dense(64, activation=tf.nn.relu)  # 隐藏层3
fc4 = layers.Dense(10, activation=tf.nn.relu)  # 输出层

x = tf.random.normal([4,28*28])
h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
h4 = fc4(h3)
```

上述中对隐藏层、输出层的使用可以使用`Sequential`容器封装成一个网络大类：

```python
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=None),
])
out = model(x)
print(out)
```

