# 张量

`tensorflow`中最为重要的是张量的运算，张量由`tensor`类，每个张量都是一个`tensor`对象

## 张量的创建


使用`tf.constant()`创建张量，`tf.constant()`的语法格式为：

```python
tf.constant(value,dtype,shape)
```

`tensor`其实是对`numpy()`的封装，张量的值可以是数字，`shape`可以为空，对于不同的`value`，张量的`dtype`也是不同的。

```python
import tensorflow as tf
a = tf.constant(value=1)
print(a)
b = tf.constant(value=1.0)
print(b)
c = tf.constant(value=1,shape=[2])
print(c)
d = tf.constant(value=2.0,shape=(2,2),dtype='float32)
```

![运行结果](./1.png)

从运行结果中可以看出，整型一维张量的`dtype`是`int32`，小数
