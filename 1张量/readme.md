# 张量

`tensorflow`中的基本数据是`tensorflow`，可以看作是多维数组或列表类型。

## 张量的创建


使用`tf.constant()`创建张量，`tf.constant()`的语法格式为：

```python
tf.constant(value,dtype,shape)
```

`value`用来指定数据，`dtype`用来显式地声明数据类型，`shape`用来指定数据的形状，

例如，要生成一个两行三列全为数字2的
