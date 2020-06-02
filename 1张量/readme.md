# ����

`tensorflow`�еĻ���������`tensorflow`�����Կ����Ƕ�ά������б����͡�

## �����Ĵ���


ʹ��`tf.constant()`����������`tf.constant()`���﷨��ʽΪ��

```python
tf.constant(value,dtype,shape)
```

`value`����ָ�����ݣ�`dtype`������ʽ�������������ͣ�`shape`����ָ�����ݵ���״��

���磬Ҫ����һ����������ȫΪ����`int32`������2������������ʹ�����´��룺

```python
import tensorflow as tf
a = tf.constant(3,dtype=tf.int32,shape=(2,3))
print(a)
```

![ʾ��ͼƬ](./img/1.png)

��������`tensor`�е���������Ĭ����`tf.int32`�ģ�`dtype`���Բ�����ʽ��ָ����

`tensorflow`�����ɵ�������������`numpy`����ת���ɶ�Ӧ�����ݣ�����

```python
import tensorflow as tf
a = tf.constant(3,dtype=tf.int32,shape=(2,3))
print(a)
print(a.numpy())
```

![���н��](./img/2.png)

`tensor`�����������У�

![ʵ��ͼƬ](./img/3.png)

`constant`������`value`�������˿����������⣬��������`numpy`���飬����

```python
import numpy as np
b=np.array([1,2,3])
c=tf.constant(b)
print(c)
```

![���н��](./img/4.png)

### ȫ0������ȫ1�����Ĵ���

ʹ��`tf.zeros`��`tf.ones`�������д������﷨��ʽ��

```python
tf.zeros(shape,dtype = tf.float32)
tf.ones(shape,dtype = tf.float32)
```

����

```python
b = tf.zeros(2)
c = tf.ones([2, 3])
print("b=", b)
print("c=", c)
```

![���н��](./img/7.png)

���Ҫָ��ά�ȴ���2�����������Խ�������д�����飬��`c = tf.ones([2, 3])`���������������е�ȫһ������

### ����Ԫ��ֵ����ͬ������

`tf.fill()`�������ڴ���ֵ����ͬ���������﷨��ʽΪ��

```python
tf.fill(dims,value)
```

`tf.fill()`����û��`dtype`�����������ݴ��ݽ��Ĳ����Զ��ж��������͵ġ�`dims`����ָ����״������

```python
d = tf.fill(dims=[2,3],value=9)
print("d=",d)
```

![���н��](./img/8.png)

### �������������

#### ��̬�ֲ�

�﷨��ʽ��

```python
tf.random.normal(shape,mean,stddev,dtype)
```

`mean`��Ϊ��ֵ��`stddev`Ϊ��׼�

��������2X2����̬�ֲ������������

```python
e = tf.random.normal([2, 2])
print("e=",e)
```

![���н��](./img/9.png)

#### �ض���̬�ֲ�

�﷨��ʽ��

```python
tf.random.truncated_normal(shape, mean, stddev,)
```

����ֵ��һ���ضϵ���̬�ֲ����ضϵı�׼��2���ı�׼�

#### �������ȷֲ�����

�﷨��ʽ��

```python
tf.random.uniform(shape, minval, maxval, dtype)
```

`minval`��ʾ��Сֵ��`maxval`��ʾ���ֵ��ǰ�պ󿪣����������ֵ��

����

```python
f = tf.random.uniform(shape=[2, 3], minval=0, maxval=10)
print("f=", f)
```

![ʵ��ͼƬ](./img/10.png)

####  �������

`tf.random.shuffle(x)`������Ϊ�����������`x`�ĵ�һά������ʵ���ǿ��Դ����б��������ģ����Һ󷵻��������͡�

#### ��������

`tf.range()`�������Դ������У��÷���python�е�`range`�������ơ��﷨��ʽ��

```python
tf.range(start, limit, delta=1, dtype)
```

`start`��`limit`�ֱ��ʾ��ʼ������������֣�ǰ�պ󿪣�`delta`��ʾ������


### �ı�������������

ʹ��`tf.cast`���Ըı��������������ͣ��﷨��ʽΪ��

```python
tf.cast(x,dtype)
```

������`tf.int32`�ı��`tf.float32`����

```python
a = tf.constant(12,dtype=tf.int32,shape=(2,3))
tf.cast(a,dtype=tf.float32)
print(a)
```

`tf.convert_to_tensor`�����ܽ��������͵�Python����ת��Ϊ�����������������������������顢Python�б��Python������

```python
a = [i for i in range(10)]
print("a_type=",type(a))
b = tf.convert_to_tensor(a)
print(b)
```

![���н��](./img/6.png)

### tensor���������

����ֱ�����������`ndim`��ά�ȣ���`shape`��`dtype`���ԣ�����

```python
a = tf.constant(value=2, shape=(2, 3), dtype=tf.float32)
print(a.ndim)
print(a.dtype)
print(a.shape)
```

![ʵ��ͼƬ](./img/11.png)

Ҳ����ʹ��`tensorflow`��`size`��`rank`��`shape`�������õ������ĳ��ȡ�ά�ȡ���״���ԡ�

```python
print(tf.size(a))
print(tf.shape(a))
print(tf.rank(a))
```

![ʾ��ͼƬ](./img/12.png)

## ������ز���

��ά��������������һά�ķ�ʽ�����洢��ͨ������ά�Ⱥ���״�����߼��ϰ������Ϊ��ά�����������ڶ�ά����Ҳͬ�����ã�

### ά�ȱ任

#### �ı�������״

ʹ��`tf.reshape`�ı���������״���﷨��ʽ��

```python
tf.reshape(tensor,shape)
```

������һά����ת������ά������

```python
a = tf.range(24)
print(a)
print(tf.reshape(a, [2, 3, 4]))
```

![ʾ��ͼƬ](./img/13.png)

#### ��ά��������

��ά������ά�ȱ�ʾ��ά�������ᣨ`axis`����

![ʾ��ͼƬ](./img/14.png)

���������������ֵ����ά����ߵ���ͣ�ʹ������0��Ϊ�������Ŀ�ʼ�����������Ϊ��������㵽���ڲ��˳�򣩡�Ҳ����ʹ�ø�����Ϊ������������-1��ʾ����ֵ�����ᣬ��һ����Python���б�һ�¡�

#### ����ά��

ʹ��`tf.expand_dims(input, axis)`��������ά�ȣ��﷨��ʽ��

```python
tf.expand_dims(input, axis)
```

ʾ��������һ��`shape`Ϊ��2��2����������Ϊ����`axis=0`���������һ��ά�ȣ�

```python
a = tf.constant([[1,2],[3,4]])
print(tf.shape(a))
a = tf.expand_dims(a, axis=0)
print(a)
```

![���н��](./img/15.png)

��Ҫע����ǣ�`tf.expand_dims()`����Ҫ��ȷָ��`axis`��ֵ��ָ�����ĸ������ԭ��������������һ��ָ���ᡣ�����������ָ��`axis=1`����ôa��`shape`�ͻ���(2, 1, 2)��

#### ɾ��ά��

ɾ��ά����`tf.squeeze()`�������ú����﷨��ʽ��

```python
tf.squeeze(input, axis=None)
```

�ú���ֻ��ɾ������Ϊ1��ά�ȣ�����ָ��`axis`�������ȷ��ָ��`axis`����ɾ�����г���Ϊ1��ά�ȣ�����ʹ���б���ָ�����Ҫɾ����ά�ȡ�

����ԭ����`shape`Ϊ(1, 2 , 3, 1,  1, 4)������ֱ�ָ��ɾ��ȫ������Ϊ1��ά�Ⱥͣ�����Ϊ��

```python
a = tf.constant(1, shape=(1, 2, 3, 1, 1, 4))
print(a.shape)
b = tf.squeeze(a)
c = tf.squeeze(a, [0, 3])
print("b_shape={},c_shape={}".format(b.shape, c.shape))
```

![���н��](./img/16.png)

#### ����ά��

����ά��ʹ��`tf.transpose(a, perm)`������`perm`ָ��ά�ȵ�˳������ԭ����״Ϊ��1��2��3��������ά��˳���ǣ�0��1��2��������ʹ��`tf.transpose()`������Ϊ0���������Ϊ1������н�����

```python
a = tf.constant(1, shape=(1, 2, 3))
a = tf.transpose(a, perm=[1, 0, 2])
print(a)
```

![���н��](./img/17.png)

���н����ʾԭ������`shape`Ϊ(1, 2, 3)��������״Ϊ(2, 1, 3)��

#### ƴ������

�����������ĳά����ƴ�ӿ���ʹ��`tf.concat(tensor, axis)`������`axis`ָ�����ĸ����Ͻ���ƴ�ӡ�

ƴ�������ǽ����������ĳ��ά���Ͻ��кϲ�������������µ�ά�ȡ�

����

```python
