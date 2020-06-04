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

�����������ĳά����ƴ�ӿ���ʹ��`tf.concat(tensors, axis)`������`tensors`�ǰ���Ҫ��ƴ�ӵ��������б�`axis`ָ�����ĸ����Ͻ���ƴ�ӡ�

ƴ�������ǽ����������ĳ��ά���Ͻ��кϲ�������������µ�ά�ȡ�

����

```python
# ָ����0����ƴ��
a = tf.constant(1,shape=(1,2))
b = tf.constant(2,shape=(1,2))
c = tf.concat([a,b],axis=0)
print(c)

# ָ����1����ƴ��
a = tf.constant(1,shape=(1,2))
b = tf.constant(2,shape=(1,2))
c = tf.concat([a,b],axis=1)
print(c)
```

![���н��](./img/18.png)

#### �ָ�����

�ָ�����������`tf.split(value, num_or_size_splits,axis=0)`������`values`�Ǵ��ָ�ı�����`num_or_size_splits`�Ƿָ�����ָ������������һ����ֵ����ʾ�ȳ��ָ��ֵ�Ƿָ�ķ�����Ҳ������һ���б���ʾ���ȳ��ָ�б������и��ÿ�ݵĳ��ȡ�����

```python
a = tf.constant(1, shape=(2, 4))
print("��0���Ϸָÿһ����2ʱ��\n", tf.split(a, num_or_size_splits=2, axis=0))
print("��1���Ϸָ�ָ����֮��Ϊ1��2��1��", tf.split(a, [1, 2, 1], 1))
```

![���н��](./img/19.png)

#### �ѵ�����

�ѵ�����ʹ��`tf.stack(values, axis)`������`values`��Ҫ�ѵ��Ķ��������`axis`ָ��������ά�ȵ�λ�á�

�ںϲ�����ʱ���ᴴ��һ���µ�ά�ȡ�����

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# ����Ϊ0�Ͻ���ƴ��
print(tf.stack([a, b], axis=0))

# ����Ϊ1�Ͻ���ƴ��
print(tf.stack([a, b], axis=1))
```

![���н��](./img/20.png)

#### �ֽ�����

�ֽ������������ѵ��������㣬ʹ��`tf.unstack(values, axis)`������

�����ᱻ�ֽ�Ϊ����������ֽ��õ���ÿ��������ԭ����������ȣ�ά�ȶ�����һά��

����

```python
a = tf.reshape(tf.range(6), shape=(2, 3))
b = tf.unstack(a, axis=0)
print(b)
```

![���н��](./img/21.png)

### ���ֲ���

#### ��������Ƭ

����Ҳ�����������б���������÷������磬����һ��һά������ȡ�������ݾͿ���������`a[0]`����ȡ�����ڶ�ά��������ͨ��`a[0, 0]`����`a[0][0]`����ȡ��һ�е�һ�е����ݡ�

```python
a = tf.constant([[[1, 2, 3],
          	[4, 5, 6]],
         	[[1, 2, 3],
          	[4, 5, 6]]])
print(a[0, 1, 1])
print(a[0])
```

![���н��](./img/22.png)

��Ƭ���÷�Ҳ���б����ƣ�����

```python
print(a[0][0][1:])
print(a[0][0:2][0:2])
print(a[0, 0:2, 0:2])
```

ע�⣬`a[0][0:2][0:2]`������Ƭ��ʽ�Ǵ���ģ�`a[0, 0:2, 0:2]`���ܵõ�������Ҫ����Ƭ�����ݣ����н�����£�

![���н��](./img/21.png)

������ʾ��`a[0][0:2][0:2]`ֻ����Ϊ1��ά�Ƚ�������Ƭ��

#### ������ȡ

`gather(params, indices)`����������һ�������б������������ж�Ӧ����ֵ��Ԫ����ȡ����������`params`���������������`indices`������ֵ�б�

������һά��������ȡ�����ֱ�Ϊ0��2��3��Ԫ�أ�

```python
a = tf.range(12, delta=2)
print(tf.gather(a, [0, 2, 3]))
```

![���н��](./img/24.png)

`tf.gather()`һ��ֻ�ܶ�һ��ά�Ƚ���������ȡ�����Դ���`axis`����ָ��Ҫ��ȡ��һά�ȡ�

`tf.gather_nd()`��������ͬʱ�Զ��ά�Ƚ���������ͨ��ָ����������������㣬

```python
a = tf.range(12, delta=2)
a = tf.reshape(a,(2,3))
print(tf.gather_nd(a, [[0,1],[1,1]]))
```

![���н��](./img/25.png)

## ��������

 ### �Ӽ��˳���������

|��������|����|
|:-:|:-:|
|`tf.add(x,y)`|��x��y��Ԫ�����|
|`tf.subtract(x,y)`|��x��y��Ԫ�����|
|`tf.multiply(x,y)`|��x��y��Ԫ�����|
|`tf.divide(x,y)`|��x��y��Ԫ�����|
|`tf.math.mod(x,y)`|��x��Ԫ������|

��Ԫ�ز�����ָ��x�е�ÿһ��Ԫ����y�е�ÿһ��Ԫ������ؽ������㣬����ʵ�����ǿ�������ѧ����`+-/*%`ȥ����ġ�����

```python
# ��Ԫ�����
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)
print(c)

# ��Ԫ�����
d = tf.subtract(b, a)
print(d)

# ��Ԫ�����
e = tf.multiply(a, b)
print(e)

# ��Ԫ�����
f = tf.divide(a, b)
print(f)

# ��Ԫ������
g = tf.math.mod(a, b)
print(g)

print(a - b)
print(a * b)
print(a / b)
print(a + b)
print(a % b)
print(a // b)
```

![���н��](./img/26.png)

��Ҫע�����`/`������Ȼ�����뱻������Ԫ�����Ͷ���`int32`�ģ����ǵõ��Ľ��ȴ��`float64`�������Ҫ�õ����������������������`//`�������������㡣

### ��ָ��������

|��������|��������|
|:-:|:-:|
|`tf.pow(x,y)`|��x��y���ݴη�|
|`tf.square(x)`|��x���Ԫ������η�|
|`tf.sqrt(x)`|��x��Ԫ�ؿ�ƽ����|
|`tf.exp(x)`|����e��x�η�|
|`tf.math.log(x)`|������Ȼ����������Ϊe|

```python
a = tf.reshape(tf.range(6), shape=(2, 3))
print(tf.pow(a, 2))
print(a ** 2)
print(tf.square(a))

print(tf.sqrt(tf.cast(a, dtype=tf.float32)))
print(tf.cast(a, dtype=tf.float32) ** (1 / 2))

print(tf.math.log(tf.cast(a, dtype=tf.float32)))
print(tf.exp(tf.cast(a,dtype=tf.float32)))
```

![���н��](./img/27.png)

������ʾ��������ÿ��Ԫ����ƽ����ƽ����Ҳ�ǿ�����`**`������ѧ���ŵģ�������Ҫע������ڼ�����Ȼ��������ƽ����ʱ��������������Ҫ�Ǹ����ͣ�ʹ�����ͻᱨ��

����`TensorFlow`��ֻ����eΪ�׵���Ȼ������û���ṩ������ֵΪ�׵Ķ������㺯�����������ǿ�������ѧ�����еĻ��׹�ʽ���������������Ķ�����$log_a b = \frac{log_c b}{log_c a}$

### ��������

![ʾ��ͼƬ](./img/28.png)

### ���Ǻ����뷴���Ǻ�������

![ʾ��ͼƬ](./img/29.png)

### ���������

![ʵ��ͼƬ](./img/30.png)

### �㲥���ƣ�broadcaing��

���磬һά�������ά�������ʱ��һά������ÿ��Ԫ�ػ����ά��������С���Ǹ�ά���е�ÿ��Ԫ�������ӡ�

![ʾ��ͼƬ](./img/31.png)

��Ҫע����������������һ��ά�ȵĳ��ȱ�����ȡ�

```python
a = tf.constant([1, 2, 3])
b = tf.reshape(tf.range(12), shape=(4, 3))
print(a + b)
```

![���н��](./img/32.png)

һά��������ά���������Ҳ��һά�������ά������ӵ������ͬ��������Сά��֮�����Ԫ����ӡ�

����+Nά����ʱ���������Ҳ���������ĸ���Ԫ�ؽ�����ӡ�

### �����˷�

Ԫ�س˷���`tf.multiply()`��������`*`��������棬

�����˷���`tf.matmul()`��������`@`��������档�����˷����õĳ˷������Դ����еľ���֮����˵����㡣

```python
a = tf.reshape(tf.range(6), shape=(2,3))
b = tf.reshape(tf.range(6), shape=(3,2))
print(a@b)
```

![���н��](./img/33.png)

���ڶ�ά�����е���ά��������ά������ͨ�˷�֮���������ѭ��ԭ���������λ�������˷�����ά���ù㲥���ƣ���ʵ��άҲ�ǣ���

һͼ˵����

![ʵ��ͼƬ](./img/34.png)

### ��������ͳ�ƺ���

��ͺ�����`tf.reduce_sum()`��������������趨`axis`�������Դ���ָ����ĳ��ά�Ƚ�����ͣ���������ã��ͻ�Ĭ�϶�����Ԫ�ؽ���������㡣

��֮���ƵĻ���`tf.reduce_mean()`���ֵ������`tf.max()`�����ֵ������`tf.min()`����Сֵ������`tf.argmax()`�����ֵ������`tf.argmin()`����Сֵ������

��Ҫע��������ֵ��������������Ϊ����ʱ�����õ��ľ�ֵ����Ҳ�����͡�

