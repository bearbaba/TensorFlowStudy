# ����

`tensorflow`����Ϊ��Ҫ�������������㣬������`tensor`�࣬ÿ����������һ��`tensor`����

## �����Ĵ���


ʹ��`tf.constant()`����������`tf.constant()`���﷨��ʽΪ��

```python
tf.constant(value,dtype,shape)
```

`tensor`��ʵ�Ƕ�`numpy()`�ķ�װ��������ֵ���������֣�`shape`����Ϊ�գ����ڲ�ͬ��`value`��������`dtype`Ҳ�ǲ�ͬ�ġ�

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

![���н��](./1.png)

�����н���п��Կ���������һά������`dtype`��`int32`��С��
