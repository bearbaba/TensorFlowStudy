from sklearn.datasets import load_boston
import pandas as pd

boston=load_boston()
print('--- %s ---' % 'boston type')
print(type(boston))
print('--- %s ---' % 'boston keys')
print(boston.keys())
print('--- %s ---' % 'boston data')
print(type(boston.data))

print('--- %s ---' % 'boston target')
print(type(boston.target))
print('--- %s ---' % 'boston data shape')
print(boston.data.shape)

print('--- %s ---' % 'boston feature names')
print(boston.feature_names);

X = boston.data
y = boston.target
df = pd.DataFrame(X, columns=boston.feature_names)

print('--- %s ---' % 'df.head')
print(df.head())
print('--- %s ---' % 'df.info')
print(df.info())
print('--- %s ---' % 'df.describe')
print(df.describe())
