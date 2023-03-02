import tensorflow_datasets as tfds

train = tfds.load('cats_vs_dogs', split='train[:80%]')
val = tfds.load('cats_vs_dogs', split='train[80%:90%]')
test = tfds.load('cats_vs_dogs', split='train[90%:]')
print(len(train))
print(len(val))
print(len(test))