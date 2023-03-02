import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import tensorflow_datasets as tfds

(train, val, test), info = tfds.load('cats_vs_dogs', 
                                     split=['train[:80%]','train[80%:90%]','train[90%:]'], 
                                     as_supervised=True, 
                                     with_info=True)

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

def format_image(image, label):
    image = tf.image.resize(image, (150,150))/255.0
    return image, label

BATCH_SIZE = 64
EPOCHS=1
train_batches = train.shuffle(num_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)
val_batches = val.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test.map(format_image).batch(1)

model = Sequential([
    layers.Conv2D(16,3, padding='same', activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

model.build()
print(tf.config.list_physical_devices('GPU'))
history = model.fit(train_batches,
                    validation_data=val_batches,
                    epochs=EPOCHS)

model.save('model.h5')

loss, acc = model.evaluate(test_batches)
print('Loss: {}, Acc: {}'.format(loss, acc))