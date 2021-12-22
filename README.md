# Data-Science

# Table of Contents
* Overview
* Dataset
* Preprocessing and Validation
* Something Missing ?

# Overview

From the beginning, our aim was to develop a real time system where people those aren't so blessed like the rest of us can interact with any computer system but providing them with an interface through which they can do so using American Hand Signs.

The dataset we will be working on contains all the 26 alphabets as well as the basic american sign phrases such as "Hello", "I love you" and so on.
Our model is based on a convolutional neural networks architecture that will predict the sign given.

The goal behind this model is to ensure a successful communication between dumb/ deaf people and the rest of people especially on public places such as hospitals, banks, restaurants and so on.

# Dataset
our model is trained in recognizing hands signs with the best accuracy. For that we start with reading the dataset :
```
#Create dataframe of {paths, labels}
train_df = get_paths_labels('../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train')

#Import another dataset (to train model on various data)
temp_df = get_paths_labels('../input/asl-alphabet-test')

#Combine both datasets
dataset = pd.concat((train_df, temp_df))
```

# Preprocessing and Validation

After that, we start the preprocessing of our data so that we can build our neural network architecture successfully.

We used tensorflow keras for implementing our model, more precisely the pretrained model tf.keras.application.MobileNet with an input shape : input_shape=(224, 224, 3), and the output dense was about : 29 and with a 'softmax' activation.

```
# Neural network architecture

pretrainedModel = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
     include_top=False,
     weights='imagenet',
     pooling='avg'
)
pretrainedModel.trainable = False

inputs = pretrainedModel.input

x = tf.keras.layers.Dense(128, activation='relu')(pretrainedModel.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)

outputs = tf.keras.layers.Dense(29, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer_adam = tf.keras.optimizers.Adam(learning_rate = 0.005)

```


After compiling, fitting, Training & Validation <> Loss & Accuracy of our model, we Visualize classifications on validation set.

```
# Training & Validation <> Loss & Accuracy

%matplotlib inline
acc = np.array(history.history['accuracy'])
val_acc = np.array(history.history['val_accuracy'])
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

epochs = np.arange(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.scatter(epochs[val_acc.argmax()], val_acc.max(), color='green', s=70)
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.scatter(epochs[val_loss.argmin()], val_loss.min(), color='green', s=70)
plt.title('Training and validation loss')
plt.legend()

plt.show()

```

[![Capture.png](https://i.postimg.cc/Kv5zqdYz/Capture.png)](https://postimg.cc/JGGMtYZV)

# Something Missing ?

If you have ideas for more “How To” recipes that should be on this model, let us know or contribute some!




