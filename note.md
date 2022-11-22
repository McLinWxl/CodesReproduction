# TensorFlow to PyTorch

## Define Model

```python
cnn_tanh = Sequential() 
cnn_tanh.add(Convolution1D(12,25,  input_shape=(L,dim), activation='tanh',name="cnn_1", padding='same'))
cnn_tanh.add(Convolution1D(6,15, activation='tanh',name="cnn_2", padding='same'))
cnn_tanh.add(Convolution1D(3,5, activation='tanh',name="cnn_4", padding='same'))
cnn_tanh.add(Convolution1D(1,3,activation='tanh',name="cnn_5", padding='same'))
cnn_tanh.compile(loss='mse', optimizer=optimizer)
cnn_tanh.summary()
```

```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

```

```

## Train Model

### TensorFlow

```python
model.fit(InputFeature, InputLabel, batch_size = , epochs = , validation_data =(InFea, InLab),  validation_split = #weight in validation set that devided to train, validation_freq = #epochs to validate)
```
