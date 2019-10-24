import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, Conv1D, Concatenate, Reshape, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.optimizers as opts
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

filepath = 'E:/Workspace/GANproject/'
x_data = np.load(filepath + 'x_edata.npy')
y_data = np.load(filepath + 'y_edata.npy')

#定义生成器
random_input = Input(shape=(100,))
coord_input = Input(shape=(4,))
h1 = Dense(10000, activation='tanh')(random_input)
h1 = BatchNormalization()(h1)
h1 = Reshape((100, 100, 1))(h1)
h2 = Conv2D(1, 4, strides=2, padding='same', activation='tanh')(h1)
h2 = Reshape((2500,))(h2)
h2 = Concatenate()([coord_input, h2])
h3 = Dense(1000, activation='tanh')(h2)
h3 = BatchNormalization()(h3)
h3 = Reshape((1000, 1))(h3)
h4 = Conv1D(1, 4, padding='same',activation='tanh')(h3)
h4 = Reshape((1000,))(h4)
h5 = Dense(900, activation='elu')(h4)
output = Concatenate()([h5, coord_input])

generater = keras.models.Model(inputs=[random_input, coord_input], outputs=output)

#定义判别器(已预训练)
discriminator = keras.models.load_model(filepath + 'gan_keras/dnnModel.minLoss.h5', compile=False)

#定义GAN模型
input0 = Input(shape=(100,))
input1 = Input(shape=(4,))
output0 = generater([input0, input1])
output = discriminator(output0)
generater_discriminator = keras.models.Model(inputs=[input0, input1], outputs=output)


#训练生成器
genSize=30000
x_random = np.random.uniform(0, 1, [genSize, 100])
x_coord = np.random.uniform(-2000, 2000, [genSize, 4])
y_ones = np.ones((genSize,))
np.save(filepath + 'gan_keras/x_random.npy', x_random)
np.save(filepath + 'gan_keras/x_coord.npy', x_coord)

discriminator.trainable = False

gen_earlystopping = EarlyStopping(monitor='acc', patience=20, mode='max')

generater_discriminator.compile(optimizer=opts.adadelta(), loss='binary_crossentropy', metrics=['accuracy'])

gen_hist = generater_discriminator.fit(x=[x_random, x_coord], y=y_ones,batch_size=512, epochs=500, callbacks=[gen_earlystopping])
generater.save(filepath + 'gan_keras/gan_generater.h5')
generater_discriminator.save(filepath + 'gan_keras/gan_gd.h5')

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax1.plot(gen_hist.history['loss'], label='Loss', color='b')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper left')
ax2=ax1.twinx()
ax2.plot(gen_hist.history['acc'], label='Accuracy', color='r')
ax2.set_ylabel("Accuracy")
ax2.legend(loc='upper right')
plt.grid()

#训练判别器
disSize = 30000
x_false_random = np.random.uniform(0, 1, [disSize, 100])
x_false_coord = np.random.uniform(-2000, 2000, [disSize, 4])
x_false = generater.predict([x_false_random, x_false_coord])
y_zeros = np.zeros([disSize,])

x_dis_data = np.concatenate((x_data, x_false), axis=0)
y_dis_data = np.concatenate((y_data, y_zeros), axis=0)

x_train, x_test, y_train, y_test = train_test_split(
    x_dis_data, y_dis_data, test_size=0.3, random_state=0)

discriminator.trainable = True

dis_earlystopping1 = EarlyStopping(monitor='val_acc', patience=20, mode='max')
dis_earlystopping2 = EarlyStopping(monitor='val_loss', patience=20, mode='min')

callbacks_list = [dis_earlystopping1, dis_earlystopping2]

discriminator.compile(optimizer=opts.sgd(lr=0.05, momentum=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

dis_hist = discriminator.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20,
                 batch_size=512, verbose=1, callbacks=callbacks_list)
discriminator.save(filepath + 'gan_keras/gan_discriminator.h5')

ax3 = fig.add_subplot(212)
ax3.plot(dis_hist.history['loss'], label='Train_Loss', color='c')
ax3.plot(dis_hist.history['val_loss'], label='Eval_Loss', color='b')
ax3.set_ylabel('Loss')
ax3.set_xlabel('Epoch')
ax3.legend(loc='upper left')
ax4 = ax3.twinx()
ax4.plot(dis_hist.history['acc'], label='Train_Accuracy', color='pink')
ax4.plot(dis_hist.history['val_acc'], label='Eval_Accuracy', color='r')
ax4.set_ylabel("Accuracy")
ax4.legend(loc='upper right')
plt.grid()
plt.savefig(filepath + 'gan_keras/ganCurve.svg')
plt.show()
