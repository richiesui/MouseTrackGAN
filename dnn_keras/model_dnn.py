import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense, BatchNormalization,Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

filepath = 'E:/Workspace/GANproject/'
x_data = np.load(filepath + 'x_edata.npy')
y_data = np.load(filepath + 'y_edata.npy')

# 生成数据集
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=0)

# 生成keras顺序模型
model = keras.models.Sequential()
model.add(Dense(4096, activation='elu', input_dim=904))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='elu'))
model.add(BatchNormalization())
model.add(Dense(256, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# 设置优化器
model.compile(optimizer=keras.optimizers.SGD(lr=0.05,momentum=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 保存点,验证集loss最小时保存
checkpoint1 = ModelCheckpoint(filepath+'dnn_keras/dnnModel.minLoss.h5', monitor='val_loss', verbose=0,
                              save_best_only=True, save_weights_only=False, mode='min')

# 验证集accuracy 50轮停止上升时结束训练
earlystopping1 = EarlyStopping(monitor='val_acc', patience=50, mode='max')

# 验证集loss 50轮停止下降时结束训练
earlystopping2 = EarlyStopping(monitor='val_loss', patience=50, mode='min')

callbacks_list = [checkpoint1, earlystopping1, earlystopping2]

# 开始训练模型
hist = model.fit(x_train, y_train,validation_data=(x_test,y_test), epochs=1000,
                 batch_size=512, verbose=1, callbacks=callbacks_list)

# 绘制训练 & 验证的图像
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.plot(hist.history['acc'], label='Train', color='c')
plt.plot(hist.history['val_acc'], label='Eval', color='r')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.subplot(212)
plt.plot(hist.history['loss'], label='Train', color='c')
plt.plot(hist.history['val_loss'], label='Eval', color='r')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.savefig(filepath + 'dnn_keras/dnnCurve.svg')
plt.show()
