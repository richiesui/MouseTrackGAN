import keras
import numpy as np
import matplotlib.pyplot as plt

filepath = 'E:/Workspace/GANproject/'

discriminator = keras.models.load_model(filepath + 'gan_keras/gan_discriminator.h5',compile=False)
generater = keras.models.load_model(
    filepath + 'gan_keras/gan_generater.h5', compile=False)
gd = keras.models.load_model(filepath + 'gan_keras/gan_gd.h5', compile=False)

x_random = np.load(filepath + 'gan_keras/x_random.npy')
x_coord = np.load(filepath + 'gan_keras/x_coord.npy')
x_edata = np.load(filepath + 'x_edata.npy')

generated_data = generater.predict([x_random,x_coord])

result1 = discriminator.predict(generated_data)
result2 = gd.predict([x_random, x_coord])
result3 = discriminator.predict(x_edata)

plt.subplot(311)
plt.hist(result1, bins=100)
plt.subplot(312)
plt.hist(result2, bins=100)
plt.subplot(313)
plt.hist(result3, bins=100)
plt.show()
