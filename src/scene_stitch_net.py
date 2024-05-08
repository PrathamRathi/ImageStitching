import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class SceneStitchNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.architecture = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1, 2), padding="same"),
            tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(filters=6, kernel_size=3, strides=(1, 1), padding="same"),
            tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=(1, 1), padding="same", activation="sigmoid")
        ])
    def call(self, inputs):
        outputs = self.architecture(inputs)
        return outputs
    def validate(self):
        tot_loss = 0
        for i in range(30):
            batch = np.load("validation_arr_data_batchsize_50/arr_" + str(i) + ".npy")
            x = np.concatenate((batch[:, :, :50, :], batch[:, :, 100:, :]), axis=2)
            y = batch[:, :, 50:100, :]
            pred = self.call(x)
            loss = self.loss_fn(pred, y)
            tot_loss += loss
        avg_loss = np.float32(tot_loss / 30.0)
        print("avg_valid_loss:", avg_loss)
    def train(self, epoch_num):
        tot_loss = 0
        for i in range(200):
            batch = np.load("train_arr_data_batchsize_50/arr_" + str(i) + ".npy")
            x = np.concatenate((batch[:, :, :50, :], batch[:, :, 100:, :]), axis=2)
            y = batch[:, :, 50:100, :]
            with tf.GradientTape() as tape:
                pred = self.call(x)
                loss = self.loss_fn(pred, y)
            tot_loss += loss
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        avg_loss = np.float32(tot_loss / 200.0)
        print("epoch_num:", epoch_num, ",", "avg_train_loss:", avg_loss)
        self.validate()

model = SceneStitchNet()
num_epochs = 20
model.validate()
for epoch_num in range(num_epochs):
    model.train(epoch_num)
test_batch = np.load("test_arr_data_batchsize_50/arr_0.npy")
test_x = np.concatenate((test_batch[:, :, :50, :], test_batch[:, :, 100:, :]), axis=2)
test_y = test_batch[:, :, 50:100, :]
test_pred = model.predict(test_x)
for i in range(5):
    plt.imshow(test_pred[i, :, :, :])
    plt.show()
    plt.imshow(test_y[i, :, :, :])
    plt.show()
