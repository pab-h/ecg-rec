import tensorflow as tf

from keras import layers

class ECGRecV1(tf.keras.Model):

    def __init__(self, latentDim, hiddenDim):
        super(ECGRecV1, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Conv1D(filters=hiddenDim, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv1D(filters=latentDim, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(filters=hiddenDim, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),

            layers.Conv1DTranspose(filters=12, kernel_size=4, strides=2, padding='same')
        ])

    def call(self, inputs):
        z = self.encoder(inputs)
        out = self.decoder(z)
        
        return out


class ECGRecV2(tf.keras.Model):

    def __init__(self, latentDim):
        super(ECGRecV2, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Conv1D(filters=32, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv1D(filters=64, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            layers.Conv1D(filters=latentDim, kernel_size=5, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(filters=128, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),

            layers.Conv1DTranspose(filters=64, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),

            layers.Conv1DTranspose(filters=32, kernel_size=4, strides=2, padding='same'),
            layers.ReLU(),

            layers.Conv1DTranspose(filters=12, kernel_size=4, strides=2, padding='same')
        ])

    def call(self, inputs):
        z = self.encoder(inputs)
        out = self.decoder(z)
        
        return out