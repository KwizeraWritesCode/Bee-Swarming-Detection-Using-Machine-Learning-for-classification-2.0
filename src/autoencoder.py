from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return autoencoder