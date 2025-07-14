import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load and preprocess fMRI data (replace with actual data loading)
data = np.load('path_to_hcp_fMRI_data.npy')  # Example: timeseries data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

# Define VAE model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(data.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim),  # Mean and log variance
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(data.shape[1], activation='linear'),
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

# Loss function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_reconstructed = model.decode(z)
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_reconstructed))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_loss

# Training
vae = VAE(latent_dim=32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        loss = compute_loss(vae, x)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    return loss

# Train the model
epochs = 50
batch_size = 32
for epoch in range(epochs):
    for i in range(0, len(data_scaled), batch_size):
        batch = data_scaled[i:i+batch_size]
        loss = train_step(batch)
    print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

# Reconstruct and visualize
mean, logvar = vae.encode(data_scaled[:100])
z = vae.reparameterize(mean, logvar)
reconstructed = vae.decode(z).numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(data_scaled[0], label='Original')
plt.plot(reconstructed[0], label='Reconstructed')
plt.title('fMRI Timeseries Reconstruction')
plt.legend()
plt.show()