import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Define the hyperparameters
batch_size = 32
latent_dim = 100
epochs = 1


# Define the generator network
def build_generator(latent_dim):
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(128 * 7 * 7, activation='relu')(inputs)
    x = tf.keras.layers.Reshape((7, 7, 128))(x)
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(1, (7, 7), padding='same', activation='tanh')(x)
    generator = tf.keras.models.Model(inputs, x)
    return generator


# Define the discriminator network
def build_discriminator():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    discriminator = tf.keras.models.Model(inputs, x)
    return discriminator


# Compile the models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.trainable = False
inputs = tf.keras.layers.Input(shape=(latent_dim,))
generated_images = generator(inputs)
outputs = discriminator(generated_images)
combined_model = tf.keras.models.Model(inputs, outputs)
combined_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

# Load the input images
input_dir = 'Desktop/Coding/AI-Photo-Generator/Sample-Images'
input_images = []
for filename in os.listdir(input_dir):
    image = Image.open(os.path.join(input_dir, filename))
    image = image.resize((28, 28))
    image = np.array(image)
    input_images.append(image)
input_images = np.array(input_images)

# Train the model
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, input_images.shape[0], batch_size)
    real_images = input_images[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_images, real)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined_model.train_on_batch(noise, real)

    # Print the progress
    print(f"Epoch {epoch + 1}, Discriminator Loss: {d_loss:.4f}, Discriminator Accuracy: {d_acc:.2f}, Generator Loss: {g_loss:.4f}")

    # Save the models
    generator.save('generator.h5')
    discriminator.save('discriminator.h5')

    # Generate new images
    num_images = 10
    noise = np.random.normal(0, 1, size=(num_images, latent_dim))
    generated_images = generator.predict(noise)

    # Rescale image size
    generated_images = (generated_images * 127.5 + 127.5).astype(np.uint8)

    # Save images
    output_dir = 'Desktop/Coding/AI-Photo-Generator/Result-Images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(num_images):
        image = Image.fromarray(generated_images[i, :, :, 0], mode='L')
        image.save(os.path.join(output_dir, f"generated_{i + 1}.png"))
    print(f"{num_images} images generated and saved in {output_dir} directory")

