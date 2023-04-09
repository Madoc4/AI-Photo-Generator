import tensorflow as tf
import tensorflow_datasets as tfds


def make_model():
    import tensorflow as tf
    import tensorflow_datasets as tfds

    def make_model():
        def make_model():
            # Generator model
            generator = tf.keras.Sequential([
                tf.keras.layers.Dense(8 * 8 * 1024, input_shape=(100 + 12,)),
                tf.keras.layers.Reshape((8, 8, 1024)),
                tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')
            ])

            # Discriminator model
            discriminator = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])

            # GAN model
            generator_input = tf.keras.layers.Input(shape=(100,))
            prompt_feature_input = tf.keras.layers.Input(shape=(12,))
            generated_image = generator(tf.keras.layers.concatenate([generator_input, prompt_feature_input]))

            discriminator.trainable = False
            validity = discriminator(generated_image)

            gan = tf.keras.models.Model(inputs=[generator_input, prompt_feature_input], outputs=[validity])
            discriminator.trainable = True

            # Compiles models
            generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
            discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                  optimizer=discriminator_optimizer)
            gan.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        optimizer=generator_optimizer)

            # Loads photos of humans from LFW dataset
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                "lfw",
                labels="inferred",
                label_mode="int",
                color_mode="rgb",
                batch_size=64,
                image_size=(64, 64),
                shuffle=True,
                seed=42,
                validation_split=0.2,
                subset="training"
            )

    # Trains GAN model
    for epoch in range(100):
        for batch in dataset:
            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch.shape[0], 100])
                prompt_feature = tf.one_hot(tf.random.uniform([batch.shape[0]], minval=0, maxval=12, dtype=tf.int32),
                                            depth=12)
                generated_images = generator(tf.concat([noise, prompt_feature], axis=1))
                real_images = batch
                combined_images = tf.concat([generated_images, real_images], axis=0)
                labels = tf.concat([tf.zeros([batch.shape[0], 1]), tf.ones([batch.shape[0], 1])], axis=0)
                labels += 0.05 * tf.random.uniform(tf.shape(labels))

                discriminator_loss = discriminator.train_on_batch(combined_images, labels)

            with tf.GradientTape() as tape:
                noise = tf.random.normal([batch.shape[0], 100])
                prompt_feature = tf.one_hot(tf.random.uniform([batch.shape[0]], minval=0, maxval=12, dtype=tf.int32),
                                            depth=12)
                generated_images = generator(tf.concat([noise, prompt_feature], axis=1))
                fake_labels = tf.ones([batch.shape[0], 1])

                generator_loss = gan.train_on_batch(tf.concat([noise, prompt_feature], axis=1), fake_labels)

            print(f"Epoch {epoch + 1}, Discriminator Loss: {discriminator_loss:.4f}, Generator Loss: {generator_loss:.4f}")

    # Saves model to an HDF5 file
    gan.save('gan_model.h5')

if __name__ == "__main__":
    make_model()