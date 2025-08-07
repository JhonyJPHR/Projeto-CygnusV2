# Arquivo: src/models/wgan_dcgan_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Dropout

class WGAN_GP:
    def __init__(self, input_dim, latent_dim=100, critic_extra_steps=5, gp_weight=10.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.critic_extra_steps = critic_extra_steps
        self.gp_weight = gp_weight

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

        self.critic = self.build_critic()
        self.generator = self.build_generator()

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))

        x = Dense(4 * 4 * 256, use_bias=False)(noise)
        x = BatchNormalization()(x)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((4, 4, 256))(x)

        x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)

        img = Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')(x)

        model = Model(noise, img)
        print("--- Generator (DCGAN) ---")
        model.summary()
        return model

    def build_critic(self):
        img_input = Input(shape=self.input_dim)

        x = Conv2D(64, kernel_size=4, strides=2, padding='same')(img_input)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)

        x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
        # CORREÇÃO AQUI
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.3)(x)
        
        x = Flatten()(x)
        validity = Dense(1)(x)

        model = Model(img_input, validity)
        print("--- Critic (DCGAN) ---")
        model.summary()
        return model

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        
        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(self.critic_extra_steps):
            with tf.GradientTape() as tape:
                noise = tf.random.normal(shape=(batch_size, self.latent_dim))
                fake_images = self.generator(noise, training=True)
                
                real_output = self.critic(real_images, training=True)
                fake_output = self.critic(fake_images, training=True)
                
                d_cost = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + self.gp_weight * gp
            
            d_grad = tape.gradient(d_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(noise, training=True)
            gen_output = self.critic(generated_images, training=True)
            g_loss = -tf.reduce_mean(gen_output)
        
        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        
        return d_loss, g_loss