import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
import cv2
import numpy as np

def normalize_L(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0
    return img

def normalize_AB(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) * 2.0
    return img

def load_l_ab(L_path, AB_path):
    L_img = cv2.imread(L_path.decode(), cv2.IMREAD_GRAYSCALE)
    AB_img = cv2.imread(AB_path.decode(), cv2.IMREAD_COLOR)

    if L_img is None or AB_img is None:
        raise ValueError(f"Error loading {L_path} or {AB_path}")

    L_img = cv2.resize(L_img, (256, 256))
    AB_img = cv2.resize(AB_img, (256, 256))
    AB_img = AB_img[:, :, :2]  # Take only A and B channels

    L_img = normalize_L(L_img)
    AB_img = normalize_AB(AB_img)
    L_img = np.expand_dims(L_img, axis=-1)

    return L_img, AB_img

def load_dataset(L_folder, AB_folder, batch_size=8):
    L_paths = sorted(glob.glob(os.path.join(L_folder, "*.png")))
    AB_paths = sorted(glob.glob(os.path.join(AB_folder, "*.png")))

    assert len(L_paths) == len(AB_paths), "Mismatch between L and AB image counts"

    dataset = tf.data.Dataset.from_tensor_slices((L_paths, AB_paths))

    def process_path(L_path, AB_path):
        L_img, AB_img = tf.numpy_function(
            load_l_ab, [L_path, AB_path], [tf.float32, tf.float32]
        )
        L_img.set_shape([256, 256, 1])
        AB_img.set_shape([256, 256, 2])
        return L_img, AB_img

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_generator():
    inputs = layers.Input(shape=[256, 256, 1])
    
    # Encoder
    down1 = layers.Conv2D(64, 4, strides=2, padding="same")(inputs)
    down1 = layers.LeakyReLU()(down1)
    
    down2 = layers.Conv2D(128, 4, strides=2, padding="same")(down1)
    down2 = layers.BatchNormalization()(down2)
    down2 = layers.LeakyReLU()(down2)
    
    down3 = layers.Conv2D(256, 4, strides=2, padding="same")(down2)
    down3 = layers.BatchNormalization()(down3)
    down3 = layers.LeakyReLU()(down3)
    
    # Decoder
    up1 = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(down3)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.ReLU()(up1)
    up1 = layers.Concatenate()([up1, down2])
    
    up2 = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(up1)
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.ReLU()(up2)
    up2 = layers.Concatenate()([up2, down1])
    
    outputs = layers.Conv2D(2, 1, activation="tanh")(up2)
    
    return keras.Model(inputs=inputs, outputs=outputs)

class Pix2Pix(keras.Model):
    def __init__(self, generator, discriminator, lambda_L1=100.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.lambda_L1 = lambda_L1

    def compile(self, gen_optimizer, disc_optimizer, gen_loss_fn, disc_loss_fn):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn

    def train_step(self, batch_data):
        input_L, real_AB = batch_data
        
        with tf.GradientTape(persistent=True) as tape:
            fake_AB = self.generator(input_L, training=True)
            
            disc_real = self.discriminator([input_L, real_AB], training=True)
            disc_fake = self.discriminator([input_L, fake_AB], training=True)
            
            gen_adv_loss = self.gen_loss_fn(tf.ones_like(disc_fake), disc_fake)
            l1_loss = tf.reduce_mean(tf.abs(real_AB - fake_AB))
            gen_total_loss = gen_adv_loss + self.lambda_L1 * l1_loss
            
            disc_real_loss = self.disc_loss_fn(tf.ones_like(disc_real), disc_real)
            disc_fake_loss = self.disc_loss_fn(tf.zeros_like(disc_fake), disc_fake)
            disc_total_loss = (disc_real_loss + disc_fake_loss) * 0.5

        gen_gradients = tape.gradient(gen_total_loss, self.generator.trainable_variables)
        disc_gradients = tape.gradient(disc_total_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return {"gen_total_loss": gen_total_loss, "disc_total_loss": disc_total_loss}

if __name__ == "__main__":
    # Load dataset
    train_dataset = load_dataset('dataset/L', 'dataset/AB')
    
    # Build and compile model
    generator = build_generator()
    discriminator = build_discriminator()
    
    pix2pix = Pix2Pix(generator, discriminator)
    pix2pix.compile(
        gen_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
        disc_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
        gen_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
        disc_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    
    # Train
    pix2pix.fit(train_dataset, epochs=50)
    
    # Save generator
    os.makedirs('saved_model', exist_ok=True)
    generator.save('saved_model/generator.keras')
    print("Generator saved successfully")