import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image

# parameters
IMG_SIZE      = (256, 256)
STYLE_WEIGHT   = 1e3
CONTENT_WEIGHT = 1e-2
TV_WEIGHT      = 10.0
LEARNING_RATE  = 2.0

STYLE_LAYERS   = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                  'block4_conv1', 'block5_conv1']
CONTENT_LAYER  = 'block5_conv2'
ALL_LAYERS     = STYLE_LAYERS + [CONTENT_LAYER]

# image processing 
def load_image(image: Image.Image):
    img = image.convert("RGB").resize(IMG_SIZE)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

def unload_image(tensor):
    img = tensor[0].numpy().copy()
    img += [103.939, 116.779, 123.68]   
    img  = img[:, :, ::-1]              
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))

# model 
def get_model():
    vgg = VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False
    outs = [vgg.get_layer(n).output for n in ALL_LAYERS]
    return Model(inputs=vgg.input, outputs=outs)

# gram matrix 
def gram(tensor):
    g    = tf.linalg.einsum("bijc,bijd->bcd", tensor, tensor)
    size = tf.cast(tf.shape(tensor)[1] * tf.shape(tensor)[2], tf.float32)
    return g / size

# loss
def total_loss(model, gen, content, style):
    # get features for all three images in one call each
    gen_out     = model(gen)
    content_out = model(content)
    style_out   = model(style)

    # split style vs content layers
    gen_style,     gen_content     = gen_out[:-1],     gen_out[-1]
    _,             target_content  = content_out[:-1], content_out[-1]
    target_style,  _               = style_out[:-1],   style_out[-1]

    # style loss
    n          = len(STYLE_LAYERS)
    style_loss = sum(
        tf.reduce_mean(tf.square(gram(g) - gram(t)))
        for g, t in zip(gen_style, target_style)
    ) / n

    # content loss
    content_loss = tf.reduce_mean(tf.square(gen_content - target_content))

    # smoothness loss
    tv_loss = tf.reduce_sum(tf.image.total_variation(gen))

    return STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss + TV_WEIGHT * tv_loss

#  main function 
def run_style_transfer(content_img: Image.Image,
                       style_img:   Image.Image,
                       steps:       int = 200,
                       progress_cb  = None) -> Image.Image:

    model     = get_model()
    content   = load_image(content_img)
    style     = load_image(style_img)
    generated = tf.Variable(content, trainable=True, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(LEARNING_RATE)

    for step in range(steps):
        with tf.GradientTape() as tape:
            loss = total_loss(model, generated, content, style)

        grads = tape.gradient(loss, generated)
        optimizer.apply_gradients([(grads, generated)])
        generated.assign(tf.clip_by_value(generated, -103.939, 151.061))

        if progress_cb:
            progress_cb((step + 1) / steps)

    return unload_image(generated)