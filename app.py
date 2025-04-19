import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Load VGG19 model
model = VGG19(include_top=False, weights='imagenet')
model.trainable = False

# Define content and style layers
content_layer = 'block5_conv2'
content_model = Model(inputs=model.input,
                      outputs=model.get_layer(content_layer).output)

style_layers = ['block1_conv1', 'block3_conv1', 'block5_conv1']
style_models = [Model(inputs=model.input, outputs=model.get_layer(
    layer).output) for layer in style_layers]
weight_of_layer = 1. / len(style_models)


def process_image(img):
    # Convert to array and preprocess
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def deprocess(img):
    # Perform the inverse of the preprocessing step
    img = img.copy()  # Create a copy to avoid modifying the original
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # Convert BGR to RGB
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Gram matrix
def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Content loss
def content_loss(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    loss = tf.reduce_mean(tf.square(a_C - a_G))
    return loss

# Style loss
def style_cost(style, generated):
    J_style = 0
    for style_model in style_models:
        a_S = style_model(style)
        a_G = style_model(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        current_cost = tf.reduce_mean(tf.square(GS - GG))
        J_style += current_cost * weight_of_layer
    return J_style

# Total Loss Function
def compute_total_loss(content, style, generated, alpha=10, beta=1000):
    J_content = content_loss(content, generated)
    J_style = style_cost(style, generated)
    return alpha * J_content + beta * J_style


def ensure_pil_image(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype('uint8'))
    return img


def neural_style_transfer(content_img, style_img, iterations=50, alpha=10, beta=1000):
    try:
        # Ensure we have PIL images
        content_img_pil = ensure_pil_image(content_img)
        style_img_pil = ensure_pil_image(style_img)

        # Resize images to a manageable size
        content_img_pil = content_img_pil.resize((300, 300), Image.LANCZOS)
        style_img_pil = style_img_pil.resize((300, 300), Image.LANCZOS)

        # Process images
        content = process_image(content_img_pil)
        style = process_image(style_img_pil)

        # Initialize with content image
        generated = tf.Variable(content, dtype=tf.float32)

        # Optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=0.7)

        progress_images = []

        for i in range(iterations):
            with tf.GradientTape() as tape:
                total_loss = compute_total_loss(
                    content, style, generated, alpha, beta)

            # Get gradients and apply
            grads = tape.gradient(total_loss, generated)
            opt.apply_gradients([(grads, generated)])

            if i % 10 == 0 or i == iterations - 1:
                # Save progress image
                current_img = generated.numpy()
                img_squeezed = np.squeeze(current_img, axis=0)
                img_deprocessed = deprocess(img_squeezed)
                progress_images.append(Image.fromarray(img_deprocessed))

            print(f"Iteration {i}, Loss: {total_loss.numpy()}")

        # Get final image
        final_img = generated.numpy()
        final_img = np.squeeze(final_img, axis=0)
        final_img = deprocess(final_img)

        return Image.fromarray(final_img), progress_images
    except Exception as e:
        print(f"Error in neural_style_transfer: {e}")
        # Return a default error image
        error_img = Image.new('RGB', (300, 300), color='red')
        return error_img, []


def style_transfer_interface(content_img, style_img, iterations=50, content_weight=10, style_weight=1000):
    # Check if images are provided
    if content_img is None or style_img is None:
        return None

    # Perform style transfer
    result_img, _ = neural_style_transfer(
        content_img,
        style_img,
        iterations=iterations,
        alpha=content_weight,
        beta=style_weight
    )

    return result_img


# Example images
content_path = "content/images/content"
style_path = "content/styles/style"

example_content_1 = f"{content_path}1.jpg"
example_content_2 = f"{content_path}2.jpg"
example_content_3 = f"{content_path}3.jpg"
example_style_1 = f"{style_path}1.jpg"
example_style_2 = f"{style_path}2.jpg"
example_style_3 = f"{style_path}3.jpg"

examples = [
    [example_content_1, example_style_1, 10, 5, 1000],
    [example_content_2, example_style_2, 20, 10, 1500],
    [example_content_3, example_style_3, 50, 15, 2000],
]

with gr.Blocks(title="Neural Style Transfer") as app:
    gr.Markdown("# Neural Style Transfer App")
    gr.Markdown(
        "Upload a content image and a style image to generate a stylized result")

    with gr.Row():
        with gr.Column():
            content_input = gr.Image(label="Content Image", type="pil")
            style_input = gr.Image(label="Style Image", type="pil")

            with gr.Row():
                iterations_slider = gr.Slider(
                    minimum=10, maximum=100, value=50, step=10,
                    label="Iterations"
                )

            with gr.Row():
                content_weight_slider = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1,
                    label="Content Weight"
                )
                style_weight_slider = gr.Slider(
                    minimum=500, maximum=2000, value=1000, step=100,
                    label="Style Weight"
                )

            submit_btn = gr.Button("Generate Stylized Image")

        with gr.Column():
            output_image = gr.Image(label="Stylized Result")

    gr.Examples(
        examples=examples,
        inputs=[content_input, style_input, iterations_slider,
                content_weight_slider, style_weight_slider],
        outputs=output_image,
        fn=style_transfer_interface,
        cache_examples=False,
    )

    submit_btn.click(
        fn=style_transfer_interface,
        inputs=[content_input, style_input, iterations_slider,
                content_weight_slider, style_weight_slider],
        outputs=output_image
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True, debug=True)