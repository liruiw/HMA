import gradio as gr
import numpy as np
from PIL import Image
import cv2
from sim.simulator import GenieSimulator
import os
import spaces


if not os.path.exists("data/mar_ckpt/langtable"):
    # download from google drive
    import gdown

    gdown.download_folder("https://drive.google.com/drive/u/2/folders/1XU87cRqV-IMZA6RLiabIR_uZngynvUFN")
    os.system("mkdir -p data/mar_ckpt/; mv langtable data/mar_ckpt/")

RES = 512
PROMPT_HORIZON = 3
IMAGE_DIR = "assets/langtable_prompt/"

# Load available images
available_images = sorted([img for img in os.listdir(IMAGE_DIR) if img.endswith(".png")])


genie = GenieSimulator(
    image_encoder_type="temporalvae",
    image_encoder_ckpt="stabilityai/stable-video-diffusion-img2vid",
    quantize=False,
    backbone_type="stmar",
    backbone_ckpt="data/mar_ckpt_long2/langtable",
    prompt_horizon=PROMPT_HORIZON,
    action_stride=1,
    domain="language_table",
)


# Helper function to reset GenieSimulator with the selected image
def initialize_simulator(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_path)
    prompt_image = np.tile(np.array(image), (genie.prompt_horizon, 1, 1, 1)).astype(np.uint8)
    prompt_action = np.zeros((genie.prompt_horizon - 1, genie.action_stride, 2)).astype(np.float32)
    genie.set_initial_state((prompt_image, prompt_action))
    reset_image = genie.reset()
    reset_image = cv2.resize(reset_image, (RES, RES))
    return Image.fromarray(reset_image)


# Example model: takes a direction and returns a random image
def model(direction: str):
    if direction == "right":
        action = np.array([0, 0.05])
    elif direction == "left":
        action = np.array([0, -0.05])
    elif direction == "down":
        action = np.array([0.05, 0])
    elif direction == "up":
        action = np.array([-0.05, 0])
    else:
        raise ValueError(f"Invalid direction: {direction}")
    next_image = genie.step(action)["pred_next_frame"]
    next_image = cv2.resize(next_image, (RES, RES))
    return Image.fromarray(next_image)


# Gradio function to handle user input
@spaces.GPU
def handle_input(direction):
    print(f"User clicked: {direction}")
    new_image = model(direction)  # Get a new image from the model
    return new_image


# Gradio function to handle image selection
def handle_image_selection(image_name):
    print(f"User selected image: {image_name}")
    return initialize_simulator(image_name)


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            image_selector = gr.Dropdown(choices=available_images, value=available_images[0], label="Select an Image")
            select_button = gr.Button("Load Image")

        with gr.Row():
            image_display = gr.Image(type="pil", label="Generated Image")

        with gr.Row():
            up = gr.Button("↑ Up")
        with gr.Row():
            left = gr.Button("← Left")
            down = gr.Button("↓ Down")
            right = gr.Button("→ Right")

        # Define interactions
        select_button.click(fn=handle_image_selection, inputs=image_selector, outputs=image_display)
        up.click(fn=lambda: handle_input("up"), outputs=image_display, show_progress="hidden")
        down.click(fn=lambda: handle_input("down"), outputs=image_display, show_progress="hidden")
        left.click(fn=lambda: handle_input("left"), outputs=image_display, show_progress="hidden")
        right.click(fn=lambda: handle_input("right"), outputs=image_display, show_progress="hidden")

    demo.launch(share=True)
