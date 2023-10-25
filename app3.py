import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from datetime import datetime

@st.cache_resource
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    return pipe

# Load the model
pipe = load_model()

# Streamlit app
st.title('Image Generation App')

# User inputs
prompt = st.text_input('Enter your prompt:')
neg_prompt = st.text_input('Enter your negative prompt:')
num_images = st.number_input('Enter the number of images to generate:', min_value=1, max_value=10, value=1, step=1)

if st.button('Generate Images'):
    for i in range(num_images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Limit the length of the prompt and replace spaces with underscores
        short_prompt = prompt[:10].replace(" ", "_")
        # Replace any other invalid characters
        valid_filename = "".join(c for c in short_prompt if c.isalnum() or c in "_-")
        filename = f'output_{valid_filename}_{timestamp}_{i}.png'
        image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]
        image.save(filename)
        st.image(filename)