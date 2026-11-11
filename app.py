import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os


from networks import UnetGenerator

# 1. Load the model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    # Fixed path to match your folder structure
    model_path = "facades_pix2pix/latest_net_G.pth" 
    
    # Fixed Architecture: Using UnetGenerator because we trained on facades
    netG = UnetGenerator(input_nc=3, output_nc=3, num_downs=8, ngf=64, norm_layer=torch.nn.BatchNorm2d, use_dropout=False)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

netG = load_model()

# 2. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Build the UI
st.set_page_config(page_title="VirtuStain AI", page_icon="🔬")
st.title("🔬 VirtuStain AI")
st.write("Upload a raw image to generate the AI-stained output.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Original
    original_image = Image.open(uploaded_file).convert("RGB")
    st.image(original_image, caption="Original Image", use_container_width=True)
    
    with st.spinner("Generating stain..."):
        # Preprocess and Predict
        input_tensor = transform(original_image).unsqueeze(0)
        with torch.no_grad():
            output_tensor = netG(input_tensor)
        
        # Post-process
        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = (output_tensor + 1) / 2.0 
        output_image = transforms.ToPILImage()(output_tensor)
        
        # Display Output
        st.image(output_image, caption="AI Generated Stain", use_container_width=True)