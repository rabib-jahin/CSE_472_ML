

import gradio as gr
import numpy as np
import torch
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from scratch_detection import ScratchDetection

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DEISMultistepScheduler
from diffusers.utils import load_image
from PIL import Image
import cv2
import time
import os

device = "cuda"


controlnet = ControlNetModel.from_pretrained("thepowefuldeez/sd21-controlnet-canny", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)

# speed up diffusion process with faster scheduler and memory optimization
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.to('cuda')
def combine_masks2(mask1, mask2):
    mask1.save("mask1.png")
    mask2.save("mask2.png")
    mask1_np = np.array(mask1)
    mask2_np = np.array(mask2)
    inverted_mask2_np = np.logical_not(mask2_np)


    result_mask = np.logical_and(mask1_np, inverted_mask2_np)
    # final_result = np.logical_or(result_mask, mask1_np)


    final_result_int = result_mask.astype(np.uint8) * 255




 



    combined_mask = Image.fromarray( final_result_int)
    return combined_mask

def combine_masks(mask1, mask2):
    mask1_np = np.array(mask1)
    mask2_np = np.array(mask2)
    combined_mask_np = np.maximum(mask1_np, mask2_np)
    combined_mask = Image.fromarray(combined_mask_np)
    return combined_mask

if not os.path.exists("input_images"):
    os.makedirs("input_images")

def generate_scratch_mask(input_dict):
    # Save the input image to a directory
    input_image = input_dict["image"].convert("RGB")
    input_image_path = "input_images/input_image.png"
    input_image_resized = resize_image(input_image, 768)


    input_image_resized.save(input_image_path)

    test_path = "input_images"
    output_dir = "output_masks"
    scratch_detector = ScratchDetection(test_path, output_dir, input_size="scale_256", gpu=0)
    scratch_detector.run()
    mask_image = scratch_detector.get_mask_image("input_image.png")
    
    # Resize the mask to match the input image size
    mask_image = mask_image.resize(input_image.size, Image.BICUBIC)
    mask_image_np = np.array(mask_image)
    if len(mask_image_np.shape) == 3:
      
        mask_image_np = cv2.cvtColor(mask_image_np, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(mask_image_np, 127, 255, cv2.THRESH_BINARY)

# Stretch the contrast of the white regions using contrast stretching
    min_intensity = np.min(mask_image_np[binary_mask == 255])
    max_intensity = np.max(mask_image_np[binary_mask == 255])
    mask_image_np_stretched = (mask_image_np - min_intensity) / (max_intensity - min_intensity) * 255

# Convert back to uint8 (assuming mask_image_np_gray is in the range [0, 255])
    mask_image_np_stretched = mask_image_np_stretched.astype(np.uint8)


    # # Apply dilation to make the lines bigger
    kernel = np.ones((4, 4), np.uint8)
    # mask_image_np = np.array(mask_image)
    mask_image_np_dilated = cv2.dilate(mask_image_np, kernel, iterations=1)
    mask_image_dilated = Image.fromarray(mask_image_np_dilated)

    return  mask_image_dilated  

def apply_sharpening_and_blur(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Convert Laplacian result back to uint8
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # Convert gray to float32 for addWeighted
    gray_float32 = gray.astype(np.float32)
    
    # Sharpen image by adding the Laplacian back to the original image
    sharpened = cv2.addWeighted(gray_float32, 1.5, laplacian, -0.5, 0, dtype=cv2.CV_32F)
    sharpened_uint8 = cv2.convertScaleAbs(sharpened)
    blurred = cv2.GaussianBlur(sharpened_uint8, (5, 5), 0)
    return  blurred.astype(np.uint8)
    
def resize_image(image, target_size):
    width, height = image.size
    aspect_ratio = float(width) / float(height)
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size
    return image.resize((new_width, new_height), Image.BICUBIC)

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(source='upload', tool='sketch', elem_id="input_image_upload", type="pil", label="Upload & Draw on Image")
        mask_image = gr.Image(label="mask",tool="sketch",elem_id="input_image_upload", type="pil")
        output_image = gr.Image(label="output")
    with gr.Row():
       control_image= gr.Image(shape=[256,256],label="Control Image").style(width=256, height=256)
      
    with gr.Row():
        generate_mask_button = gr.Button("Generate Scratch Mask")
        submit = gr.Button("Inpaint")
    
    def inpaint(input_dict, mask2):
        image = input_dict["image"].convert("RGB")
        draw_mask = input_dict["mask"].convert("RGB")
        mask= combine_masks2(mask2['image'].convert("RGB"), mask2['mask'].convert("RGB"))
       
        mask.save("masks.png")
        image = resize_image(image, 768)
        
        # mask = Image.fromarray(mask)
        mask = resize_image(mask, 768)
        draw_mask = resize_image(draw_mask, 768)

        image = np.array(image)
        low_threshold = 30
        high_threshold = 150
        canny = cv2.Canny(image, low_threshold, high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
      
        canny_image = Image.fromarray(canny)
        canny_cv=cv2.cvtColor(np.array(canny_image) , cv2.COLOR_RGB2BGR)
        mask_=cv2.cvtColor(np.array(mask) , cv2.COLOR_RGB2GRAY)
        mask_ = cv2.bitwise_not(mask_)
        result = cv2.bitwise_and(canny_cv, canny_cv, mask=mask_)
        result=Image.fromarray(result)
        result=resize_image(result,768)
        result.save("canny.png")
        
        generator = torch.manual_seed(0)

        # Combine drawn mask and generated mask
        combined_mask = combine_masks(draw_mask, mask)
        combined_mask.save("com.png")

        output = pipe(
            prompt="",
            num_inference_steps=20,
            generator=generator,
            image=image,
            control_image=result,
            controlnet_conditioning_scale=0.5,
           
            mask_image=combined_mask
        ).images[0]
       

        
        return output,result

    generate_mask_button.click(generate_scratch_mask, inputs=[input_image], outputs=[mask_image])
    submit.click(inpaint, inputs=[input_image, mask_image], outputs=[output_image,control_image])
    demo.launch(share=True)


       
