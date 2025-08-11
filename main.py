# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

## added
from PIL import Image
import torch
import os
## end

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
# model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

## added
img_dir = './img'

with os.scandir(img_dir) as entries:
    filenames = [entry.name for entry in entries if entry.is_file()]

print(filenames)
# Load local image
image_path = os.path.join(img_dir, filenames[0])
local_image = Image.open(image_path).convert("RGB")
## end

def use_model(image, text, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
                {"type": "image", "image": iamge},
                {"type": "text", "text": text}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
    	messages,
    	add_generation_prompt=True,
    	tokenize=True,
    	return_dict=True,
    	return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])

## updated but questioned, so it's commented out
# Generate
# with torch.inference_mode():
#     outputs = model.generate(**inputs, max_new_tokens=40)

# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
##

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

for i in range(1):
    image_path = os.path.join(img_dir, filenames[i])
    local_image = Image.open(image_path).convert("RGB")
    text = "What text do you see from this image?"
    use_model(local_image, text, processor, model)
    print('=================')

