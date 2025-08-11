# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq

## added
from PIL import Image
import torch
## end

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
# model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

## added
# Load local image
image_path = "/path/to/your/image.jpg"
local_image = Image.open(image_path).convert("RGB")
## end

messages = [
    {
        "role": "user",
        "content": [
            # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "image", "image": local_iamge},
            {"type": "text", "text": "What text do you see from this image?"}
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

## updated but questioned, so it's commented out
# Generate
# with torch.inference_mode():
#     outputs = model.generate(**inputs, max_new_tokens=40)

# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
##

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
