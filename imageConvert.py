#BlipForConditional has the pretrained model while the other is for preparing the image and text
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract #OCR extraction

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") #Processor for converting image and text to tensors for the model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base") #loading the actual model
def describe_image_and_context(image_path):
    img = Image.open(image_path).convert("RGB")

    # OCR step
    img_for_ocr = img.convert("L")
    ocr_text = pytesseract.image_to_string(img_for_ocr, lang="eng").strip()

    # Scene caption
    inputs_scene = processor(images=img, return_tensors="pt")
    out_scene = model.generate(**inputs_scene, max_length=50, num_beams=5)
    scene_caption = processor.decode(out_scene[0], skip_special_tokens=True)

    # Natural integration
    if ocr_text and len(ocr_text) > 3:
        keywords = ["sign", "board", "bus", "poster", "label", "paper", "building"]
        if any(word in scene_caption.lower() for word in keywords):
            final_caption = scene_caption + f" labeled '{ocr_text}'"
        else:
            final_caption = scene_caption + f", with the text '{ocr_text}' visible"
    else:
        final_caption = scene_caption

    return final_caption


print(describe_image_and_context("Files/File_005.png"))