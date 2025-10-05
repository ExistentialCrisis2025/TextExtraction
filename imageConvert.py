import warnings
warnings.filterwarnings("ignore")

import easyocr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json, re

#Loading the necessary models
ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

#Check if GPU can be used, otherwise use CPU itself and load the models
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

#Describe the scene and return the caption
def describe_scene(image_path, raw_text):
    img = Image.open(image_path).convert("RGB")
    inputs = blip_processor(img, return_tensors="pt").to(device)
    caption_ids = blip_model.generate(**inputs, max_new_tokens=50)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    if raw_text:
        caption += f". The image also shows the text: '{raw_text}'"
    return caption

#Describe logbook and return the text associated with each row
def extract_logbook_structured(image_path):
    results = ocr_reader.readtext(image_path)  # [(bbox, text, conf), ...]

    
    rows = {}
    #Align rows based on average Y coordinate
    for (bbox, text, conf) in results:
        if conf < 0.4:
            continue
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        row_key = round(y_center / 20)
        rows.setdefault(row_key, []).append((bbox, text))

    structured = []
    #Store the data associated with each row in order based on top-left coordinates
    for row_key in sorted(rows.keys()):
        row_items = sorted(rows[row_key], key=lambda x: x[0][0][0])  # left-to-right
        fields = [text for _, text in row_items]
        if len(fields) >= 2:
            structured.append((row_items, fields))

    #Detect the header rows and extract it. Seperating it from the actual data to be passed on
    headers = None
    data_rows = []
    for (row_items, fields) in structured:
        joined = " ".join(fields).lower()
        if any(h in joined for h in ["date", "name", "reason", "time", "sign"]):
            # Save header positions (X midpoint of each word)
            headers = [( (bbox[0][0] + bbox[2][0]) / 2, text.lower()) for bbox, text in row_items]
        else:
            data_rows.append((row_items, fields))

    #Assign each word to the nearest header present
    parsed_rows = []
    if headers:
        header_positions = [pos for pos, h in headers]
        header_names = [h for _, h in headers]

        for (row_items, fields) in data_rows:
            row_dict = {h: "" for h in header_names}
            for bbox, text in row_items:
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                
                nearest_idx = min(range(len(header_positions)), key=lambda i: abs(header_positions[i]-x_center))
                row_dict[header_names[nearest_idx]] += (" " + text).strip()
            parsed_rows.append(row_dict)

    return parsed_rows

#Extracting the text from a normal document
def parse_document_text(image_path):
    results = ocr_reader.readtext(image_path, detail=0)
    raw_text = " ".join(results).strip()
    return {"raw_text": raw_text}

#Clean IAM text extraction by replacing common mistakes
def clean_iam_text(raw_text: str) -> str:
    fixed = raw_text
    fixed = fixed.replace("53:", "s3:").replace("S3:", "s3:").replace(" 53", " s3")
    fixed = fixed.replace("Effect ;", "Effect:").replace("Action ;", "Action:")
    fixed = fixed.replace("Resource ;", "Resource:")

    #Restoring the * as its missed constantly by OCR
    if 'Resource' in fixed and '*' not in fixed:
        fixed = re.sub(r'(Resource"\s*:\s*")([^"]*)(")', r'\1*\3', fixed)

    return fixed

#Extract the IAM policy in either a json or a yaml format
def parse_iam_policy(raw_text):
    cleaned = clean_iam_text(raw_text)

    try:
        policy = json.loads(cleaned)
        return {"format": "json", "policy": policy}
    except Exception:
        if "Effect" in cleaned and "Action" in cleaned:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            return {"format": "yaml-like", "policy": lines}
    return {"raw_text": raw_text, "cleaned_text": cleaned}

#Classify the input image into a document type
def classify_image_type(raw_text):
    if re.search(r'\"Version\"|\"Statement\"|Effect:|Action:|Resource:', raw_text):
        return "policy"
    doc_keywords = ["log book", "invoice", "certificate", "report", "date", "reason"]
    if any(kw in raw_text.lower() for kw in doc_keywords) or len(raw_text.split()) > 20:
        return "document"
    return "scene"

#Main function which gives the meaningful description based on the type of image
def analyze_image(image_path):
    ocr_result = ocr_reader.readtext(image_path, detail=0)
    raw_text = " ".join(ocr_result).strip()

    img_type = classify_image_type(raw_text)

    if img_type == "scene":
        return {"type": "scene", "output": describe_scene(image_path, raw_text)}

    elif img_type == "policy":
        return {"type": "policy", "output": parse_iam_policy(raw_text)}

    else:
        if "log book" in raw_text.lower():
            return {"type": "document", "output": extract_logbook_structured(image_path)}
        else:
            return {"type": "document", "output": parse_document_text(image_path)}


if __name__ == "__main__":
    test_images = [
        "Files/File_001.png",  
        "Files/File_002.png",  
        "Files/File_004.png",  
        "Files/File_005.png" 
    ]

    for img in test_images:
        print(f"\n--- {img} ---")
        result = analyze_image(img)
        print(result)
