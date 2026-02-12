import os
from PIL import Image

# Try to import ML libraries
ML_AVAILABLE = False
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    ML_AVAILABLE = True
    print("✅ ML Libraries loaded successfully!")
except ImportError:
    print("⚠️ ML Libraries not found. Run: pip install torch transformers")

# 1. Define Categories (What the AI looks for)
# Key = The tag stored in DB
# Value = The text description the AI uses to compare
CATEGORY_MAP = {
    "awards": ["award ceremony", "holding a trophy", "receiving a certificate", "medal", "winner on stage"],
    "certifications": ["holding a certificate", "diploma distribution", "paper document"],
    "passport": ["passport size photo", "id card photo", "formal headshot", "face closeup"],
    "dance": ["people dancing", "dance performance", "choreography", "ballet"],
    "music": ["playing guitar", "singing with microphone", "musical instrument", "band performing", "drummer"],
    "stage": ["person giving speech", "podium", "anchor holding mic", "stage presentation"],
    "group": ["group of friends", "team photo", "class photo", "crowd gathering"],
    "food": ["food buffet", "eating lunch", "snacks", "cake"],
    "candid": ["candid moment", "people laughing", "random shot"]
}

# Flatten for model input
SEARCH_TEXTS = [desc for sublist in CATEGORY_MAP.values() for desc in sublist]

# Global variables to cache model
model = None
processor = None

def load_model():
    global model, processor
    if not ML_AVAILABLE: return False
    
    if model is None:
        print("🔄 Loading AI Model (This happens only once)...")
        try:
            # Using a smaller, faster version of CLIP
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("✅ AI Model Loaded!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    return True

def analyze_image(image_path):
    """
    Analyzes image and returns tags based on visual content
    """
    if not ML_AVAILABLE or not load_model():
        return "event"

    try:
        image = Image.open(image_path)
        
        # Prepare inputs
        inputs = processor(text=SEARCH_TEXTS, images=image, return_tensors="pt", padding=True)
        
        # Run AI Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # percentages

        # Get top 3 matches
        values, indices = probs[0].topk(3)
        
        detected_tags = set()
        
        for i in range(3):
            score = values[i].item()
            index = indices[i].item()
            
            # If AI is more than 20% confident
            if score > 0.2:
                matched_desc = SEARCH_TEXTS[index]
                # Find which category this description belongs to
                for tag, descriptions in CATEGORY_MAP.items():
                    if matched_desc in descriptions:
                        detected_tags.add(tag)
        
        # Always add 'event' tag
        detected_tags.add('event')
        
        final_tags = ",".join(list(detected_tags))
        print(f"🧐 Analyzed {os.path.basename(image_path)} -> Tags: {final_tags}")
        return final_tags

    except Exception as e:
        print(f"Analysis Failed: {e}")
        return "event"