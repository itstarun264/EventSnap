FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the CLIP model so it is cached in the Docker image
RUN python -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Pre-download DeepFace VGG-Face model weights
RUN python -c "from deepface import DeepFace; import numpy as np; img = np.zeros((224, 224, 3), dtype=np.uint8); \
    try: DeepFace.verify(img, img, model_name='VGG-Face', enforce_detection=False) \
    except Exception as e: print('Weights cached or expected error:', e)"

# Copy the rest of the application files
COPY --chown=user . $HOME/app

# Expose port (HF Spaces defaults to 7860)
EXPOSE 7860
ENV PORT=7860

# Command to run the application
CMD ["python", "app.py"]
