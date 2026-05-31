import os
import torch
from PIL import Image
import numpy as np

class FaceMatcher:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn_single = None
        self.mtcnn_multi = None
        self.resnet = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return True
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            
            # MTCNN for single face (Selfie / Profile Picture)
            # keep_all=False ensures we only extract the primary face
            self.mtcnn_single = MTCNN(
                image_size=160, margin=14, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=self.device, keep_all=False
            )
            
            # MTCNN for multiple faces (Event Photos)
            # keep_all=True detects and crops all faces in the photo
            self.mtcnn_multi = MTCNN(
                image_size=160, margin=14, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                device=self.device, keep_all=True
            )
            
            # Load FaceNet model pre-trained on VGGFace2
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.initialized = True
            print(f"✅ FaceMatcher (MTCNN + FaceNet) initialized on {self.device}")
            return True
        except Exception as e:
            print(f"❌ Error initializing FaceMatcher: {e}")
            return False

    def get_selfie_embedding(self, selfie_path):
        """
        Loads the selfie image, detects the primary face, and extracts its 512-dim embedding.
        Returns embedding as a 1D numpy array, or None if no face is detected.
        """
        if not self.initialize():
            return None
        try:
            if not os.path.exists(selfie_path):
                print(f"⚠️ Selfie file not found: {selfie_path}")
                return None
                
            img = Image.open(selfie_path).convert('RGB')
            # Detect and crop face
            face = self.mtcnn_single(img)
            if face is not None:
                with torch.no_grad():
                    # face tensor shape: [3, 160, 160] -> unsqueeze to [1, 3, 160, 160]
                    embedding = self.resnet(face.unsqueeze(0).to(self.device))
                    # Return as 1D numpy array
                    return embedding[0].cpu().numpy()
            else:
                print(f"⚠️ No face detected in selfie: {selfie_path}")
        except Exception as e:
            print(f"❌ Error getting selfie embedding for {selfie_path}: {e}")
        return None

    def match_selfie_to_photo(self, selfie_embedding, photo_path, threshold=0.8):
        """
        Compares a pre-calculated selfie embedding against all faces detected in a target photo.
        Returns True if at least one face matches, False otherwise.
        """
        if selfie_embedding is None:
            return False
        if not self.initialize():
            return False
        try:
            if not os.path.exists(photo_path):
                return False
                
            img = Image.open(photo_path).convert('RGB')
            # Detect all faces in target photo
            faces = self.mtcnn_multi(img)
            if faces is not None:
                # faces shape: [N, 3, 160, 160]
                with torch.no_grad():
                    embeddings = self.resnet(faces.to(self.device)) # [N, 512]
                    
                    # Convert selfie embedding to tensor
                    selfie_tensor = torch.tensor(selfie_embedding, device=self.device) # [512]
                    
                    # Compute Euclidean distances between selfie embedding and all target embeddings
                    distances = torch.norm(embeddings - selfie_tensor, dim=1) # [N]
                    
                    min_distance = torch.min(distances).item()
                    return min_distance < threshold
        except Exception as e:
            print(f"❌ Error matching selfie to photo {photo_path}: {e}")
        return False
