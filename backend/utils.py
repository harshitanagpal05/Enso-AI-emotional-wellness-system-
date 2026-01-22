import cv2
import numpy as np
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'emotion_model.h5')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'models', 'emotion_labels.json')

# Personal PyTorch model path
PERSONAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_model', 'personal_face_recognition_model.pth')

custom_model = None
personal_pytorch_model = None
face_cascade = None
model_config = {
    'img_size': 96,
    'color_mode': 'rgb'
}
pytorch_model_config = {
    'img_size': 48,  # Common size for emotion models, will be auto-detected
    'device': 'cpu'
}

# ResNet Basic Block
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

# ResNet Bottleneck Block
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

# ResNet Model for Emotion Recognition
class ResNetEmotion(nn.Module):
    """ResNet-based architecture for emotion recognition"""
    def __init__(self, block, layers, num_classes=7):
        super(ResNetEmotion, self).__init__()
        self.inplanes = 64
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def resnet34_emotion(num_classes=7):
    """ResNet-34 for emotion recognition"""
    return ResNetEmotion(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50_emotion(num_classes=7):
    """ResNet-50 for emotion recognition"""
    return ResNetEmotion(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def load_personal_pytorch_model():
    """Load the personal PyTorch face recognition model"""
    global personal_pytorch_model, pytorch_model_config
    
    if not os.path.exists(PERSONAL_MODEL_PATH):
        print(f"Personal PyTorch model not found at {PERSONAL_MODEL_PATH}")
        return None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pytorch_model_config['device'] = device
        
        # Try to load the model
        checkpoint = torch.load(PERSONAL_MODEL_PATH, map_location=device)
        
        # Try different loading strategies
        model = None
        
        # Strategy 1: Load as complete model (if saved as full model)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.eval()
        # Strategy 2: Load from state_dict - detect ResNet architecture
        elif isinstance(checkpoint, dict):
            # Check if it's a ResNet architecture (has layer1, layer2, etc.)
            if any('layer' in k for k in checkpoint.keys()):
                # Try ResNet-34 first (most common)
                try:
                    model = resnet34_emotion(num_classes=7)
                    model.load_state_dict(checkpoint, strict=False)
                    model.eval()
                    print("   Loaded as ResNet-34")
                except:
                    # Try ResNet-50
                    try:
                        model = resnet50_emotion(num_classes=7)
                        model.load_state_dict(checkpoint, strict=False)
                        model.eval()
                        print("   Loaded as ResNet-50")
                    except:
                        # Try with state_dict wrapper
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            model = resnet34_emotion(num_classes=7)
                            model.load_state_dict(state_dict, strict=False)
                            model.eval()
                            print("   Loaded as ResNet-34 (from state_dict)")
                        else:
                            raise Exception("Could not determine ResNet variant")
            # Strategy 3: Try simple CNN (for older models) - fallback removed for now
            # If model doesn't match ResNet, it will raise an error
            else:
                raise Exception("Model architecture not recognized. Expected ResNet structure.")
        else:
            raise Exception("Unknown checkpoint format")
        
        model.to(device)
        print(f"✅ Loaded personal PyTorch model from {PERSONAL_MODEL_PATH}")
        print(f"   Device: {device}, Image size: {pytorch_model_config['img_size']}")
        return model
        
    except Exception as e:
        print(f"⚠️ Failed to load personal PyTorch model: {e}")
        print(f"   Attempting to use model architecture inference...")
        import traceback
        traceback.print_exc()
        return None

def load_custom_model():
    global custom_model, personal_pytorch_model, face_cascade, model_config
    
    # Load personal PyTorch model first (highest priority)
    personal_pytorch_model = load_personal_pytorch_model()
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                model_config['img_size'] = config.get('img_size', 96)
                print(f"Loaded model config: {config}")
        except Exception as e:
            print(f"Config load error: {e}")
    
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow import keras
            custom_model = keras.models.load_model(MODEL_PATH, compile=False)
            custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            print(f"Loaded custom Kaggle-trained model from {MODEL_PATH}")
            print(f"Model input shape: {custom_model.input_shape}")
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            custom_model = None
    else:
        print(f"Custom model not found at {MODEL_PATH}. Using fallback FER library.")
        custom_model = None
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

load_custom_model()

def preprocess_face_for_model(face_img):
    img_size = model_config.get('img_size', 96)
    
    if len(face_img.shape) == 2:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    else:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    face_resized = cv2.resize(face_rgb, (img_size, img_size))
    
    face_normalized = face_resized / 255.0
    
    face_input = np.expand_dims(face_normalized, axis=0)
    return face_input

def preprocess_face_for_pytorch(face_img):
    """Preprocess face image for PyTorch model (ResNet expects 224x224)"""
    img_size = 224  # ResNet standard input size
    device = pytorch_model_config.get('device', 'cpu')
    
    if len(face_img.shape) == 2:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
    else:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for torchvision transforms
    pil_image = Image.fromarray(face_rgb)
    
    # Define transforms (ImageNet normalization for ResNet)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    tensor_image = transform(pil_image).unsqueeze(0)
    tensor_image = tensor_image.to(device)
    
    return tensor_image

def detect_emotion_custom_model(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    faces = face_cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
    
    if len(faces) == 0:
        print("DEBUG: No face detected")
        return "neutral", 0.3
    
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    x, y, w, h = faces[0]
    padding = int(0.15 * w)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    face_roi = img[y1:y2, x1:x2]
    
    face_input = preprocess_face_for_model(face_roi)
    
    predictions = custom_model.predict(face_input, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[::-1][:3]
    print(f"DEBUG: Top 3 predictions:")
    for idx in top_indices:
        print(f"  {EMOTIONS[idx]}: {predictions[idx]:.3f}")
    
    top_idx = top_indices[0]
    top_emotion = EMOTIONS[top_idx]
    confidence = float(predictions[top_idx])
    
    return top_emotion, confidence

def detect_emotion_fer_fallback(img):
    from fer.fer import FER
    detector = FER(mtcnn=True)
    
    results = detector.detect_emotions(img)
    
    if not results:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        results = detector.detect_emotions(enhanced_img)
    
    if not results:
        print("DEBUG: FER - No face detected")
        return "neutral", 0.0
    
    emotions = results[0]["emotions"]
    print(f"DEBUG: FER Predictions: {emotions}")
    
    top_emotion = max(emotions, key=emotions.get)
    confidence = emotions[top_emotion]
    
    print(f"DEBUG: FER Top emotion: {top_emotion} ({confidence})")
    
    return top_emotion, confidence

def detect_emotion_pytorch_model(img):
    """Detect emotion using personal PyTorch model"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    faces = face_cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
    
    if len(faces) == 0:
        print("DEBUG: No face detected (PyTorch)")
        return "neutral", 0.3
    
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    x, y, w, h = faces[0]
    padding = int(0.15 * w)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    face_roi = img[y1:y2, x1:x2]
    
    # Preprocess for PyTorch
    face_input = preprocess_face_for_pytorch(face_roi)
    
    # Get predictions
    with torch.no_grad():
        outputs = personal_pytorch_model(face_input)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predictions = probabilities[0].cpu().numpy()
    
    top_indices = np.argsort(predictions)[::-1][:3]
    print(f"DEBUG: Top 3 predictions (Personal PyTorch Model):")
    for idx in top_indices:
        print(f"  {EMOTIONS[idx]}: {predictions[idx]:.3f}")
    
    top_idx = top_indices[0]
    top_emotion = EMOTIONS[top_idx]
    confidence = float(predictions[top_idx])
    
    return top_emotion, confidence

def detect_emotion_from_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, 0
    
    # Priority: Personal PyTorch model > Custom TensorFlow model > FER fallback
    if personal_pytorch_model is not None:
        print("DEBUG: Using personal PyTorch face recognition model")
        return detect_emotion_pytorch_model(img)
    elif custom_model is not None:
        print("DEBUG: Using custom Kaggle-trained model (EfficientNet)")
        return detect_emotion_custom_model(img)
    else:
        print("DEBUG: Using FER library fallback")
        return detect_emotion_fer_fallback(img)

def map_emotion_to_mood(emotion):
    mapping = {
        "happy": "positive state",
        "sad": "low",
        "angry": "frustrated and high stressed",
        "disgust": "strong dislike",
        "fear": "anxious",
        "surprise": "shocking and unexpected wonders",
        "neutral": "stable and calm state"
    }
    return mapping.get(emotion.lower(), "stable and calm state")
