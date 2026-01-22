"""
Helper script to inspect and load the personal PyTorch model
Run this to check if your model loads correctly and see its architecture
"""
import torch
import os
import sys

PERSONAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_model', 'personal_face_recognition_model.pth')

def inspect_model():
    """Inspect the PyTorch model structure"""
    if not os.path.exists(PERSONAL_MODEL_PATH):
        print(f"[ERROR] Model not found at: {PERSONAL_MODEL_PATH}")
        return
    
    print(f"[OK] Model found at: {PERSONAL_MODEL_PATH}")
    print(f"   File size: {os.path.getsize(PERSONAL_MODEL_PATH) / (1024*1024):.2f} MB")
    
    try:
        device = torch.device('cpu')
        checkpoint = torch.load(PERSONAL_MODEL_PATH, map_location=device)
        
        print("\nModel Structure:")
        print("=" * 50)
        
        if isinstance(checkpoint, torch.nn.Module):
            print("Type: Full PyTorch Model (nn.Module)")
            print(f"Model: {checkpoint}")
        elif isinstance(checkpoint, dict):
            print("Type: Dictionary/Checkpoint")
            print(f"Keys: {list(checkpoint.keys())}")
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\nState dict keys (first 10):")
                for i, key in enumerate(list(state_dict.keys())[:10]):
                    print(f"  {key}: {state_dict[key].shape}")
                if len(state_dict) > 10:
                    print(f"  ... and {len(state_dict) - 10} more")
            
            # Check for metadata
            for key in ['img_size', 'num_classes', 'model_type', 'architecture']:
                if key in checkpoint:
                    print(f"\n{key}: {checkpoint[key]}")
        else:
            print(f"Type: {type(checkpoint)}")
            print(f"Content: {checkpoint}")
        
        print("\n" + "=" * 50)
        print("[OK] Model loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model()

