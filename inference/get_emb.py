import os
import sys
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ==========================================
# 1. SETUP PATHS
# ==========================================
# Use the local repo backbones package
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from backbones import get_model

CHECKPOINT_PATH = str(ROOT / "inference" / "model_dqmface_alpha05.pt")
IMAGE_PATH = str(ROOT / "inference" / "0.jpg")
SAVE_DIR = str(ROOT / "inference")
os.makedirs(SAVE_DIR, exist_ok=True)

# Preprocessing: Standard for Face Recognition (112x112, normalized to [-1, 1])
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==========================================
# 2. ARCHITECTURE DEFINITION
# ==========================================
class PowerMagAttention(torch.nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 16),
            torch.nn.GELU(),
            torch.nn.Linear(in_features // 16, in_features),
            torch.nn.Sigmoid()
        )
        self.refiner = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        mag_q = (torch.clamp(norm, 10.0, 110.0) - 10.0) / 100.0
        gated_x = x * self.gate(x)
        sem_q = self.refiner(gated_x)
        
        # Weighted fusion (Adaptive quality)
        combined_quality = 0.6 * mag_q + 0.4 * sem_q
        return combined_quality

# ==========================================
# 3. LOAD MODEL & WEIGHTS
# ==========================================
def load_model():
    print("--> Building r100 backbone and attention head...")
    backbone = get_model("r100", dropout=0.0, fp16=False, num_features=512)
    attn_head = PowerMagAttention()

    if torch.cuda.is_available():
        backbone = backbone.cuda()
        attn_head = attn_head.cuda()

    ckpt = torch.load(CHECKPOINT_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    def clean(sd):
        return {k.replace('module.', ''): v for k, v in sd.items()}

    print("--> Loading weights from checkpoint...")
    if "state_dict_backbone" in ckpt:
        backbone.load_state_dict(clean(ckpt["state_dict_backbone"]))
    if "state_dict_fc" in ckpt:
        fc_sd = clean(ckpt["state_dict_fc"])
        attn_sd = {k.replace('attention_net.', ''): v for k, v in fc_sd.items() if 'attention_net' in k}
        attn_head.load_state_dict(attn_sd)

    return backbone.eval(), attn_head.eval()

# ==========================================
# 4. EXTRACTION AND PRINTING
# ==========================================
def run_extraction():
    backbone, attn = load_model()

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    print(f"--> Processing image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        feat = backbone(img_tensor)
        qual = attn(feat)

    print(f"Image: {os.path.basename(IMAGE_PATH)}")
    print("Embedding:", feat.cpu().numpy().flatten())
    print("Quality:", qual.cpu().item())

if __name__ == "__main__":
    run_extraction()