import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.preprocessing import normalize
import cv2

# Import your backbone
from backbones.iresnet import iresnet100

# ==========================================
# 1. HELPERS FROM TRAINING SCRIPT
# ==========================================

def load_bin(path, image_size=(112, 112)):
    """Load validation bin file (LFW, CFP_FP, etc)"""
    with open(path, "rb") as f:
        # encoding='bytes' is required for python3 to load mxnet-generated bins
        bins, issame_list = pickle.load(f, encoding="bytes")

    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)

    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        # Use OpenCV to decode the raw bytes from the bin
        img = cv2.imdecode(np.frombuffer(_bin, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = (img / 255.0 - 0.5) / 0.5  # Standard InsightFace normalization

        for flip in [0, 1]:
            if flip == 1:
                img_f = torch.flip(img, [2])
                data_list[flip][idx] = img_f
            else:
                data_list[flip][idx] = img

    return data_list, issame_list

def evaluate_accuracy(embeddings, issame_list, nrof_folds=10):
    """Calculate accuracy using threshold sweep"""
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    # Calculate L2 distance
    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), 1)

    thresholds = np.arange(0, 4, 0.01)
    issame = np.array(issame_list)

    accuracy_list = []
    for threshold in thresholds:
        predict_issame = np.less(dist, threshold)
        accuracy = np.mean(predict_issame == issame)
        accuracy_list.append(accuracy)

    best_acc = max(accuracy_list)
    best_threshold = thresholds[np.argmax(accuracy_list)]

    return best_acc, best_threshold

@torch.no_grad()
def perform_val(backbone, data_path, val_targets, batch_size=64):
    """Main validation loop"""
    backbone.eval()
    
    for name in val_targets:
        bin_path = os.path.join(data_path, f"{name}.bin")
        if not os.path.exists(bin_path):
            print(f"Target {name} not found at {bin_path}")
            continue

        data_list, issame_list = load_bin(bin_path)
        print(f"Evaluating {name}...")

        embeddings_list = []
        for data in data_list:
            embeddings = []
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size].cuda()
                emb = backbone(batch).cpu().numpy()
                embeddings.append(emb)
            embeddings_list.append(np.concatenate(embeddings))

        # Horizontal Flip Test fusion (summing original + flipped)
        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = normalize(embeddings)

        acc, threshold = evaluate_accuracy(embeddings, issame_list)
        print(f"Result for {name}: Accuracy = {acc*100:.2f}% | Best Threshold = {threshold:.3f}")

# ==========================================
# 2. EXECUTION LOGIC
# ==========================================

if __name__ == "__main__":
    # CONFIGURATION
    MODEL_PATH = "/slurm/homes/bel/Atten dualFace Project /output/ablation_dual_attention/17_12_25/model.pt"
    DATA_PATH = "/slurm/homes/bel/Atten dualFace Project /faces_emore/faces_emore"
    TARGETS = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw","cfp_ff"]
    GPU_ID = 0
    
    # 1. Load Model
    print("Initializing ResNet100...")
    device = torch.device(f"cuda:{GPU_ID}")
    model = iresnet100(num_features=512).to(device)
    
    # 2. Load Weights
    print(f"Loading weights from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    # Handle the 'state_dict_backbone' nesting found in your training script's checkpoints
    if 'state_dict_backbone' in state_dict:
        state_dict = state_dict['state_dict_backbone']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Remove 'module.' prefix if it exists
    new_state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items() }
    
    model.load_state_dict(new_state_dict)
    print("Weights loaded successfully.")

    # 3. Run Validation
    perform_val(model, DATA_PATH, TARGETS)