# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import cv2
import sys
import argparse
from prettytable import PrettyTable
from pathlib import Path
import warnings 

import torch
import torch.nn as nn
import sys
import os

# Ensure the current directory is in the path so we can find 'backbones'
sys.path.append(os.getcwd())

# Import the model - assuming backbones/iresnet.py contains iresnet100
try:
    from backbones.iresnet import iresnet100
except ImportError:
    # Adjust this import if your file structure is different (e.g., from backbones import iresnet100)
    from backbones import iresnet100

warnings.filterwarnings("ignore")

# ==========================================
# 1. PYTORCH MODEL WRAPPER
# ==========================================
class PyTorchEmbedding:
    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id >= 0 else "cpu")
        
        # 1. Initialize the ResNet100 architecture
        # num_features=512 is standard for InsightFace; change if yours is different
        self.model = iresnet100(num_features=512) 
        
        print(f"Loading state_dict from: {model_path}")
        # 2. Load the state_dict (the OrderedDict)
        state_dict = torch.load(model_path, map_location='cpu')
        
        # 3. Clean the state_dict
        # Handle cases where the weights are wrapped in a 'state_dict' key
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix (added by DistributedDataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 4. Load the weights into the architecture
        msg = self.model.load_state_dict(new_state_dict, strict=True)
        print(f"Model loaded: {msg}")
        
        self.model.to(self.device)
        self.model.eval()

    def get_alignment_mat(self, lmk):
        ref_pts = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        tfm, _ = cv2.estimateAffinePartial2D(lmk, ref_pts)
        return tfm

    def preprocess(self, img, lmk):
        # Alignment
        M = self.get_alignment_mat(lmk)
        face_img = cv2.warpAffine(img, M, (112, 112))
        
        # Standard InsightFace Preprocessing
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1)) # HWC to CHW
        face_img = (face_img - 127.5) / 128.0
        return torch.from_numpy(face_img).float()

    @torch.no_grad()
    def get(self, img, lmk):
        # Handle Flip Test
        input_t = self.preprocess(img, lmk).unsqueeze(0).to(self.device)
        
        if use_flip_test:
            # Create flipped version
            input_flip = torch.flip(input_t, dims=[3])
            inputs = torch.cat([input_t, input_flip], dim=0)
            
            # Inference
            feats = self.model(inputs)
            # If the model outputs a list or tuple (like feats and norms), take the first item
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
                
            feats = feats.cpu().numpy()
            # Return concatenated original + flipped features
            # The script later slices this into [0:dim/2] and [dim/2:] to add them
            return np.concatenate([feats[0], feats[1]], axis=0)
        else:
            feat = self.model(input_t)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            return feat.cpu().numpy().flatten()
    
# ==========================================
# 2. HELPER FUNCTIONS (Preserved)
# ==========================================

def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:,1].astype(int)
    medias = ijb_meta[:,2].astype(int)
    return templates, medias

def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:,0].astype(int)
    t2 = pairs[:,1].astype(int)
    label = pairs[:,2].astype(int)
    return t1, t2, label

def get_image_feature(img_path, img_list_path, model_path, gpu_id):
    img_list = open(img_list_path)
    embedding = PyTorchEmbedding(model_path, gpu_id)
    files = img_list.readlines()
    print('Total images:', len(files))
    faceness_scores = []
    img_feats = []
    
    for img_index, each_line in enumerate(files):
        if img_index % 1000 == 0:
            print('Processing image', img_index)
        
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        
        if img is None:
            print(f"Failed to read {img_name}")
            continue
            
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32).reshape((5, 2))
        
        feat = embedding.get(img, lmk)
        img_feats.append(feat)
        faceness_scores.append(name_lmk_score[-1])
        
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)
    return img_feats, faceness_scores

def image2template_feature(img_feats, templates, medias):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:
                media_norm_feats += [np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)]
        
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0: 
            print('Finish Calculating {} template features.'.format(count_template))
            
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates

def verification(template_norm_feats, unique_templates, p1, p2):
    template2id = np.zeros((max(unique_templates)+1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    
    score = np.zeros((len(p1),))
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]].flatten()]
        feat2 = template_norm_feats[template2id[p2[s]].flatten()]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, len(sublists)))
    return score

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do ijb test with pytorch')
    parser.add_argument('--model-path', default='model.pt', help='path to pytorch model file')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--job', default='insightface_pytorch', type=str, help='job name')
    parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
    args = parser.parse_args()

    target = args.target
    model_path = args.model_path
    gpu_id = args.gpu
    use_norm_score = True 
    use_detector_score = True
    use_flip_test = True 
    job = args.job

    # Load Meta Data
    print(f"Loading {target} meta data...")
    templates, medias = read_template_media_list(os.path.join(f'{target}/meta', f'{target.lower()}_face_tid_mid.txt'))
    p1, p2, label = read_template_pair_list(os.path.join(f'{target}/meta', f'{target.lower()}_template_pair_label.txt'))

    # Step 2: Get Image Features
    start = timeit.default_timer()
    img_path = f'./{target}/loose_crop'
    img_list_path = f'./{target}/meta/{target.lower()}_name_5pts_score.txt'
    img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model_path, gpu_id)
    stop = timeit.default_timer()
    print('Feature Extraction Time: %.2f s. ' % (stop - start))

    # Step 3: Template Aggregation
    if use_flip_test:
        # If flip test was used, features are concatenated (e.g., 512 + 512 = 1024)
        # We sum them to get a 512-dim feature (standard practice)
        img_input_feats = img_feats[:, 0:img_feats.shape[1]//2] + img_feats[:, img_feats.shape[1]//2:]
    else:
        img_input_feats = img_feats

    if not use_norm_score:
        img_input_feats = sklearn.preprocessing.normalize(img_input_feats)
        
    if use_detector_score:
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]

    template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)

    # Step 4: Verification
    score = verification(template_norm_feats, unique_templates, p1, p2)

    # Step 5: Save and Plot
    save_path = f'./{target}_result'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    score_save_file = os.path.join(save_path, f"{job}.npy")
    np.save(score_save_file, score)

    # Plot ROC (Standard Matplotlib code from original)
    x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
    
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure()
    plt.plot(fpr, tpr, lw=1, label=('[%s (AUC = %0.4f %%)]' % (job, roc_auc*100)))
    
    tpr_fpr_row = [f"{job}-{target}"]
    for fpr_val in x_labels:
        idx = np.argmin(np.abs(fpr - fpr_val))
        tpr_fpr_row.append('%.2f' % (tpr[idx] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
    
    plt.xlim([10**-6, 0.1])
    plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    fig.savefig(os.path.join(save_path, f'{job}.pdf'))
    print(tpr_fpr_table)