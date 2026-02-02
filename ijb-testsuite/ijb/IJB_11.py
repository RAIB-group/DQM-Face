
import os
import numpy as np

# ==========================================
# 1. CRITICAL NUMPY FIX FOR MXNET
# This MUST run before 'import mxnet'
# ==========================================
try:
    np.bool = np.bool_
    np.int = np.int_
    np.float = np.float64
    np.complex = np.complex128
    np.object = np.object_
    np.str = np.str_
except Exception as e:
    pass

from mxnet import np as mx_np
from mxnet import npx
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import cv2
import sys
import glob
from prettytable import PrettyTable
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")  

# --- PYTORCH IMPORTS ---
import torch
# Ensure backbones.py is in this folder or path!
try:
    from .backbones import get_model 
except ImportError:
    # Fallback if backbones.py is not found, define minimal ResNet100 here?
    # For now, we assume user copied backbones.py.
    print("Error: backbones.py not found. Please copy it to ijb-testsuite/ijb/")
    sys.exit(1)

parser = argparse.ArgumentParser(description='do ijb test')
parser.add_argument('--model-prefix', default='', help='path to load model.pt')
parser.add_argument('--model-epoch', default=1, type=int, help='')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--batch-size', default=32, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--network', default='r100', type=str, help='backbone network')
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
gpu_id = args.gpu
epoch = args.model_epoch
use_norm_score = True 
use_detector_score = True 
use_flip_test = True 
job = args.job

# ==========================================
# 2. NEW PYTORCH EMBEDDING CLASS
# ==========================================
class TorchEmbedding:
    def __init__(self, model_path, network='r100', gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Loading PyTorch Model from {model_path} on {self.device}...")
        
        # Initialize Backbone
        # Assuming r100 for your project
        self.model = get_model(network, dropout=0.0, fp16=False, num_features=512).to(self.device)
        
        # Load Weights
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle DDP 'module.' prefix if necessary
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            sys.exit(1)
            
        self.model.eval()

    def get(self, img, lmk=None):
        if img is None:
            return np.zeros(512)
            
        # Convert to Tensor
        # Resize to 112x112 if not already
        if img.shape[0] != 112 or img.shape[1] != 112:
             img = cv2.resize(img, (112, 112))
             
        img = np.transpose(img, (2, 0, 1)) # C, H, W
        img = torch.from_numpy(img).unsqueeze(0).float() # 1, C, H, W
        img.sub_(127.5).div_(128.0)
        img = img.to(self.device)
        
        with torch.no_grad():
            feat = self.model(img)
            feat = feat.cpu().numpy().flatten()
            
        return feat

def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:,1].astype(np.int)
    medias = ijb_meta[:,2].astype(np.int)
    return templates, medias

def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:,0].astype(np.int)
    t2 = pairs[:,1].astype(np.int)
    label = pairs[:,2].astype(np.int)
    return t1, t2, label

def get_image_feature(img_path, img_list_path, model_path, epoch, gpu_id):
    img_list = open(img_list_path)
    
    # --- USE NEW CLASS ---
    embedding = TorchEmbedding(model_path, network=args.network, gpu_id=gpu_id)
    # ---------------------
    
    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_feats = []
    
    for img_index, each_line in enumerate(files):
        if img_index % 1000 == 0:
            print('processing', img_index)
        
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        
        img = cv2.imread(img_name)
        
        # Get Feature using PyTorch model
        feat = embedding.get(img)
        img_feats.append(feat)
        
        faceness_scores.append(name_lmk_score[-1])
        
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)

    return img_feats, faceness_scores

def image2template_feature(img_feats = None, templates = None, medias = None):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u,ct in zip(unique_medias, unique_media_counts):
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

def verification(template_norm_feats = None, unique_templates = None, p1 = None, p2 = None):
    template2id = np.zeros((max(unique_templates)+1,1),dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    
    score = np.zeros((len(p1),)) 
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000 
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

# =============================================================
# MAIN LOGIC
# =============================================================

assert target=='IJBC' or target=='IJBB'

start = timeit.default_timer()
templates, medias = read_template_media_list(os.path.join('%s/meta'%target, '%s_face_tid_mid.txt'%target.lower()))
stop = timeit.default_timer()
print('Time Meta 1: %.2f s. ' % (stop - start))

start = timeit.default_timer()
p1, p2, label = read_template_pair_list(os.path.join('%s/meta'%target, '%s_template_pair_label.txt'%target.lower()))
stop = timeit.default_timer()
print('Time Meta 2: %.2f s. ' % (stop - start))

# --- Extract Features ---
start = timeit.default_timer()
img_path = './%s/loose_crop' % target
img_list_path = './%s/meta/%s_name_5pts_score.txt' % (target, target.lower())

# Call our new function
img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model_path, epoch, gpu_id)

stop = timeit.default_timer()
print('Time Features: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))

# --- Process Templates ---
start = timeit.default_timer()

if use_flip_test:
    img_input_feats = img_feats
else:
    img_input_feats = img_feats

if not use_norm_score:
    img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))    
    
if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[:,np.newaxis]

template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time Templates: %.2f s. ' % (stop - start))

# --- Verification ---
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time Verification: %.2f s. ' % (stop - start))

save_path = './%s_result' % target
if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy"%job)
np.save(score_save_file, score)

# --- Plotting ---
files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file)) 

methods = np.array(methods)
scores = dict(zip(methods,scores))
x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) 
    plt.plot(fpr, tpr, lw=1, label=('[%s (AUC = %0.4f %%)]' % (method.split('-')[-1], roc_auc*100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s"%(method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index]*100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels) 
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True)) 
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf'%job))
print(tpr_fpr_table)
