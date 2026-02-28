# Kaggle-ready Python code cell: Siamese Network + XAI pipeline
# Saves outputs to /kaggle/working/output
# Assumptions: dataset root has subfolders per class: root/<class_name>/*.jpg
# Adjust DATA_ROOT to point to your dataset folder path in Kaggle input.

# Mathematical formulas used (LaTeX):
# Contrastive Loss:
# L = (1 - y) * 0.5 * D^2 + y * 0.5 * max(0, m - D)^2, where D = ||f(x1)-f(x2)||_2
#
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
#
# Expected Calibration Error (ECE) with M bins:
# ECE = sum_{m=1..M} (|B_m|/n) * |acc(B_m) - conf(B_m)|
#
# Attribution sparsity (k covering 90% mass):
# Given attribution map A (abs), find smallest k s.t. sum_{i=1..k} A_sorted[i] >= 0.9 * sum(A)
# Sparsity = k / (H * W)
#
# Statistical test (two-sample t-test) comparing same-class vs diff-class distances:
# t, p = ttest_ind(dist_same, dist_diff, equal_var=False)

import os
import random
import math
import shutil
from glob import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from scipy.stats import ttest_ind

# Config
DATA_ROOT = "/kaggle/input/your_dataset_root"  # change this to your dataset input path
OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42
BATCH_SIZE = 32
EMBEDDING_SIZE = 256
LR = 1e-4
EPOCHS = 15
MARGIN = 1.0  # for contrastive loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------
# Dataset utilities
# -------------------------
def gather_image_paths(root):
    classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
    paths = []
    labels = []
    for cls in classes:
        p = list(Path(root).glob(f"{cls}/*"))
        for f in p:
            paths.append(str(f))
            labels.append(cls)
    return paths, labels, classes

class ImageListDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], p

class SiamesePairDataset(Dataset):
    def __init__(self, paths, labels, label_encoder, transform=None, n_pairs_per_epoch=None):
        self.paths = np.array(paths)
        self.labels = np.array(labels)
        self.transform = transform
        self.le = label_encoder
        self.n = len(paths)
        self.indices_by_label = {}
        for lbl in np.unique(self.labels):
            self.indices_by_label[lbl] = np.where(self.labels == lbl)[0].tolist()
        self.n_pairs = n_pairs_per_epoch or (self.n * 2)

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        # create a positive or negative pair with 50% prob
        if random.random() < 0.5:
            # positive
            lbl = random.choice(list(self.indices_by_label.keys()))
            i1, i2 = random.sample(self.indices_by_label[lbl], 2)
            label = 0  # distance label: 0 for same
        else:
            # negative
            lbl1, lbl2 = random.sample(list(self.indices_by_label.keys()), 2)
            i1 = random.choice(self.indices_by_label[lbl1])
            i2 = random.choice(self.indices_by_label[lbl2])
            label = 1  # 1 for different
        p1 = self.paths[i1]
        p2 = self.paths[i2]
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, float(label), self.labels[i1], self.labels[i2]

# -------------------------
# Prepare data splits
# -------------------------
paths, labels, classes = gather_image_paths(DATA_ROOT)
if len(paths) == 0:
    raise RuntimeError(f"No images found in {DATA_ROOT}. Ensure subfolders per class exist.")

le = LabelEncoder()
y = le.fit_transform(labels)
class_names = le.classes_.tolist()
n_total = len(paths)
indices = list(range(n_total))
random.shuffle(indices)

n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_paths = [paths[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_paths = [paths[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]
test_paths = [paths[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]

print(f"Total: {n_total}, Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# Transforms (keep small to save memory)
input_size = 224
train_tf = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
eval_tf = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Create datasets
train_pair_ds = SiamesePairDataset(train_paths, train_labels, le, transform=train_tf, n_pairs_per_epoch=len(train_paths)*2)
val_pair_ds = SiamesePairDataset(val_paths, val_labels, le, transform=eval_tf, n_pairs_per_epoch=len(val_paths)*2)

train_loader = DataLoader(train_pair_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_pair_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# For embedding evaluation sets
train_embed_ds = ImageListDataset(train_paths, train_labels, transform=eval_tf)
val_embed_ds = ImageListDataset(val_paths, val_labels, transform=eval_tf)
test_embed_ds = ImageListDataset(test_paths, test_labels, transform=eval_tf)
train_embed_loader = DataLoader(train_embed_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_embed_loader = DataLoader(val_embed_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_embed_loader = DataLoader(test_embed_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# -------------------------
# Model definition
# -------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # remove fc layer
        modules = list(resnet.children())[:-2]  # keep conv layers
        self.feature_extractor = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        return e1, e2

# Contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        # label: 0 for same, 1 for different
        distances = (e1 - e2).pow(2).sum(1).sqrt()
        same_loss = (1 - label) * 0.5 * distances.pow(2)
        diff_loss = label * 0.5 * F.relu(self.margin - distances).pow(2)
        loss = same_loss + diff_loss
        return loss.mean()

# -------------------------
# Training siamese
# -------------------------
embedding_net = EmbeddingNet().to(DEVICE)
model = SiameseNet(embedding_net).to(DEVICE)
criterion = ContrastiveLoss(margin=MARGIN)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_history = {"train_loss": [], "val_loss": []}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for img1, img2, lbl, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - train"):
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        lbl = lbl.to(DEVICE)
        optimizer.zero_grad()
        e1, e2 = model(img1, img2)
        loss = criterion(e1, e2, lbl)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img1.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_history["train_loss"].append(epoch_loss)

    # validation
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for img1, img2, lbl, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - val"):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            lbl = lbl.to(DEVICE)
            e1, e2 = model(img1, img2)
            loss = criterion(e1, e2, lbl)
            val_running += loss.item() * img1.size(0)
    val_loss = val_running / len(val_loader.dataset)
    train_history["val_loss"].append(val_loss)
    print(f"Epoch {epoch+1}: Train Loss {epoch_loss:.4f}, Val Loss {val_loss:.4f}")

# Save training history plot
plt.figure(figsize=(6,4))
plt.plot(train_history["train_loss"], label="train_loss")
plt.plot(train_history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training History (Contrastive Loss)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"), dpi=150)
plt.close()

# -------------------------
# Create embeddings and train linear classifier
# -------------------------
def compute_embeddings(dataloader, model):
    model.eval()
    embs = []
    labels = []
    paths = []
    with torch.no_grad():
        for img, lbl, p in tqdm(dataloader, desc="Embedding"):
            img = img.to(DEVICE)
            e = model.embedding_net(img)
            embs.append(e.cpu().numpy())
            labels.extend(lbl)
            paths.extend(p)
    embs = np.vstack(embs)
    return embs, np.array(labels), paths

train_embs, train_lbls, train_paths_ordered = compute_embeddings(train_embed_loader, model)
val_embs, val_lbls, val_paths_ordered = compute_embeddings(val_embed_loader, model)
test_embs, test_lbls, test_paths_ordered = compute_embeddings(test_embed_loader, model)

# Encode labels to ints
y_train = le.transform(train_lbls)
y_val = le.transform(val_lbls)
y_test = le.transform(test_lbls)

# Train logistic regression on embeddings (CPU, small)
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(train_embs, y_train)

# Predict
train_probs = clf.predict_proba(train_embs)
train_pred = clf.predict(train_embs)
val_probs = clf.predict_proba(val_embs)
val_pred = clf.predict(val_embs)
test_probs = clf.predict_proba(test_embs)
test_pred = clf.predict(test_embs)

# Metrics
acc_train = accuracy_score(y_train, train_pred)
acc_val = accuracy_score(y_val, val_pred)
acc_test = accuracy_score(y_test, test_pred)
f1_train = f1_score(y_train, train_pred, average='macro')
f1_val = f1_score(y_val, val_pred, average='macro')
f1_test = f1_score(y_test, test_pred, average='macro')

print(f"Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Test Acc: {acc_test:.4f}")
print(f"Train F1: {f1_train:.4f}, Val F1: {f1_val:.4f}, Test F1: {f1_test:.4f}")

# Save training confusion matrix (use training dataset as requested)
cm = confusion_matrix(y_train, train_pred)
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Training Set)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_train.png"), dpi=150)
plt.close()

# Per-class performance
report = classification_report(y_val, val_pred, target_names=class_names, output_dict=True)
# Save per-class performance plot
per_class_f1 = [report[c]["f1-score"] for c in class_names]
plt.figure(figsize=(8,4))
plt.bar(class_names, per_class_f1)
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1-score")
plt.title("Per-class F1 (Validation)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150)
plt.close()

# Save classification reports
with open(os.path.join(OUTPUT_DIR, "classification_report_val.txt"), "w") as f:
    f.write(classification_report(y_val, val_pred, target_names=class_names))

# -------------------------
# Confidence Calibration: ECE
# -------------------------
def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if mask.any():
            prop = mask.mean()
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += np.abs(acc - conf) * prop
    return ece

ece_val = compute_ece(val_probs, y_val, n_bins=15)
print(f"Val ECE: {ece_val:.4f}")

# Reliability diagram
prob_true, prob_pred = calibration_curve(y_val == np.argmax(val_probs, axis=1), np.max(val_probs, axis=1), n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Reliability Diagram (Validation)")
plt.savefig(os.path.join(OUTPUT_DIR, "reliability_diagram.png"), dpi=150)
plt.close()

# -------------------------
# Attribution: Grad-CAM Implementation
# -------------------------
# We'll attach a small linear classifier head to embedding_net for Grad-CAM: use the trained logistic regression weights.
# Create a PyTorch linear head initialized from sklearn weights (for compatibility).
class LinearProbe(nn.Module):
    def __init__(self, embedding_net, n_classes, sklearn_clf):
        super().__init__()
        self.embedding_net = embedding_net
        self.embedding_net.eval()
        self.fc = nn.Linear(EMBEDDING_SIZE, n_classes)
        # Transfer weights from sklearn logistic regression (coef_ shape: n_classes x emb_dim)
        coef = sklearn_clf.coef_
        intercept = sklearn_clf.intercept_
        self.fc.weight.data = torch.tensor(coef, dtype=torch.float32)
        self.fc.bias.data = torch.tensor(intercept, dtype=torch.float32)

    def forward(self, x):
        with torch.no_grad():
            e = self.embedding_net(x)
        return self.fc(e)

# For Grad-CAM we need the feature maps and gradients from the last conv layer of feature_extractor
# Hook functions
gradients = {}
activations = {}

def save_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

def save_gradient(name):
    def hook(module, grad_in, grad_out):
        gradients[name] = grad_out[0].detach()
    return hook

# Register hooks on the last conv layer of the backbone
backbone = embedding_net.feature_extractor  # sequential
# Find last conv layer
last_conv = None
for name, module in reversed(list(backbone.named_modules())):
    if isinstance(module, nn.Conv2d):
        last_conv = module
        break
if last_conv is None:
    raise RuntimeError("Could not find last conv layer for Grad-CAM")

last_conv.register_forward_hook(save_activation("last_conv"))
last_conv.register_backward_hook(save_gradient("last_conv"))

probe = LinearProbe(embedding_net, n_classes=len(class_names), sklearn_clf=clf).to(DEVICE)
probe.eval()

def grad_cam(img_tensor, target_class=None):
    # img_tensor: single image tensor (1,C,H,W)
    probe.zero_grad()
    activations.clear(); gradients.clear()
    output = probe(img_tensor)
    if target_class is None:
        target_class = int(output.argmax(dim=1).item())
    score = output[0, target_class]
    score.backward(retain_graph=True)
    act = activations["last_conv"]  # shape [1, C, H, W]
    grad = gradients["last_conv"]   # shape [1, C, H, W]
    weights = grad.mean(dim=(2,3), keepdim=True)  # global avg pool over H,W -> [1,C,1,1]
    cam = (weights * act).sum(dim=1, keepdim=True)  # [1,1,H,W]
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(input_size, input_size), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam  # shape H,W normalized

# Save Grad-CAM visualizations for a few sample images from test set (one per class)
samples_per_class = 1
saved = 0
for cls in class_names:
    # find one image in test set of this class
    for p, lbl in zip(test_paths_ordered, test_lbls):
        if lbl == cls:
            img = Image.open(p).convert("RGB")
            inp = eval_tf(img).unsqueeze(0).to(DEVICE)
            cam = grad_cam(inp)
            # overlay
            img_np = np.array(img.resize((input_size, input_size))).astype(np.float32)/255.0
            heatmap = plt.cm.jet(cam)[:,:,:3]
            overlay = 0.5 * img_np + 0.5 * heatmap
            overlay = np.clip(overlay, 0, 1)
            out_path = os.path.join(OUTPUT_DIR, f"gradcam_{cls}.png")
            plt.imsave(out_path, overlay)
            saved += 1
            break
    if saved >= len(class_names):
        break

# -------------------------
# Attribution sparsity computation
# -------------------------
def attribution_sparsity(cam_map, mass_threshold=0.9):
    flat = np.abs(cam_map).flatten()
    flat_sorted = np.sort(flat)[::-1]
    cumsum = np.cumsum(flat_sorted)
    total = cumsum[-1] if cumsum[-1] > 0 else 1e-9
    k = np.searchsorted(cumsum, mass_threshold * total) + 1
    return k / flat.size

# Compute sparsity for sample of validation images
sparsities = []
for i in range(min(200, len(val_paths))):
    img = Image.open(val_paths[i]).convert("RGB")
    inp = eval_tf(img).unsqueeze(0).to(DEVICE)
    cam = grad_cam(inp)
    sp = attribution_sparsity(cam, mass_threshold=0.9)
    sparsities.append(sp)
mean_sparsity = np.mean(sparsities)
print(f"Mean attribution sparsity (fraction pixels for 90% mass): {mean_sparsity:.4f}")

# Save a histogram of sparsities
plt.figure(figsize=(6,4))
plt.hist(sparsities, bins=30)
plt.xlabel("Sparsity (fraction of pixels for 90% mass)")
plt.title("Attribution Sparsity Distribution (Validation samples)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "attribution_sparsity_hist.png"), dpi=150)
plt.close()

# -------------------------
# Statistical significance: t-test on distances
# -------------------------
# Build same-class and diff-class distances on validation set using embeddings
def pairwise_distances_between_sets(embs, labels, n_samples=5000):
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(embs))
    same_d = []
    diff_d = []
    for _ in range(n_samples):
        i = rng.integers(0, len(embs))
        j = rng.integers(0, len(embs))
        if i == j:
            continue
        d = np.linalg.norm(embs[i] - embs[j])
        if labels[i] == labels[j]:
            same_d.append(d)
        else:
            diff_d.append(d)
    return np.array(same_d), np.array(diff_d)

same_d, diff_d = pairwise_distances_between_sets(val_embs, y_val, n_samples=2000)
t_stat, p_val = ttest_ind(same_d, diff_d, equal_var=False)
print(f"T-test same vs diff distances: t={t_stat:.3f}, p={p_val:.3e}")

# -------------------------
# Save summary metrics to file
# -------------------------
summary = {
    "train_acc": acc_train, "val_acc": acc_val, "test_acc": acc_test,
    "train_f1": f1_train, "val_f1": f1_val, "test_f1": f1_test,
    "val_ece": ece_val, "mean_attribution_sparsity": mean_sparsity,
    "t_stat": float(t_stat), "p_value": float(p_val)
}
import json
with open(os.path.join(OUTPUT_DIR, "summary_metrics.json"), "w") as f:
    json.dump(summary, f, indent=2)

# -------------------------
# Learning curves: embed distance separation plot
# -------------------------
plt.figure(figsize=(6,4))
plt.hist(same_d, bins=50, alpha=0.6, label='same-class d')
plt.hist(diff_d, bins=50, alpha=0.6, label='diff-class d')
plt.legend()
plt.xlabel("Embedding Distance")
plt.title("Embedding Distance Separation (Validation)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distance_separation.png"), dpi=150)
plt.close()

# -------------------------
# Save model (lightweight)
# -------------------------
torch.save(embedding_net.state_dict(), os.path.join(OUTPUT_DIR, "embedding_net.pth"))

# -------------------------
# Optionally save some example explanations side-by-side
# -------------------------
def save_side_by_side(img_path, cam_overlay_path, out_path):
    a = Image.open(img_path).resize((input_size, input_size))
    b = Image.open(cam_overlay_path).resize((input_size, input_size))
    new = Image.new('RGB', (input_size*2, input_size))
    new.paste(a, (0,0))
    new.paste(b, (input_size,0))
    new.save(out_path)

for cls in class_names:
    cam_path = os.path.join(OUTPUT_DIR, f"gradcam_{cls}.png")
    # find a test image
    for p, lbl in zip(test_paths_ordered, test_lbls):
        if lbl == cls:
            outp = os.path.join(OUTPUT_DIR, f"explain_{cls}.png")
            save_side_by_side(p, cam_path, outp)
            break

print("All outputs saved to", OUTPUT_DIR)