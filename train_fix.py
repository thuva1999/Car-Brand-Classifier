import os, random, torch, torch.nn as nn
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

ROOT = "data/brand_dataset"
OUT = "models/car_brand_classifier_fixed.pth"
MIN_IMAGES, MAX_PER_CLASS, SIZE = 30, 250, 224
EPOCHS_HEAD, EPOCHS_FULL = 5, 20
random.seed(0); torch.manual_seed(0)

# 1) collect brands that have enough images, cap huge ones
classes, items = [], []
for b in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, b)
    if not os.path.isdir(d): continue
    files = [os.path.join(d, f) for f in os.listdir(d)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if len(files) < MIN_IMAGES:
        print(f"skip {b}: {len(files)} images"); continue
    random.shuffle(files)
    classes.append(b)
    items += [(f, len(classes) - 1) for f in files[:MAX_PER_CLASS]]
print(len(classes), "classes:", classes)

# 2) split 85/15 per class
train, val = [], []
for c in range(len(classes)):
    cs = [it for it in items if it[1] == c]
    k = max(1, int(len(cs) * 0.15))
    val += cs[:k]; train += cs[k:]

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.3, 0.05),
    transforms.RandomRotation(8),
    transforms.ToTensor(), norm])
tf_val = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), norm])

class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y

cnt = Counter(y for _, y in train)
weights = [1.0 / cnt[y] for _, y in train]
sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
dl_train = DataLoader(DS(train, tf_train), batch_size=32, sampler=sampler, num_workers=0)
dl_val = DataLoader(DS(val, tf_val), batch_size=64, num_workers=0)

# 3) model
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

def evaluate():
    model.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in dl_val:
            ok += (model(x).argmax(1) == y).sum().item(); n += len(y)
    return ok / n

def run(epochs, lr, tag):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    best = 0
    for e in range(epochs):
        model.train(); tot = 0
        for x, y in dl_train:
            opt.zero_grad(); l = loss_fn(model(x), y); l.backward(); opt.step(); tot += l.item()
        acc = evaluate()
        print(f"[{tag}] epoch {e+1}/{epochs} loss {tot/len(dl_train):.3f} val_acc {acc:.3f}")
        global best_acc
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(), "class_names": classes,
                        "architecture": "mobilenet_v3_small", "image_size": SIZE,
                        "val_accuracy": acc}, OUT)

best_acc = 0
for p in model.features.parameters(): p.requires_grad = False
run(EPOCHS_HEAD, 1e-3, "head")
for p in model.features.parameters(): p.requires_grad = True
run(EPOCHS_FULL, 1e-4, "full")
print("best val accuracy:", best_acc, "->", OUT)