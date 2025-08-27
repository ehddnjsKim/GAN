# C:\Dw\GAN\GAN_iris\clf_train.py
import os, glob, random, torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, roc_auc_score

REAL = r"C:\Dw\GAN\GAN_iris\data\real"
FAKE = r"C:\Dw\GAN\GAN_iris\data\fake"
SPLIT = r"C:\Dw\GAN\GAN_iris\data\splits"

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 224; BATCH=32; EPOCHS=8

def read_list(p):
    with open(p, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

real_train = read_list(os.path.join(SPLIT, "train.txt"))
real_val   = read_list(os.path.join(SPLIT, "val.txt"))
real_test  = read_list(os.path.join(SPLIT, "test.txt"))

fake_all = sorted(glob.glob(os.path.join(FAKE, "*.png"))) + sorted(glob.glob(os.path.join(FAKE, "*.jpg")))
random.shuffle(fake_all)
n=len(fake_all)
fake_train = fake_all[:int(n*0.7)]
fake_val   = fake_all[int(n*0.7):int(n*0.85)]
fake_test  = fake_all[int(n*0.85):]

tfm_tr = transforms.Compose([
    transforms.Resize((IMG,IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1,0.1,0.1,0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tfm_te = transforms.Compose([
    transforms.Resize((IMG,IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

class RFDS(Dataset):
    def __init__(self, rp, fp, tfm):
        self.items = [(p,0) for p in rp] + [(p,1) for p in fp]
        random.shuffle(self.items)
        self.tfm = tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), torch.tensor(y, dtype=torch.long)

tr_dl = DataLoader(RFDS(real_train, fake_train, tfm_tr), batch_size=BATCH, shuffle=True)
va_dl = DataLoader(RFDS(real_val,   fake_val,   tfm_te), batch_size=BATCH)
te_dl = DataLoader(RFDS(real_test,  fake_test,  tfm_te), batch_size=BATCH)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = nn.CrossEntropyLoss()

best=0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    for x,y in tr_dl:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward(); opt.step()

    # val
    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for x,y in va_dl:
            x = x.to(device)
            logits = model(x).cpu()
            preds += logits.softmax(1)[:,1].tolist()
            gts   += y.tolist()
    bin_pred = [1 if p>=0.5 else 0 for p in preds]
    f1 = f1_score(gts, bin_pred)
    auc = roc_auc_score(gts, preds)
    print(f"[{epoch}] val F1={f1:.3f} AUC={auc:.3f}")
    score = (f1+auc)/2
    if score > best:
        best=score
        torch.save(model.state_dict(), "rf_best.pt")

# test
model.load_state_dict(torch.load("rf_best.pt", map_location=device))
model.eval(); preds=[]; gts=[]
with torch.no_grad():
    for x,y in te_dl:
        x = x.to(device)
        logits = model(x).cpu()
        preds += logits.softmax(1)[:,1].tolist()
        gts   += y.tolist()
bin_pred = [1 if p>=0.5 else 0 for p in preds]
f1 = f1_score(gts, bin_pred)
auc = roc_auc_score(gts, preds)
acc = (sum(int(a==b) for a,b in zip(bin_pred,gts)))/len(gts)
print(f"TEST  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
