# C:\Dw\GAN\GAN_iris\gan_train.py
import os, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import glob

REAL = r"C:\Dw\GAN\GAN_iris\data\real"
FAKE = r"C:\Dw\GAN\GAN_iris\data\fake"
SPLIT = r"C:\Dw\GAN\GAN_iris\data\splits"

os.makedirs(FAKE, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG = 64
Z = 100
BATCH = 64
EPOCHS = 200

class IrisDataset(Dataset):
    def __init__(self, split="train"):
        with open(os.path.join(SPLIT, f"{split}.txt")) as f:
            self.paths = [l.strip() for l in f if l.strip()]
        self.tfm = transforms.Compose([
            transforms.Resize((IMG, IMG)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img)

dl = DataLoader(IrisDataset("train"), batch_size=BATCH, shuffle=True, num_workers=0)


def block(in_c,out_c,t=True):
    if t:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c,out_c,4,2,1,bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,4,2,1,bias=False),
            nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, True)
        )

class G(nn.Module):
    def __init__(self):
        super().__init__()
        ngf=64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(Z, ngf*8, 4,1,0,bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            block(ngf*8, ngf*4, t=True),
            block(ngf*4, ngf*2, t=True),
            block(ngf*2, ngf,   t=True),
            nn.ConvTranspose2d(ngf, 3, 4,2,1,bias=False), nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        ndf=64
        self.net = nn.Sequential(
            nn.Conv2d(3, ndf, 4,2,1,bias=False), nn.LeakyReLU(0.2, True),
            block(ndf, ndf*2, t=False),
            block(ndf*2, ndf*4, t=False),
            block(ndf*4, ndf*8, t=False),
            nn.Conv2d(ndf*8, 1, 4,1,0,bias=False), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x).view(-1,1).squeeze(1)

Gnet, Dnet = G().to(device), D().to(device)
optG = torch.optim.Adam(Gnet.parameters(), lr=2e-4, betas=(0.5,0.999))
optD = torch.optim.Adam(Dnet.parameters(), lr=2e-4, betas=(0.5,0.999))
bce = nn.BCELoss()
fixed = torch.randn(64, Z,1,1, device=device)

for epoch in range(1, EPOCHS+1):
    for real in dl:
        real = real.to(device)
        bs = real.size(0)

        # D
        z = torch.randn(bs, Z,1,1, device=device)
        fake = Gnet(z).detach()
        lossD = bce(Dnet(real), torch.ones(bs, device=device)) + \
                bce(Dnet(fake), torch.zeros(bs, device=device))
        optD.zero_grad(); lossD.backward(); optD.step()

        # G
        z = torch.randn(bs, Z,1,1, device=device)
        fake = Gnet(z)
        lossG = bce(Dnet(fake), torch.ones(bs, device=device))
        optG.zero_grad(); lossG.backward(); optG.step()

    with torch.no_grad():
        samples = Gnet(fixed).cpu()
        utils.save_image((samples+1)/2, os.path.join(FAKE, f"epoch_{epoch:03d}.png"), nrow=8)
    print(f"[{epoch}/{EPOCHS}] D={lossD.item():.3f} G={lossG.item():.3f}")

torch.save(Gnet.state_dict(), "G_dcgan.pth")
print("done.")
