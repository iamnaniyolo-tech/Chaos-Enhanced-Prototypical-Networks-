

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

# ==================== AMD GPU (DirectML) ====================
try:
    import torch_directml
    device = torch_directml.device()
    print(f"Using AMD GPU: {torch_directml.device_name(0)}")
except:
    device = torch.device("cpu")
    print("Using CPU")

# ==================== Chaos Module (Fine-tuned) ====================
class LogisticChaos(nn.Module):
    def __init__(self, intensity=0.04):
        super().__init__()
        self.intensity = intensity
        self.r = 3.99
    def forward(self, x):
        if not self.training:
            return x
        noise = torch.rand_like(x)
        for _ in range(8):
            noise = self.r * noise * (1 - noise)
        noise = (noise - 0.5) * 2 * self.intensity
        return x + noise

# ==================== ResNet-18 ProtoNet (L2 + Cosine) ====================
class MedicalProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.scale = nn.Parameter(torch.tensor(20.0))

        # Unfreeze last two blocks for fine-tuning
        for name, param in resnet.named_parameters():
            if "layer3" in name or "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        z = self.encoder(x).flatten(1)
        return F.normalize(z, dim=1)

# ==================== Medical Task Sampler ====================
class MedicalTaskSampler:
    def __init__(self, targets, n_way=4, k_shot=5, query=15, episodes=1000):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query = query
        self.episodes = episodes
        targets = np.array(targets)
        self.classes = np.unique(targets)
        self.idx_map = {c: np.where(targets == c)[0] for c in self.classes}

    def __len__(self): return self.episodes
    def __iter__(self):
        for _ in range(self.episodes):
            batch = []
            classes = np.random.choice(self.classes, self.n_way, replace=False)
            for c in classes:
                idxs = self.idx_map[c]
                replace = len(idxs) < (self.k_shot + self.query)
                chosen = np.random.choice(idxs, self.k_shot + self.query, replace=replace)
                batch.append(chosen)
            yield np.concatenate(batch)

# ==================== Training Epoch ====================
def train_epoch(model, chaos, loader, opt):
    model.train()
    total_loss = total_acc = 0
    for data, _ in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        data = data.view(4, 20, 3, 224, 224)  # (N_way, K+Q, C, H, W)
        support = data[:, :5].contiguous().view(-1, 3, 224, 224)
        query   = data[:, 5:].contiguous().view(-1, 3, 224, 224)
        y_query = torch.arange(4, device=device).repeat_interleave(15)

        z_s = model(support)
        z_q = model(query)
        z_s = chaos(z_s)                     # Chaos only on support
        proto = z_s.view(4, 5, -1).mean(1)    # Prototypes

        # Cosine similarity (L2-norm already applied)
        sim = torch.mm(z_q, proto.t())       # (60, 4)
        loss = F.cross_entropy(sim * model.scale, y_query)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        total_acc += (sim.argmax(1) == y_query).float().mean().item()

    return total_loss/len(loader), total_acc/len(loader)

# ==================== Validation ====================
@torch.no_grad()
def validate(model, loader):
    model.eval()
    accs = []
    for data, _ in loader:
        data = data.to(device)
        data = data.view(4, 20, 3, 224, 224)
        support = data[:, :5].contiguous().view(-1, 3, 224, 224)
        query   = data[:, 5:].contiguous().view(-1, 3, 224, 224)
        y_query = torch.arange(4, device=device).repeat_interleave(15)

        proto = model(support).view(4, 5, -1).mean(1)
        sim = torch.mm(model(query), proto.t())
        accs.append((sim.argmax(1) == y_query).float().mean().item())
    return np.mean(accs)

# ==================== MAIN ====================
if __name__ == "__main__":
    DATA_PATH = r"D:\Final Project\BrainTumorDataset"
    EPOCHS = 50
    EPISODES_TRAIN = 500
    EPISODES_VAL   = 600

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = ImageFolder(os.path.join(DATA_PATH, "train"), train_tf)
    val_ds   = ImageFolder(os.path.join(DATA_PATH, "test"),  val_tf)

    train_loader = DataLoader(train_ds, batch_sampler=MedicalTaskSampler(train_ds.targets, episodes=EPISODES_TRAIN))
    val_loader   = DataLoader(val_ds,   batch_sampler=MedicalTaskSampler(val_ds.targets,   episodes=EPISODES_VAL))

    model = MedicalProtoNet().to(device)
    chaos = LogisticChaos(intensity=0.18)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best = 0.0
    print("Starting Training...\n")
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_epoch(model, chaos, train_loader, opt)
        val_acc = validate(model, val_loader)
        sched.step()

        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Train: {acc*100:5.2f}% | Val: {val_acc*100:5.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), "chaos_18.pth")
            print("NEW BEST MODEL SAVED!")

    print(f"\nFINAL BEST ACCURACY: {best*100:.2f}%")
    print("Model saved as: chaos_18.pth")
