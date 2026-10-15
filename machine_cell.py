import czifile
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.morphology import closing, disk
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
from datetime import datetime, date

# Load CZI file
CZI_PATH = ["111_Scan.czi","59_Scan.czi","35_Scan.czi","107_Scan.czi","81_Scan.czi","127_Scan.czi","74_Scan.czi"]

print(datetime.now())

# Cell counting per slice 
def count_cells(ch_slice):
    if ch_slice.max() == 0:
        return 0
    t = threshold_otsu(ch_slice)
    mask = ch_slice > t
    mask = remove_small_objects(mask, max_size=9)  # remove noise pixels
    if mask.mean() <= 0.02: # filter
        return None
    return len(regionprops(label(mask)))    

def gen_mask(fitc, hoechst, tritc):
    combined = np.maximum(fitc, np.maximum(hoechst, tritc))
    if combined.max() == 0:
        return None, None
    t = threshold_otsu(combined)
    mask = combined > t
    mask = remove_small_objects(mask, max_size=50)  # remove noise pixels
    mask = closing(mask, disk(5))
    mask = binary_fill_holes(mask)


    #labeled = label(mask)
    #if labeled.max() == 0:
    #    return None
    #sizes = np.bincount(labeled.ravel())
    #sizes[0] = 0
    #mask = labeled == sizes.argmax()


    if mask.mean() <= 0.02: # filter
        return None, None
    return mask, combined

class SpheroidDataSet(torch.utils.data.Dataset):
    def __init__(self, slices, masks, augment = False):
        self.slices = slices
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        normal_slice = self.slices[index] / 255
        normal_mask = self.masks[index]

        if self.augment:
            if np.random.rand() < 0.5:
                normal_slice = np.fliplr(normal_slice)
                normal_mask = np.fliplr(normal_mask)

            k = np.random.randint(4) 
            normal_slice = np.rot90(normal_slice, k)
            normal_mask = np.rot90(normal_mask, k)

        return normal_slice.copy(), normal_mask.copy()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias= False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False),
            nn.ReLU(inplace=True))

    def forward(self, x):
            return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels= 1, features = [64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size= 2, stride= 2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            concat_skip = torch.cat((skip_connection, x), dim = 1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

stack_slice = {}
stack_mask = {}
stack_z = {}

for n in CZI_PATH:
    czi = czifile.CziFile(n)
    arr = czi.asarray()

    fitc    = arr[0]   # channel 0 — FITC (green)
    hoechst = arr[1]   # channel 1 — Hoechst (nuclei, blue)
    tritc   = arr[2]   # channel 2 — TRITC (red)

    n_slices = fitc.shape[0]
    print(f"Loaded: {n_slices} z-slices, image size {fitc.shape[1]}x{fitc.shape[2]}")

    fitc_counts    = []
    hoechst_counts = []
    tritc_counts   = []
    ratios         = [] 
    stack_slice[n] = []
    stack_mask[n] = []
    stack_z[n] = []


    for z in range(n_slices):
        
        mask, combined = gen_mask(fitc[z], hoechst[z], tritc[z])

        if mask is not None:
            stack_slice[n].append(combined)
            stack_mask[n].append(mask)
            stack_z[n].append(z)

        f  = count_cells(fitc[z])
        h  = count_cells(hoechst[z])
        tr = count_cells(tritc[z])

        fitc_counts.append(f)
        hoechst_counts.append(h)
        tritc_counts.append(tr)

        if f and tr and tr > 0:
            ratio = f / tr
        else:
           ratio = np.nan
         
        ratios.append(ratio)

    z_axis = np.arange(n_slices)    

trainSL = stack_slice["111_Scan.czi"] + stack_slice["59_Scan.czi"] + stack_slice["35_Scan.czi"]
trainMA = stack_mask["111_Scan.czi"] + stack_mask["59_Scan.czi"] + stack_mask["35_Scan.czi"]

valSL = stack_slice["107_Scan.czi"] + stack_slice["81_Scan.czi"]
valMA = stack_mask["107_Scan.czi"] + stack_mask["81_Scan.czi"]

testSL = stack_slice["127_Scan.czi"] + stack_slice["74_Scan.czi"]
testMA = stack_mask["127_Scan.czi"] + stack_mask["74_Scan.czi"]


train_dataset = SpheroidDataSet(trainSL, trainMA)
val_dataset = SpheroidDataSet(valSL, valMA)
test_dataset = SpheroidDataSet(testSL, testMA)

def loss_func(predictions, targets, smooth = 1):
    intersection = (predictions * targets).sum(dim = (1, 2, 3))
    denom = predictions.sum(dim = (1,2,3)) + targets.sum(dim = (1,2,3))
    dice = (2 * intersection + smooth) / (denom + smooth)
    dice_loss = 1 - dice.mean()

    bce = torch.nn.functional.binary_cross_entropy(predictions, targets)

    return dice_loss + bce


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = UNET(in_channels=1, out_channels=1).to(device)
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location= device))
    print('Saved')
else:
    print("Fresh")

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)


dataset = SpheroidDataSet(trainSL, trainMA, augment= False)
dataloader = DataLoader(dataset=dataset, batch_size= 8, shuffle=False)

val_dataset = SpheroidDataSet(valSL, valMA)
val_loader = DataLoader(val_dataset, batch_size= 8, shuffle= False)


MODE = "view"  # "train", "test", "checkbest", "view", "view_dice", "augcheck "

print(datetime.now())

with open("Time_Log.txt", "a") as f:
            print("-" * 60, file=f)

if MODE == "train":
    best_val = float('inf')

    for epoch in range(120):

        start = time.time()
        model.train()
        train_losses = []
        test_dice = []
        for images, masks in dataloader:
            images = images.float().unsqueeze(1).to(device)
            masks = masks.float().unsqueeze(1).to(device)
            predictions = torch.sigmoid(model(images))
            train_losses.append(loss_func(predictions, masks).item())

            loss = loss_func(predictions, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_dice = np.array(test_dice)
        train_loss = np.mean(train_losses)

        elapsedTrain = time.time() - start

        val_start = time.time()
        model.eval()
        val_losses = []
        val_dice = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.float().unsqueeze(1).to(device)
                masks = masks.float().unsqueeze(1).to(device)
                val_predictions = torch.sigmoid(model(images))
                val_losses.append(loss_func(val_predictions, masks).item())

                hard = (val_predictions > 0.5).float()
                intersec = (hard * masks).sum(dim = (1,2,3))
                denom = hard.sum(dim = (1,2,3)) + masks.sum(dim = (1,2,3))
                d = (2 * intersec + 1e-6) / (denom + 1e-6)
                val_dice.extend(d.cpu().tolist())

        val_dice = np.array(val_dice)

        print(f"Test Dice: mean {val_dice.mean():.3f}, median {np.median(val_dice):.3f}, min {val_dice.min():.3f}")
        np.save("test_dice.npy", val_dice)

        val_loss = np.mean(val_losses)

        elapsedVal = time.time() - val_start

        torch.save(model.state_dict(), "model.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f} ({elapsedTrain/60:.1f} min), Val Loss: {val_loss:.4f} ({elapsedVal/60:.1f} min)")


        with open("Time_Log.txt", "a") as f:
            datetime.now()
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f} ({elapsedTrain/60:.1f} min), Val Loss: {val_loss:.4f} ({elapsedVal/60:.1f} min)", file=f)

    with open("Time_Log.txt", "a") as f:
            print("-" * 60, file=f)



elif MODE == "test":
    model.load_state_dict(torch.load('best_model.pth', map_location= device))
    model.eval()

    test_dataset = SpheroidDataSet(testSL, testMA)
    test_loader = DataLoader(test_dataset, batch_size= 8 , shuffle = False)

    test_losses = []
    test_dice = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.float().unsqueeze(1).to(device)
            masks = masks.float().unsqueeze(1).to(device)
            prediction = torch.sigmoid(model(images))
            test_losses.append(loss_func(prediction, masks).item())

            hard = (prediction > 0.5).float()
            intersec = (hard * masks).sum(dim = (1,2,3))
            denom = hard.sum(dim = (1,2,3)) + masks.sum(dim = (1,2,3))
            d = (2 * intersec + 1e-6) / (denom + 1e-6)
            test_dice.extend(d.cpu().tolist())

    test_dice = np.array(test_dice)
    np.save("test_dice.npy", test_dice)

    print(f"Test loss: {np.mean(test_losses):.4f}")
    print(f"Test Dice: mean {np.mean(test_dice):.3f}, median {np.median(test_dice):.3f}, min {np.min(test_dice):.3f}")

if MODE == "checkbest":
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    val_losses, val_dice = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.float().unsqueeze(1).to(device)
            masks = masks.float().unsqueeze(1).to(device)
            preds = torch.sigmoid(model(images))
            val_losses.append(loss_func(preds, masks).item())

            hard = (preds > 0.5).float()
            inter = (hard * masks).sum(dim = (1,2,3))
            denom = hard.sum(dim = (1,2,3)) + masks.sum(dim = (1,2,3))
            val_dice.extend(((2 * inter + 1e-6) /  (denom + 1e-6)).cpu().tolist())

    val_dice = np.array(val_dice)
    print(f"Best model -> Val loss: {np.mean(val_losses):.4f}, Test Dice: mean {val_dice.mean():.3f}, median {np.median(val_dice):.3f}, min {val_dice.min():.3f}")
    np.save("val_dice_best.npy", val_dice)

torch.save(model.state_dict(), "model.pth")

if MODE == "augcheck":
    check_dataset = SpheroidDataSet(trainSL, trainMA, augment=True)
    for i in range(5):
        image, mask = check_dataset[i]
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image, cmap='gray')
        axes[1].imshow(mask, cmap='gray')
        plt.savefig(f"aug_check_{i}.png")
        plt.close()
    print("Saved aug check images")

if MODE == "view":
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    val_dataset = SpheroidDataSet(valSL, valMA)
    test_dataset = SpheroidDataSet(testSL, testMA)

    for idx in [91,88,92,87,89,93,90,94,86,95]:
        image, mask = test_dataset[idx]
        x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Raw image")
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Otsu mask")
        axes[2].imshow(pred > 0.5, cmap='gray')
        axes[2].set_title("Model prediction")
        plt.savefig(f"val_view_{idx}.png")
        plt.close()

if MODE == "view_dice":
    val_dice = np.load("test_dice.npy")
    val_stacks = ["127_Scan.czi", "74_Scan.czi"]
    print(len(np.load("test_dice.npy")))
    plt.savefig("test_dice_by_slice.png", dpi=150)

    sizes = [len(stack_z[s]) for s in val_stacks]
    assert sum(sizes) == len(val_dice), f"stacks have {sum(sizes)} slices {sizes}, file has {len(val_dice)} scores"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    start = 0
    rows = []
    for ax, s, size in zip(axes, val_stacks, sizes):
        d = val_dice[start:start + size]
        z = stack_z[s]
        cov = [m.mean() for m in stack_mask[s]]

        ax.plot(z, d, "o-", ms=3, color="tab:blue")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Dice", color="tab:blue")
        ax.set_xlabel("z-slice")
        ax.set_title(s)

        ax2 = ax.twinx()
        ax2.plot(z, cov, color="gray", alpha=0.5)
        ax2.set_ylabel("Mask coverage", color="gray")

        for i in range(size):
            rows.append((d[i], s, z[i], cov[i], start + i))
        start += size

    plt.tight_layout()
    plt.savefig("val_dice_by_slice.png", dpi=150)
    plt.close()

    print("Worst 10 val slices:")
    for dice, s, z, cov, idx in sorted(rows)[:10]:
        print(f"  {s} z={z:3d}  Dice={dice:.3f}  coverage={cov:.3f}  val_idx={idx}")

print("")
print(datetime.now())
print("")