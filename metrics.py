# benchmark.py (version: utilise CD30 virtuel déjà généré)
# pip install torchmetrics lpips tqdm pytorch-fid torchvision pillow

import os
import re
import torch
from PIL import Image
from tqdm import tqdm
import src.config as config
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from src.data_loader import create_dataloaders, get_image_paths, set_seed, RANDOM_SEED
import lpips

CKPT = "best_models/batch-32-scale-1.0-lambda-10.0-width-256-lrg-0.0001-lrd-0.0001.pth"
BATCH_SIZE = 4
NUM_WORKERS = 0
LPIPS_NET = "alex"
SEED = 42
img_range = "[-1, 1]"
img_width = config.IMG_WIDTH
img_height = config.IMG_HEIGHT


ihc_dir = "dataset/test/CD30"  # ground truth CD30 réels du test set ex: c_patch_x4000_y30000.jpg
inference_dir = "inference/CD30"  # virtual CD30 c_patch_x4000_y30000_VIRTUAL.png

# List all image files in the directories
def list_image_files(directory, exts=(".jpg", ".png")):
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    ])


# Apparier les images par nom de base (sans extension ni _VIRTUAL)
ihc_files = list_image_files(ihc_dir, exts=(".jpg",))
inference_files = list_image_files(inference_dir, exts=(".png",))

# Créer un dict {base_name: path} pour les virtuels
def get_base_name(path, virtual=False):
    name = os.path.splitext(os.path.basename(path))[0]
    if virtual and name.endswith('_VIRTUAL'):
        name = name[:-8]
    return name

virtual_dict = {get_base_name(f, virtual=True): f for f in inference_files}

# Apparier chaque image réelle à son virtuel par nom de base
paired_files = []
for real_path in ihc_files:
    base = get_base_name(real_path)
    virt_path = virtual_dict.get(base)
    if virt_path:
        paired_files.append((real_path, virt_path))

ihc_files = [r for r, v in paired_files]
inference_files = [v for r, v in paired_files]

# Main benchmarking function
def benchmark(img_width, img_height, lpips_net, seed, img_range, virtual_dir, ihc_files, inference_files):

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  
    set_seed(RANDOM_SEED)

    import random
    combined = list(zip(ihc_files, inference_files))
    random.shuffle(combined)
    ihc_files, inference_files = zip(*combined)
    ihc_files, inference_files = list(ihc_files), list(inference_files)

    total_size = len(ihc_files)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    test_inference_files = inference_files[train_size + valid_size:]  # liste des CD30 réels du test set

    # Transform pour lire les PNG virtuels en tensor [0,1]
    from torchvision import transforms
    to_tensor = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),  # [0,1]
    ])

    # Initialize metrics once
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    lpips_model = lpips.LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    lpips_sum = 0.0
    num_images = 0

    # Loop through DataLoader
    with torch.no_grad():
        for x_path, y_path in tqdm(combined):
            # x_path: path to real image, y_path: path to virtual image
            # Load images and convert to tensors
            x_img = Image.open(x_path).convert("RGB")
            y_img = Image.open(y_path).convert("RGB")
            x = to_tensor(x_img).unsqueeze(0).to(device)  # [1,3,H,W]
            y = to_tensor(y_img).unsqueeze(0).to(device)  # [1,3,H,W]

            if img_range == '[-1, 1]':
                x = (x + 1) / 2
                y = (y + 1) / 2
            elif img_range == '[0, 1]':
                pass
            else:
                raise ValueError('Invalid range specified.')

            x = x.clamp(0, 1)
            y = y.clamp(0, 1)

            # Ensure 3 channels for metrics (au cas où y serait 1ch)
            x_3ch = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            y_3ch = y.repeat(1, 3, 1, 1) if y.shape[1] == 1 else y

            # Update PSNR and SSIM metrics
            psnr_metric.update(y_3ch, x_3ch)
            ssim_metric.update(y_3ch, x_3ch)

            # Update FID metric
            fid_metric.update(x_3ch, real=True)
            fid_metric.update(y_3ch, real=False)

            # Calculate LPIPS for this batch
            x_lpips = 2 * x_3ch - 1
            y_lpips = 2 * y_3ch - 1
            lpips_batch = lpips_model(y_lpips, x_lpips).mean()
            lpips_sum += lpips_batch.item() * x.shape[0]
            num_images += x.shape[0]

    # Compute final metrics
    psnr_score = psnr_metric.compute().item()
    ssim_score = ssim_metric.compute().item()
    fid_score = fid_metric.compute().item()
    lpips_score = lpips_sum / num_images

    # Summary
    print(f'Number of images evaluated: {num_images}')
    print(f'Mean PSNR: {psnr_score:.2f}')
    print(f'Mean SSIM: {ssim_score:.2f}')
    print(f'Mean LPIPS: {lpips_score:.2f}')
    print(f'FID: {fid_score:.2f}')

benchmark(
    img_width=img_width,
    img_height=img_height,
    lpips_net=LPIPS_NET,
    seed=SEED,
    img_range=img_range,
    virtual_dir=inference_dir,
    ihc_files=ihc_files,
    inference_files=inference_files
)