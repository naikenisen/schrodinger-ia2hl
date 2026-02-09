import os
import sys
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# Imports locaux
import config as cfg
from models.unet import UNetModel
from dataloader import dataloader

# ============================================================================
#  REUSE: LANGEVIN (Copie nécessaire pour avoir la même physique qu'en train)
# ============================================================================

def grad_gauss(x, m, var):
    return -(x - m) / var

class Langevin(torch.nn.Module):
    def __init__(self, num_steps, shape, gammas, device=None, mean_match=True):
        super().__init__()
        self.mean_match = mean_match
        self.num_steps = num_steps
        self.d = shape
        self.gammas = gammas.float()
        self.device = device if device is not None else gammas.device
        self.time = torch.cumsum(self.gammas, 0).to(self.device).float()

    def sample(self, net, init_samples):
        """
        Effectue l'inférence (sampling) sur init_samples en utilisant le réseau 'net'.
        Retourne la trajectoire complète et l'image finale.
        """
        x = init_samples
        N = x.shape[0]
        steps = self.time.reshape((1, self.num_steps, 1)).repeat((N, 1, 1))
        
        # Pour stocker la trajectoire si on veut faire un GIF
        x_tot = torch.zeros(N, self.num_steps, *self.d).to(x.device)

        with torch.no_grad():
            for k in range(self.num_steps):
                gamma = self.gammas[k]
                
                # Le réseau prédit le saut
                net_out = net(x, steps[:, k, :])
                
                if self.mean_match:
                    t_old = net_out
                else:
                    t_old = x + net_out

                # Pas de bruit à la toute dernière étape pour une meilleure netteté (optionnel)
                if k == self.num_steps - 1:
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                
                x_tot[:, k, :] = x
        
        return x_tot, x

# ============================================================================
#  UTILS
# ============================================================================

def get_model(device):
    """Initialise le modèle avec la config"""
    image_size = cfg.IMAGE_SIZE
    if image_size == 256: channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64: channel_mult = (1, 2, 3, 4)
    elif image_size == 32: channel_mult = (1, 2, 2, 2)
    else: raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = [image_size // int(res) for res in cfg.ATTENTION_RESOLUTIONS.split(",")]

    kwargs = {
        "in_channels": cfg.CHANNELS, "model_channels": cfg.NUM_CHANNELS,
        "out_channels": cfg.CHANNELS, "num_res_blocks": cfg.NUM_RES_BLOCKS,
        "attention_resolutions": tuple(attention_ds), "dropout": cfg.DROPOUT,
        "channel_mult": channel_mult, "use_checkpoint": cfg.USE_CHECKPOINT,
        "num_heads": cfg.NUM_HEADS, "num_heads_upsample": cfg.NUM_HEADS_UPSAMPLE,
        "use_scale_shift_norm": cfg.USE_SCALE_SHIFT_NORM,
    }
    net = UNetModel(**kwargs)
    return net.to(device)

def get_test_dataloader():
    """Charge le Dataset de TEST spécifiquement"""
    test_transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    
    root = os.path.join(cfg.DATA_DIR, 'dataset_v2')
    
    # NOTE: Ici, on suppose que le DataLoader a une option pour le 'test' set.
    # Si votre classe dataloader ne gère pas 'split', il faudra pointer vers le dossier test.
    # Pour cet exemple, on charge le domaine 'HES' (source) pour le transformer en 'CD30'
    test_ds = dataloader(root, image_size=cfg.IMAGE_SIZE, domain='HES', transform=test_transform)
    
    # On peut limiter la taille pour l'inférence ou prendre tout le set
    loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)
    return loader

def save_results(input_batch, output_batch, output_dir, batch_idx):
    """Sauvegarde les images input vs output"""
    # Denormalize if necessary (assuming [0,1] range here)
    comparison = torch.cat([input_batch, output_batch], dim=3) # Concatenate side-by-side
    vutils.save_image(comparison, f"{output_dir}/pred_{batch_idx}.png", nrow=4, normalize=False)

# ============================================================================
#  INFERENCE ENGINE
# ============================================================================

def run_inference(ckpt_path, output_dir='./results'):
    device = torch.device(cfg.DEVICE)
    print(f"Running inference on {device} using {ckpt_path}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Charger le modèle
    net = get_model(device)
    try:
        # Load weights
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Gestion si le checkpoint contient 'state_dict' ou est le dictionnaire directement
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    net.eval()

    # 2. Configurer Langevin Dynamics
    n = cfg.NUM_STEPS // 2
    gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n) if cfg.GAMMA_SPACE == 'linspace' else np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
    gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    gammas = torch.tensor(gammas).to(device)
    
    langevin = Langevin(cfg.NUM_STEPS, (cfg.CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), gammas, device=device, mean_match=cfg.MEAN_MATCH)

    # 3. Charger les données de TEST
    test_loader = get_test_dataloader()
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # 4. Boucle d'inférence
    print("Starting generation...")
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            # Adapter selon ce que retourne le dataloader (image, label) ou juste (image)
            if isinstance(data, list):
                batch = data[0].to(device)
            else:
                batch = data.to(device)
            
            # Exécuter Langevin (Inférence)
            # On passe le batch initial (HES) et le réseau génère la transformation
            _, final_image = langevin.sample(net, batch)
            
            # Sauvegarder
            save_results(batch, final_image, output_dir, i)

    print(f"Inference done. Results saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the model checkpoint (.ckpt)')
    parser.add_argument('--out', type=str, default='./results', help='Directory to save results')
    args = parser.parse_args()

    run_inference(args.ckpt, args.out)