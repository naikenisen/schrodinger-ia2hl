import os
import sys
import argparse
import numpy as np
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import config as cfg
from models.unet import UNetModel
from dataloader import get_test_dataloader, TEST_IHC
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

class Langevin(torch.nn.Module):
    """
    Implémente une trajectoire de sampling par dynamique de Langevin.

    Concept débutant :
    - on part d'un état initial (images ou bruit)
    - on applique plusieurs petites mises à jour
    - à chaque étape, on ajoute un bruit aléatoire
    - le réseau (net) sert à guider ces mises à jour
    """   
     
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
        
        x_tot = torch.zeros(N, self.num_steps, *self.d).to(x.device)

        with torch.no_grad():
            for k in range(self.num_steps):
                gamma = self.gammas[k]

                net_out = net(x, steps[:, k, :])
                
                if self.mean_match:
                    t_old = net_out
                else:
                    t_old = x + net_out

                if k == self.num_steps - 1:
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(2 * gamma) * z
                
                x_tot[:, k, :] = x
        
        return x_tot, x

def get_model(device):

    att_ds = tuple(256 // int(res) for res in cfg.ATTENTION_RESOLUTIONS.split(","))
    net = UNetModel(
            model_channels=cfg.NUM_CHANNELS,
            num_res_blocks=cfg.NUM_RES_BLOCKS,
            attention_resolutions=att_ds,
            dropout=cfg.DROPOUT,
            channel_mult=(1, 1, 2, 2, 4, 4)
        )

    return net.to(device)

def save_results(input_batch, output_batch, output_dir, batch_idx, batch_fnames=None, cd30_dir=None):
    """
    Sauvegarde HES (input), CD30 virtuel (output), CD30 réel (paired) côte à côte.
    """
    os.makedirs(output_dir, exist_ok=True)
    batch_size = input_batch.shape[0]
    
    for i in range(batch_size):
        hes_img = to_pil_image(input_batch[i].cpu())
        cd30_virtual_img = to_pil_image(output_batch[i].cpu())
        cd30_real_img = None
        
        # Récupération de l'image réelle grâce au nom de fichier du batch
        if batch_fnames is not None and cd30_dir is not None:
            fname = batch_fnames[i]
            cd30_real_path = os.path.join(cd30_dir, fname)
            if os.path.exists(cd30_real_path):
                cd30_real_img = Image.open(cd30_real_path).convert('RGB')

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(hes_img)
        axes[0].set_title('HES')
        axes[0].axis('off')
        
        axes[1].imshow(cd30_virtual_img)
        axes[1].set_title('CD30 virtuel')
        axes[1].axis('off')
        
        if cd30_real_img is None:
            arr = np.zeros((hes_img.size[1], hes_img.size[0], 3), dtype=np.uint8)
            axes[2].set_title('CD30 réel (introuvable)')
        else:
            arr = np.array(cd30_real_img)
            axes[2].set_title('CD30 réel')
            
        axes[2].imshow(arr)
        axes[2].axis('off')

        plt.tight_layout()
        
        # Bonus : On utilise le VRAI nom du fichier pour sauvegarder l'image finale
        # C'est beaucoup plus facile pour retrouver tes résultats ensuite !
        if batch_fnames is not None:
            out_name = os.path.join(output_dir, f"result_{batch_fnames[i]}")
        else:
            out_name = os.path.join(output_dir, f"pred_{batch_idx}_{i}.png")
            
        plt.savefig(out_name)
        plt.close(fig)

def run_inference(ckpt_path, output_dir='./results'):
    device = torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    net = get_model(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.eval()

    n = cfg.NUM_STEPS // 2
    gamma_half = np.linspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n) if cfg.GAMMA_SPACE == 'linspace' else np.geomspace(cfg.GAMMA_MIN, cfg.GAMMA_MAX, n)
    gammas = np.concatenate([gamma_half, np.flip(gamma_half)])
    gammas = torch.tensor(gammas).to(device)
    
    langevin = Langevin(cfg.NUM_STEPS, (cfg.CHANNELS, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), gammas, device=device, mean_match=True)

    test_loader = get_test_dataloader()

    print("Starting generation...")
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            # Ton dataloader renvoie (image, filename)
            batch = data[0].to(device)
            batch_fnames = data[1] # <--- Liste des noms de fichiers de ce batch
            
            _, final_image = langevin.sample(net, batch)
            
            # On passe batch_fnames et TEST_IHC directement à save_results
            save_results(batch, final_image, output_dir, i, batch_fnames, TEST_IHC)

checkpoint = 'checkpoints/net_f_10.ckpt' # Exemple de chemin vers un checkpoint
out_dir = 'results' # Exemple de répertoire de sortie

run_inference(checkpoint, out_dir)