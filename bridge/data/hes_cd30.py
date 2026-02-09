import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class HES_CD30(Dataset):

    def __init__(self, root, image_size=256, transform=None):
        super().__init__()
        self.root = root
        self.image_size = image_size

        hes_dir = os.path.join(root, 'HES')
        cd30_dir = os.path.join(root, 'CD30')

        hes_files = set(os.listdir(hes_dir))
        cd30_files = set(os.listdir(cd30_dir))
        # On ne garde que les fichiers présents dans les deux dossiers
        paired_files = sorted(list(hes_files & cd30_files))
        if len(paired_files) == 0:
            raise RuntimeError(
                f"Aucune paire trouvée entre {hes_dir} et {cd30_dir}. "
                f"Vérifiez que les noms de fichiers correspondent."
            )
        self.paired_files = paired_files
        self.hes_dir = hes_dir
        self.cd30_dir = cd30_dir

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

        print(f"[HES_CD30] {len(self.paired_files)} paires HES/CD30 trouvées.")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, index):
        fname = self.paired_files[index]
        hes_path = os.path.join(self.hes_dir, fname)
        cd30_path = os.path.join(self.cd30_dir, fname)
        hes_img = Image.open(hes_path).convert('RGB')
        cd30_img = Image.open(cd30_path).convert('RGB')
        hes_img = self.transform(hes_img)
        cd30_img = self.transform(cd30_img)
        return hes_img, cd30_img, fname
