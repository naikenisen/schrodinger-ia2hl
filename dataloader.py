import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import config as cfg

# --- Définition des chemins (pour référence ou usage global si besoin) ---
# Ces chemins sont intégrés dans la logique de la classe ci-dessous.
TRAIN_HES = "/work/imvia/in156281/ia2hl/dataset_tiled_512/train/HES"
TRAIN_IHC = "/work/imvia/in156281/ia2hl/dataset_tiled_512/train/CD30"

VALID_HES = "/work/imvia/in156281/ia2hl/dataset_tiled_512/valid/HES"
VALID_IHC = "/work/imvia/in156281/ia2hl/dataset_tiled_512/valid/CD30"

TEST_HES = "/work/imvia/in156281/ia2hl/dataset_tiled_512/test/HES"
TEST_IHC = "/work/imvia/in156281/ia2hl/dataset_tiled_512/test/CD30"


class PairedDataset(Dataset):
    def __init__(self, split='train', domain='HES', transform=None):
        """
        Args:
            split (str): 'train', 'valid' ou 'test'.
            domain (str): 'HES' ou 'CD30'.
            transform (callable, optional): Transformation à appliquer sur les images.
        """
        super().__init__()
        self.transform = transform
        self.domain = domain.upper()
        self.split = split.lower()

        # 1. Définition des chemins en fonction du split
        if self.split == 'train':
            hes_dir = TRAIN_HES
            ihc_dir = TRAIN_IHC
        elif self.split == 'valid':
            hes_dir = VALID_HES
            ihc_dir = VALID_IHC
        elif self.split == 'test':
            hes_dir = TEST_HES
            ihc_dir = TEST_IHC
        else:
            raise ValueError(f"Split inconnu : {split}. Utilisez 'train', 'valid' ou 'test'.")

        # Vérification de l'existence des dossiers
        if not os.path.exists(hes_dir) or not os.path.exists(ihc_dir):
            raise FileNotFoundError(f"Dossiers introuvables pour le split '{split}' : \n{hes_dir}\n{ihc_dir}")

        # 2. Logique d'appariement (Pairing)
        # On ne garde que les noms de fichiers qui existent dans les DEUX dossiers (Intersection)
        files_hes = set(os.listdir(hes_dir))
        files_ihc = set(os.listdir(ihc_dir))
        
        # On trie la liste pour garantir un ordre déterministe (important pour la reproductibilité)
        self.common_files = sorted(list(files_hes & files_ihc))
        
        if len(self.common_files) == 0:
            print(f"Attention : Aucun fichier commun trouvé entre {hes_dir} et {ihc_dir} !")

        # 3. Définition du dossier source selon le domaine demandé
        self.source_dir = hes_dir if self.domain == 'HES' else ihc_dir

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, index):
        fname = self.common_files[index]
        img_path = os.path.join(self.source_dir, fname)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {img_path}: {e}")
            # En cas d'erreur, on pourrait renvoyer une image noire ou relancer l'erreur
            raise e
        
        if self.transform:
            img = self.transform(img)
            
        return img, fname

def get_datasets():
    """
    Charge les datasets d'entraînement pour HES et CD30.
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        # Penser à ajouter une normalisation ici si nécessaire, ex: transforms.Normalize(...)
    ])

    # On instancie les datasets pour le split 'train'
    init_ds = PairedDataset(split='train', domain='HES', transform=transform)
    final_ds = PairedDataset(split='train', domain='CD30', transform=transform)

    # Valeurs par défaut conservées de votre code original
    mean_final = torch.tensor(0.)
    var_final = torch.tensor(1000.)

    return init_ds, final_ds, mean_final, var_final

def get_test_dataloader():
    """
    Crée le DataLoader pour le test (domaine HES par défaut, ou adaptable).
    """
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # On charge le split 'test'
    dataset = PairedDataset(split='test', domain='HES', transform=transform)

    return DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )