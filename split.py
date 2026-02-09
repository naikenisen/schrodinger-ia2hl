import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_dataset_v4(src_root, dst_root, split_ratio=0.8):
    # Chemins sources
    src_hes = os.path.join(src_root, 'HES')
    src_cd30 = os.path.join(src_root, 'CD30')

    # 1. Identifier les paires valides (présentes dans les deux dossiers)
    hes_files = set(os.listdir(src_hes))
    cd30_files = set(os.listdir(src_cd30))
    paired_files = sorted(list(hes_files & cd30_files))
    
    print(f"🔍 Paires trouvées : {len(paired_files)}")
    print(f"⚠️ Fichiers ignorés (non appariés) : {len(hes_files ^ cd30_files)}")

    # 2. Split train/test
    train_files, test_files = train_test_split(
        paired_files, 
        train_size=split_ratio, 
        random_state=42,
        shuffle=True
    )

    # 3. Création de la structure de destination
    for split in ['train', 'test']:
        for folder in ['HES', 'CD30']:
            os.makedirs(os.path.join(dst_root, split, folder), exist_ok=True)

    # 4. Fonction de copie
    def copy_set(file_list, split_name):
        print(f"📦 Copie du set {split_name}...")
        for fname in tqdm(file_list):
            # Copie HES
            shutil.copy2(
                os.path.join(src_hes, fname),
                os.path.join(dst_root, split_name, 'HES', fname)
            )
            # Copie CD30
            shutil.copy2(
                os.path.join(src_cd30, fname),
                os.path.join(dst_root, split_name, 'CD30', fname)
            )

    # Exécution
    copy_set(train_files, 'train')
    copy_set(test_files, 'test')

    print(f"\n✅ Terminé ! Dataset v4 créé dans : {dst_root}")
    print(f"Détail : {len(train_files)} paires en train, {len(test_files)} paires en test.")

if __name__ == "__main__":
    SRC = "/home/naiken/coding/diffusion_schrodinger_bridge/dataset_v3"
    DST = "dataset_v4"
    create_dataset_v4(SRC, DST)