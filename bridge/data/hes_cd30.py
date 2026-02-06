import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class HES_CD30(Dataset):
    """Dataset for paired HES / CD30 virtual staining.
    
    Expected folder structure:
        root/
        ├── HES/
        │   ├── c/
        │   │   ├── patch_x2000_y32000.jpg
        │   │   └── ...
        │   ├── e/
        │   └── ...
        └── CD30/
            ├── c/
            │   ├── patch_x2000_y32000.jpg
            │   └── ...
            ├── e/
            └── ...
    
    Each HES image has a paired CD30 image with the same subfolder
    and filename.
    """

    def __init__(self, root, image_size=256, domain='HES', transform=None):
        """
        Args:
            root: path to dataset_v2/ directory
            image_size: resize images to this size
            domain: 'HES' or 'CD30'
            transform: optional torchvision transform (overrides default)
        """
        super().__init__()
        self.root = root
        self.domain = domain
        self.image_size = image_size

        # Collect all image paths from the chosen domain
        domain_dir = os.path.join(root, domain)
        self.image_paths = sorted(glob.glob(os.path.join(domain_dir, '*', '*.jpg')))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {domain_dir}. "
                f"Expected structure: {domain_dir}/<subfolder>/*.jpg"
            )

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

        print(f"[HES_CD30] Loaded {len(self.image_paths)} images from {domain_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, 0  # 0 is a dummy label for compatibility
