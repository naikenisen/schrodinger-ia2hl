import os
import shutil

def flatten_dataset(root_dir):
    for stain in os.listdir(root_dir):
        stain_path = os.path.join(root_dir, stain)
        if not os.path.isdir(stain_path):
            continue
        for label in os.listdir(stain_path):
            label_path = os.path.join(stain_path, label)
            if not os.path.isdir(label_path):
                continue
            for fname in os.listdir(label_path):
                src = os.path.join(label_path, fname)
                if os.path.isfile(src):
                    dst = os.path.join(stain_path, f"{label}_{fname}")
                    shutil.move(src, dst)
            # Optionally remove the now-empty label folder
            if not os.listdir(label_path):
                os.rmdir(label_path)

if __name__ == "__main__":
    root = "dataset_v2"  # Change this to your dataset root
    flatten_dataset(root)