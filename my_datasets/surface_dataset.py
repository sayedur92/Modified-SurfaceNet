import sys
sys.path.append('/home/hpc/vlgm/vlgm103v/genmatpro/idea/')
import torch
import os
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from my_datasets.utils import textures_mapping, texture_maps


map_names = ["real", "diffuse", "normal", "roughness", "specular"]


def MapTransform(load_size=256):
    return T.Compose([
        T.Resize(load_size),
        T.CenterCrop(load_size),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])

class SurfaceDataset(Dataset):
    
    def __init__(self, dset_dir, sketch_dir=None, load_size=256):
        
        self.dset_dir = Path(dset_dir)
        self.sketch_dir = Path(sketch_dir) if sketch_dir else None #change
        self.files = list(self.dset_dir.glob("**/*.png"))  
    
        self.sketch_files = {}
        if self.sketch_dir: #change
            for f in self.sketch_dir.glob("*.png"):
                prefix = f.stem  
                self.sketch_files[prefix] = f

        self.load_size = load_size
        self.transforms = MapTransform(load_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        
        img = Image.open(file_path)
        width, height = img.size

        if width == 1440 and height == 288:
            segment_width = width // len(map_names)
            maps = {}

            for i, map_name in enumerate(map_names):
                map_image = img.crop((
                    i * segment_width, 0,
                    (i + 1) * segment_width, height
                ))
                maps[map_name] = self.transforms(map_image)

            
            maps["render"] = maps["real"]
        else:
            
            maps = {"render": self.transforms(img)}
        

        file_prefix = file_path.stem  
        sketch_path = self.sketch_files.get(file_prefix) if self.sketch_dir else None #change

        if sketch_path:
            sketch_image = Image.open(sketch_path).convert("L")  
            maps["sketch"] = self.transforms(sketch_image)
        else:
            print(f"Warning: Sketch map for {file_prefix} not found in {self.sketch_dir}")
            maps["sketch"] = torch.zeros(1, self.load_size, self.load_size)  

        
        return maps


class PicturesDataset(Dataset):
    
    def __init__(self, dset_dir, sketch_dir, load_size=256):
        
        self.dset_dir = Path(dset_dir)
        self.sketch_dir = Path(sketch_dir)
        self.files = list(self.dset_dir.glob('*.png')) 
        
        self.sketch_files = {}
        for f in self.sketch_dir.glob("*.png"):
            prefix = f.stem.split(';')[0]  
            self.sketch_files[prefix] = f

        self.transforms = MapTransform(load_size)

    def __len__(self):
        """Dataset size."""
        return len(self.files)

    def __getitem__(self, index):
        
        
        img_path = self.files[index]
        file_prefix = img_path.stem.split(';')[0]  
        sketch_path = self.sketch_files.get(file_prefix)

        if sketch_path is None:
            print(f"Warning: Sketch map for {file_prefix} not found in {self.sketch_dir}")
            sketch = None 
        else:
            sketch = Image.open(sketch_path).convert("L")  

        img = Image.open(img_path).convert("RGB")

        
        img = self.transforms(img)
        sketch = self.transforms(sketch) if sketch else None

        return {"image": img, "sketch": sketch}
 