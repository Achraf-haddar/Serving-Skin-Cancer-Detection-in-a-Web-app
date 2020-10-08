from PIL import Image, ImageFile
import torch
import numpy as np 

class ClassificationDataset:
    def __init__(self, image_path, targets, resize=None, augmentations=None):
        self.image_path = image_path
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, item):
        image = Image.open(self.image_path[item])
        image = image.convert("RGB")
        targets = self.targets[item]
        if self.resize is not None:
            image = Image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
        image = np.array(image)
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"] 
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return{
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }
        