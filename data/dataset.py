import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CPDDDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_length=80):
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.image_paths, self.poem_paths = self._load_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        poem_path = self.poem_paths[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with open(poem_path, 'r', encoding='utf-8') as f:
            poem = f.read().strip()

        # Convert poem to tensor of indices
        poem_tensor = torch.zeros(self.max_length, dtype=torch.long)
        poem_chars = list(poem)[:self.max_length]
        for i, char in enumerate(poem_chars):
            poem_tensor[i] = ord(char)  # Use Unicode code point as index

        return image, poem_tensor

    def _load_data(self):
        image_paths = []
        poem_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
                    poem_paths.append(os.path.join(root, file[:-4] + '.txt'))
        return image_paths, poem_paths
