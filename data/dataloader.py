from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import CPDDDataset
from config import config

def get_dataloader(split='train'):
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CPDDDataset(
        root_dir=f"{config.data_path}/{split}",
        transform=transform,
        max_length=config.max_poem_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.num_workers
    )

    return dataloader
