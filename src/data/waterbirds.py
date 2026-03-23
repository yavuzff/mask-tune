import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import logging


class WaterbirdsDataset(Dataset):
    def __init__(self, root='./data/Waterbirds/waterbird_complete95_forest2water2', train=True, transform=None):
        """
        Waterbirds dataset mapping target (bird type) and confounder (background).
        """
        self.root = root
        self.train = train
        self.transform = transform

        # load metadata
        csv_path = os.path.join(self.root, 'metadata.csv')
        self.metadata = pd.read_csv(csv_path)

        # map train=True to split 0 (train), and train=False to split 2 (test).
        # we skip split 1 (val) as it is unused.
        target_split = 0 if self.train else 2
        self.metadata = self.metadata[self.metadata['split'] == target_split].reset_index(drop=True)

        logging.info(
            f"Generating {'Training' if train else 'Testing'} ORIGINAL Waterbirds dataset: {len(self.metadata)} samples.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # the img_filename is a relative path like '001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg'
        img_filename = row['img_filename']
        img_file_path = os.path.join(self.root, img_filename)

        # load image
        img = Image.open(img_file_path).convert('RGB')

        # extract target (bird) and confounder (background)
        target = int(row['y'])
        confounder = int(row['place'])

        if self.transform:
            img = self.transform(img)

        # return the 4 items for ERM and mask generation
        return img, target, img_file_path, confounder


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)


    # potentially define a train transform
    # from torchvision import transforms
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    waterbirds_data = WaterbirdsDataset(root='./data/Waterbirds/waterbird_complete95_forest2water2', train=True, transform=None)

    bird_map = {0: 'Landbird', 1: 'Waterbird'}
    bg_map = {0: 'Land', 1: 'Water'}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i in range(5):
        img, target, img_path, confounder = waterbirds_data[i]
        axes[i].imshow(img)
        bird_name = bird_map[target]
        bg_name = bg_map[confounder]
        axes[i].set_title(f"Bird: {bird_name}\nBg: {bg_name}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
