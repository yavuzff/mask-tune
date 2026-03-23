import os
import pandas as pd
import logging
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, root='./data/CelebA/raw', img_dir=None, train=True, transform=None):
        """
        CelebA dataset mapping target (Blond Hair) and confounder (Male).
        """
        self.root = root
        self.train = train
        self.transform = transform

        # if an explicit image directory is provided (like a masked folder), use it.
        # otherwise, default to the standard raw CelebA image folder.
        if img_dir is not None:
            self.img_dir = img_dir
        else:
            self.img_dir = os.path.join(self.root, 'img_align_celeba')

        # load partitions (0: train, 1: val, 2: test)
        partition_df = pd.read_csv(os.path.join(self.root, 'list_eval_partition.csv'))
        attrs_df = pd.read_csv(os.path.join(self.root, 'list_attr_celeba.csv'))

        # merge partitions and attributes
        df = pd.merge(partition_df, attrs_df, on='image_id')

        # filter split
        target_split = 0 if self.train else 2
        self.df = df[df['partition'] == target_split].reset_index(drop=True)

        # extract target (Blond_Hair) and confounder (Male)
        # CelebA uses 1 for True, and -1 for False. We map -1 to 0.
        self.df['Blond_Hair'] = self.df['Blond_Hair'].replace(-1, 0)
        self.df['Male'] = self.df['Male'].replace(-1, 0)

        # convert to lists for fast lookup
        self.image_ids = self.df['image_id'].tolist()
        self.targets = self.df['Blond_Hair'].tolist()
        self.confounders = self.df['Male'].tolist()

        logging.info(f"Generating {'Training' if train else 'Testing'} CelebA dataset from {self.img_dir}: {len(self.image_ids)} samples.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_filename = self.image_ids[idx]
        img_file_path = os.path.join(self.img_dir, img_filename)

        img = Image.open(img_file_path).convert('RGB')

        target = self.targets[idx]
        confounder = self.confounders[idx]

        if self.transform:
            img = self.transform(img)

        return img, target, img_file_path, confounder


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)

    celeba_data = CelebADataset(
        root='./data/CelebA/raw',
        train=True,
        transform=None
    )

    hair_map = {0: 'Dark Hair', 1: 'Blond'}
    gender_map = {0: 'Female', 1: 'Male'}

    print("Dataset size:", len(celeba_data))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        img, target, img_path, confounder = celeba_data[i]

        axes[i].imshow(img)
        hair_name = hair_map[target]
        gender_name = gender_map[confounder]

        axes[i].set_title(f"Hair: {hair_name}\nGender: {gender_name}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()