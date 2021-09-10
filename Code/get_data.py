# encoding: utf-8
import os
import sys
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

# stand output
std_out = sys.stdout

def get_RAF(img_dir, txt_dir):
    imgs_label_all = []
    with open(txt_dir, 'r', encoding='gb18030') as txtf:
        for line in txtf:
            name = line.split(' ')[0]
            path = os.path.join(img_dir,name)
            label = line.split(' ')[-1]
            label = int(label)-1
            imgs_label_all.append((path, label))
            print(path, file=std_out)
    return imgs_label_all

def load_imgs(img_dir, txt_dir, dataset_name):
    if dataset_name == 'RAF-DB':
        imgs_label_all = get_RAF(img_dir, txt_dir)
    return imgs_label_all


class ImageDataset(data.Dataset):
    def __init__(self, img_dir, txt_dir, dataset_name, transform=None):
        self.imgs_label_all = load_imgs(img_dir, txt_dir, dataset_name)
        self.transform = transform
        self.totensor = transforms.ToTensor()

    def __getitem__(self, index):
        path, label = self.imgs_label_all[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        img = img.resize((224,224))
        img = self.totensor(img)
        return img, label
    def __len__(self):
        return len(self.imgs_label_all)