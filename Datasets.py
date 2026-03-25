
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import cv2
import glob
import os
import torchvision.transforms as transforms
import random
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
to_tensor = transforms.Compose([transforms.ToTensor()])
p = 128

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.pt"))))
    data.sort()
    filenames.sort()

    pt_files = [f for f in data if f.endswith('.pt')]
    if pt_files:
        return pt_files, [os.path.basename(f) for f in pt_files]
    return data, filenames

class Fusion_dataset(Dataset):
    def __init__(self, transform = to_tensor):
        super(Fusion_dataset, self).__init__()
        data_dir_vis = ''
        data_dir_ir = ''

        data_dir_vitext = ''
        data_dir_irtext = ''

        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)

        self.filepath_vitext, self.filenames_vitext = prepare_data_path(data_dir_vitext)
        self.filepath_irtext, self.filenames_irtext = prepare_data_path(data_dir_irtext)

        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        self.transform  = transform

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        if self.filepath_vitext:
            vitext_path = self.filepath_vitext[min(index, len(self.filepath_vitext) - 1)]
        else:
            vitext_path = None
            
        if self.filepath_irtext:
            irtext_path = self.filepath_irtext[min(index, len(self.filepath_irtext) - 1)]
        else:
            irtext_path = None

        image_vis = np.asarray(Image.open(vis_path), dtype=np.float32)/255.0
        image_inf = cv2.imread(ir_path, 0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0

        try:
            if vitext_path:
                vitext_data = torch.load(vitext_path, map_location='cpu')
                if isinstance(vitext_data, torch.Tensor):
                    vitext = vitext_data.numpy()
                else:
                    vitext = vitext_data
            else:
                vitext = np.zeros((1, 512), dtype=np.float32)
        except Exception as e:
            print(f"加载vitext失败: {e}")
            vitext = np.zeros((1, 512), dtype=np.float32)
        
        try:
            if irtext_path:
                irtext_data = torch.load(irtext_path, map_location='cpu')
                if isinstance(irtext_data, torch.Tensor):
                    irtext = irtext_data.numpy()
                else:
                    irtext = irtext_data
            else:
                irtext = np.zeros((1, 512), dtype=np.float32)
        except Exception as e:
            print(f"加载irtext失败: {e}")
            irtext = np.zeros((1, 512), dtype=np.float32)

        image_vis = self.transform(image_vis)
        image_ir = self.transform(image_ir)

        vitext = self.transform(vitext)
        irtext = self.transform(irtext)
        name = self.filenames_vis[index]

        C, H, W = image_vis.shape
        y = random.randint(0,H-p-1)
        x = random.randint(0,W-p-1)
        image_ir = image_ir[:,y:y+p,x:x+p]
        image_vis = image_vis[:,y:y + p, x:x + p]
        return image_vis, image_ir, vitext, irtext, name

    def __len__(self):
        return self.length
class Test_dataset(Dataset):
    def __init__(self, transform = to_tensor):
        super(Test_dataset, self).__init__()

        data_dir_vis = ''
        data_dir_ir = ''

        data_dir_vitext = ''
        data_dir_irtext = ''

        self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
        self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)

        self.filepath_vitext, self.filenames_vitext = prepare_data_path(data_dir_vitext)
        self.filepath_irtext, self.filenames_irtext = prepare_data_path(data_dir_irtext)

        self.length = min(len(self.filenames_vis), len(self.filenames_ir))
        self.transform  = transform

    def __getitem__(self, index):
        vis_path = self.filepath_vis[index]
        ir_path = self.filepath_ir[index]

        if self.filepath_vitext:
            vitext_path = self.filepath_vitext[min(index, len(self.filepath_vitext) - 1)]
        else:
            vitext_path = None
            
        if self.filepath_irtext:
            irtext_path = self.filepath_irtext[min(index, len(self.filepath_irtext) - 1)]
        else:
            irtext_path = None

        image = Image.open(vis_path)
        width,height = image.size
        image_vis = np.asarray(Image.open(vis_path), dtype=np.float32)/255.0
        image_inf = cv2.imread(ir_path, 0)
        image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
        image_vis = self.transform(image_vis)
        image_ir = self.transform(image_ir)

        try:
            if vitext_path:
                vitext = torch.load(vitext_path, map_location='cpu')
                if isinstance(vitext, torch.Tensor):
                    vitext = vitext.numpy()
                vitext = self.transform(vitext)
            else:
                vitext = torch.zeros(1, 1, 512).float()
        except Exception as e:
            print(f"加载vitext失败: {e}")
            vitext = torch.zeros(1, 1, 512).float()
        
        try:
            if irtext_path:
                irtext = torch.load(irtext_path, map_location='cpu')
                if isinstance(irtext, torch.Tensor):
                    irtext = irtext.numpy()
                irtext = self.transform(irtext)
            else:
                irtext = torch.zeros(1, 1, 512).float()
        except Exception as e:
            print(f"加载irtext失败: {e}")
            irtext = torch.zeros(1, 1, 512).float()

        name = self.filenames_vis[index]

        return image_vis, image_ir, vitext, irtext, name


    def __len__(self):
        return self.length