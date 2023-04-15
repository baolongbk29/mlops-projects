from omegaconf import OmegaConf
import torch
import os
import argparse
from PIL import Image
import torch.nn.functional as F

from src.model import TIMMModel
from src.dataset import get_image_transforms




def get_model(cfg):
        
        model = TIMMModel(cfg.model)
        ckpt = torch.load(r"D:\MLops-Projects\outputs\chest_xray\lightning_logs\version_9\checkpoints\epoch=12-step=8475.ckpt")
        model.load_state_dict(ckpt["state_dict"])

        return model


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Inference Chest Xray")
    parser.add_argument("-test-dir", "--directory", default="test.png")
    parser.add_argument("-cfg", "--config", default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_name = ['NORMAL','PNEUMONIA']
    model = get_model(cfg).eval().to(device)

    transforms = get_image_transforms(cfg.dataset.crop_size, False, None)

    for label in os.listdir(args.directory):
        for file_name in os.listdir(os.path.join(args.directory, label)):

            image = Image.open(os.path.join(args.directory, label, file_name)).convert('RGB')

            image = transforms(image).unsqueeze(0)
            image = image.to(device)
            logit = model(image)
            logit = F.softmax(logit, dim=-1)
            preds = torch.argmax(logit, 1) 
            print(f"predict: {label_name[preds]} ground_truth: {label}")