import numpy as np
import torch
import onnxruntime as ort
import argparse
from omegaconf import OmegaConf
import os




from PIL import Image
from src.dataset import get_image_transforms


def setting_inference(cfg, model_path):

    ort_session = ort.InferenceSession(model_path)
    transforms = get_image_transforms(cfg.dataset.crop_size, False, None)

    return ort_session, transforms

def onnx_predictor(ort_session, transforms, image):
    image = transforms(image)
    ort_inputs = {
                    "image": np.expand_dims(image, axis=0),
                }
    ort_outs = ort_session.run(None, ort_inputs)    
    pred = np.argmax(ort_outs[0] , 1) 
    return pred

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Inference Chest Xray")
    parser.add_argument("-test-dir", "--directory", default="test.png")
    parser.add_argument("-cfg", "--config", default=None)
    parser.add_argument("-model", "--model_path", default=None)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_name = ['NORMAL','PNEUMONIA']

    ort_session, transforms = setting_inference(cfg, model_path=args.model_path)

    for label in os.listdir(args.directory):
        for file_name in os.listdir(os.path.join(args.directory, label)):

            image = Image.open(os.path.join(args.directory, label, file_name)).convert('RGB')
            pred = onnx_predictor(ort_session, transforms, image)
            print(f"predict: {label_name[pred[0]]} ground_truth: {label}")
 