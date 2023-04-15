from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import sys
import logging
from omegaconf import OmegaConf


from inference_onnx import setting_inference, onnx_predictor

app = FastAPI(title="Check Xray Classification MLops")
config_path = "outputs/chest_xray/.hydra/config.yaml"
model_path = "weights/model.onnx"
label_name = ['NORMAL','PNEUMONIA']

cfg = OmegaConf.load(config_path)
ort_session, transforms = setting_inference(cfg, model_path)



@app.get("/")
def home():
    return "<h2>Test API</h2>"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')    

    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert('RGB')
        pred = onnx_predictor(ort_session, transforms, image)
        logging.info(f"predict: {label_name[pred[0]]}")
        
        return {
            "filename": file.filename, 
            "contentype": file.content_type,            
            "likely_class": label_name[pred[0]],
        }

    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))