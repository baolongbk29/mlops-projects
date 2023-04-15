import torch
import hydra
import logging
from omegaconf.omegaconf import OmegaConf

from src.model import TIMMModel
from src.dataset import ChestXrayDataset, ChestXrayDatamodule


logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="default")
def convert_model(cfg):


    # Setup Path, logger 
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/weights/epoch=12-step=8475.ckpt"
    logger.info(f"Loading pre-trained model from: {model_path}")

    # Model and data loader
    model = TIMMModel(cfg.model)
    model.load_from_checkpoint(model_path)
    data = ChestXrayDatamodule(cfg.dataset)
    data.setup()


    input_batch = next(iter(data.train_dataloader()))
    input_sample = {
        "image": input_batch[0],
    }

    # Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        model,  # model being run
        (
            input_sample["image"],
           
        ),  # model input (or a tuple for multiple inputs)

        f"{root_dir}/weights/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=10,
        input_names=["image"],  # the model's input names
        output_names=["label"],  # the model's output names
        dynamic_axes={
            "image": {0: "batch_size"},  # variable length axes
            "label": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {root_dir}/models/model.onnx"
    )


if __name__ == "__main__":
    convert_model()



