import torch
from omegaconf import OmegaConf

from .AIModelsUtils.mvdream_utils import MVDream
from .AIModelsUtils.imagedream_utils import ImageDream
from .AIModelsUtils.sd_utils import StableDiffusion
from .AIModelsUtils.zero123_utils import Zero123


def preload_model(opt: OmegaConf, device: str):
    models = []
    torch_device = torch.device(device)
    if opt.mvdream:
        print("[INFO] loading MVDream...")
        model1 = MVDream(torch_device)
        models.append(model1)
        print("[INFO] loaded MVDream!")
    if opt.imagedream:
        print("[INFO] loading ImageDream...")
        model2 = ImageDream(torch_device)
        models.append(model2)
        print("[INFO] loaded ImageDream!")
    if opt.stablediff:
        print("[INFO] loading SD...")
        model3 = StableDiffusion(torch_device)
        models.append(model3)
        print("[INFO] loaded SD!")
    if opt.stable_zero123:
        print("[INFO] loading stable zero123...")
        model4 = Zero123(torch_device, model_key="ashawkey/stable-zero123-diffusers")
        models.append(model4)
        print("[INFO] loaded stable zero123!")
    if opt.zero123_xl:
        print("[INFO] loading zero123-xl...")
        model5 = Zero123(torch_device, model_key="ashawkey/zero123-xl-diffusers")
        models.append(model5)
        print("[INFO] loaded zero123-xl!")

    if (len(models) == 0) or (len(models) > 2):
        raise ValueError("Check config file. Specified model tag parameter is absent.")

    return models
