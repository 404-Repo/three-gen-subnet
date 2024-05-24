import gc
import glob
import os
import sys
import inspect
import base64
import yaml
from typing import List
from time import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir+"/validation")

import torch
from loguru import logger

from validation.rendering_pipeline import RenderingPipeline
from validation.validation_pipeline import ValidationPipeline
from validation.hdf5_loader import HDF5Loader


def get_all_h5_file_names(folder_path: str):
    """ Function for getting all file names from the folder

    Parameters
    ----------
    folder_path: a path to the folder with files

    Returns
    -------
    a list with file names
    """
    h5_files = glob.glob(os.path.join(folder_path, '**/*.h5'), recursive=True)
    if len(h5_files) == 0:
        raise RuntimeWarning(f"No HDF5 files were found in <{folder_path}>. Nothing to process!")

    return h5_files


def validate(renderer: RenderingPipeline,
             validator: ValidationPipeline,
             data_dict: dict,
             prompt: str,
             views: int,
             cam_rad: float,
             gs_scale: float,
             data_ver: int):
    """ Function for validating the input data

    Parameters
    ----------
    renderer : a rendering pipeline object
    validator: a validation pipeline object
    data_dict: unpacked data that will be rendered
    prompt : an input prompt
    views: the amount of views to render
    cam_rad: the radius of the camera orbit
    gs_scale: the scaling factor for rendering the gaussian splatting model
    data_ver: version of the input data format: 0 - corresponds to dream gaussian
                                                1 - corresponds to new data format (default)

    Returns
    -------
    a set of rendered images, computed float score
    """
    # print("[INFO] Start validating the input 3D data.")
    logger.info(f" Input prompt: {prompt}")
    t1 = time()

    images = renderer.render_gaussian_splatting_views(data_dict, views, cam_rad, gs_scale=gs_scale, data_ver=data_ver)
    score = validator.validate(images, prompt)

    t2 = time()
    logger.info(f" Validation took: {t2 - t1} sec\n\n")

    return images, score


def load_config():
    """ Function that loads the data from config file

    Returns
    -------
    loaded config data as a dictionary
    """
    with open("local_test_runner_conf.yml", "r") as file:
        config_data = yaml.safe_load(file)
    assert config_data != {}
    return config_data


def match_promtps(prompts_in: List[str], file_names: List[str]):
    """ Function that matches input prompts with the name of the files

    Parameters
    ----------
    prompts_in: a list of input prompts
    file_names: a list of input file names

    Returns
    -------
    matched list of prompts
    """
    prompts = []
    for file_path in file_names:
        filename = os.path.basename(file_path)
        for prompt in prompts_in:
            if any(word in filename.split("_") for word in prompt.split(" ")):
                prompts.append(prompt)
                break
    return prompts


if __name__ == '__main__':
    # loading config file
    config_data = load_config()

    # getting all hdf5 file names
    h5_files = get_all_h5_file_names(config_data["hdf5_folder"])

    # reading prompts either from the file or from config
    if config_data["prompts_file"] != "":
        with open(config_data["prompts_file"]) as file:
            prompts_in = [line.rstrip() for line in file]
    elif len(config_data["prompts"]) > 0:
        prompts_in = config_data["prompts"]
    else:
        raise ValueError("No prompts were given by either 'prompts_file' or 'prompts' fields in the config! Nothing to be processed.")

    if len(prompts_in) == len(h5_files):
        prompts = match_promtps(prompts_in, h5_files)
    elif len(prompts_in) == 1:
        prompts = prompts_in
    else:
        raise ValueError("The amount of provided prompts should be the same as the amount of input files or "
                         "it should be 1 prompt for all files.")

    # prompts = ["a peace of jewelry", "a ghost", "a turtle", "goblin with a weapon",]
    save_images = config_data["save_images"]

    assert config_data["img_width"] != "" and config_data["img_width"] > 0
    assert config_data["img_height"] != "" and config_data["img_height"] > 0

    # preparing rendering system
    renderer = RenderingPipeline(config_data["img_width"], config_data["img_height"])

    # preparing validator
    validator = ValidationPipeline(debug=True)
    validator.preload_scoring_model()

    # preparing HDF5 loader
    hdf5_loader = HDF5Loader()

    assert config_data["iterations"] != "" and int(config_data["iterations"]) > 0

    # running validation loop
    for i in range(int(config_data["iterations"])):
        for j, h5_file in enumerate(h5_files):

            # loading h5 file
            file_name, _ = os.path.splitext(os.path.basename(h5_file))
            file_path = os.path.abspath(h5_file)
            file_path = os.path.dirname(file_path)

            print("[INFO] file path: ", h5_file)

            data_dict = hdf5_loader.load_point_cloud_from_h5(file_name, file_path)

            # preparing data
            data_io = hdf5_loader.pack_point_cloud_to_io_buffer(data_dict["points"],
                                                                data_dict["normals"],
                                                                data_dict["features_dc"],
                                                                data_dict["features_rest"],
                                                                data_dict["opacities"],
                                                                data_dict["scale"],
                                                                data_dict["rotation"],
                                                                data_dict["sh_degree"])
            data_io_encoded = base64.b64encode(data_io.getbuffer())

            # prompts preparing
            if len(prompts) == 1:
                prompt = prompts[0]
            else:
                assert len(prompts) == len(h5_files)
                prompt = prompts[j]

            # preloading data
            res, data_dict = renderer.prepare_data(data_io_encoded)
            if res:
                images, _ = validate(renderer,
                                     validator,
                                     data_dict,
                                     prompt,
                                     config_data["views"],
                                     config_data["cam_rad"],
                                     config_data["gs_scale"],
                                     config_data["data_ver"])

                if save_images:
                    path = os.path.join(os.curdir, "renders")
                    if not os.path.exists(path):
                        os.mkdir(path)
                    path = os.path.join(path, file_name)
                    os.mkdir(path)

                    renderer.save_rendered_images(images, file_name, path)

        gc.collect()
        torch.cuda.empty_cache()
