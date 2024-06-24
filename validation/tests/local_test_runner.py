import gc
import glob
import os
import sys
import inspect
import yaml
from time import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/validation_lib")

import torch
import pandas as pd
from loguru import logger

from validation_lib.memory import enough_gpu_mem_available
from validation_lib.rendering.rendering_pipeline import RenderingPipeline
from validation_lib.validation.validation_pipeline import ValidationPipeline
from validation_lib.io.hdf5 import HDF5Loader


def get_all_h5_file_names(folder_path: str):
    """Function for getting all file names from the folder

    Parameters
    ----------
    folder_path: a path to the folder with files

    Returns
    -------
    a list with file names
    """
    h5_files = glob.glob(os.path.join(folder_path, "**/*.h5"), recursive=True)
    h5_files.sort(key=os.path.getctime)

    if len(h5_files) == 0:
        raise RuntimeWarning(f"No HDF5 files were found in <{folder_path}>. Nothing to process!")

    return h5_files


def validate(
    renderer: RenderingPipeline,
    validator: ValidationPipeline,
    data_dict: dict,
    prompt: str,
    views: int,
    cam_rad: float,
    data_ver: int,
):
    """Function for validating the input data

    Parameters
    ----------
    renderer : a rendering pipeline object
    validator: a validation pipeline object
    data_dict: unpacked data that will be rendered
    prompt : an input prompt
    views: the amount of views to render
    cam_rad: the radius of the camera orbit
    data_ver: version of the input data format: 0 - corresponds to dream gaussian
                                                1 - corresponds to new data format (default)

    Returns
    -------
    a set of rendered images, computed float score
    """
    # print("[INFO] Start validating the input 3D data.")
    logger.info(f" Input prompt: {prompt}")
    t1 = time()

    images = renderer.render_gaussian_splatting_views(data_dict, views, cam_rad, data_ver=data_ver)
    score = validator.validate(images, prompt)

    t2 = time()
    logger.info(f" Validation took: {t2 - t1} sec\n\n")

    return images, score


def load_config():
    """Function that loads the data from config file

    Returns
    -------
    loaded config data as a dictionary
    """
    with open("local_test_runner_conf.yml", "r") as file:
        config_data = yaml.safe_load(file)
    assert config_data != {}
    return config_data


def load_prompts(config_data: dict):
    """

    Parameters
    ----------
    config_data

    Returns
    -------

    """
    if config_data["prompts_file"] != "":
        with open(config_data["prompts_file"]) as file:
            prompts_in = [line.rstrip() for line in file]
    elif len(config_data["prompts"]) > 0:
        prompts_in = config_data["prompts"]
    else:
        raise ValueError(
            "No prompts were given by either 'prompts_file' or 'prompts' fields in the config! "
            "Nothing to be processed."
        )

    return prompts_in


def main():
    # loading config file
    config_data = load_config()

    # save images enable/disable
    save_images = config_data["save_images"]
    debug_output = True

    assert config_data["img_width"] != "" and config_data["img_width"] > 0
    assert config_data["img_height"] != "" and config_data["img_height"] > 0
    assert config_data["iterations"] != "" and int(config_data["iterations"]) > 0

    # getting all hdf5 file names
    h5_files = get_all_h5_file_names(config_data["hdf5_folder"])
    logger.info(f" Files to process: {h5_files}")

    # reading prompts either from the file or from config
    prompts = load_prompts(config_data)
    logger.info(f" Prompts used for generating files: {prompts}")

    # preparing rendering system
    renderer = RenderingPipeline(config_data["img_width"], config_data["img_height"])

    # preparing validator
    validator = ValidationPipeline(verbose=debug_output)
    validator.preload_model()

    # preparing HDF5 loader
    hdf5_loader = HDF5Loader()

    data = []
    for i in range(int(config_data["iterations"])):
        data.append(["Iteration_" + str(i)])

    # running validation_lib loop
    for i in range(int(config_data["iterations"])):
        for j, h5_file in enumerate(h5_files):
            # loading h5 file
            file_name, _ = os.path.splitext(os.path.basename(h5_file))
            file_path = os.path.abspath(h5_file)
            file_path = os.path.dirname(file_path)

            logger.info(f" File Path: {h5_file}")

            data_dict = hdf5_loader.from_file(file_name, file_path)

            # prompts preparing
            if len(prompts) == 1:
                prompt = prompts[0]
            else:
                assert len(prompts) == len(h5_files)
                prompt = prompts[j]

            if not enough_gpu_mem_available(data_dict):
                return

            images, score = validate(
                renderer,
                validator,
                data_dict,
                prompt,
                config_data["views"],
                config_data["cam_rad"],
                config_data["data_ver"],
            )
            data[i].append(score)

            if save_images:
                path = os.path.join(os.curdir, "renders")
                if not os.path.exists(path):
                    os.mkdir(path)
                path = os.path.join(path, file_name)
                os.mkdir(path)

                renderer.save_rendered_images(images, file_name, path)

        gc.collect()
        torch.cuda.empty_cache()

    # saving statistics to .csv file
    cols = [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
    cols.insert(0, "Iteration")

    df = pd.DataFrame(data, columns=cols)
    df.to_csv("statistics.csv", float_format="%.5f")


if __name__ == "__main__":
    t1 = time()
    main()
    t2 = time()
    logger.info(f"Total time spent: {(t2-t1)/60.0} min")
