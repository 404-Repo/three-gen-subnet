import gc
import glob
import os
import sys
import inspect
import base64
import yaml
from time import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir+"/lib")

import torch

from lib.rendering_pipeline import Renderer
from lib.validation_pipeline import Validator
from lib.hdf5_loader import HDF5Loader


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


def validate(renderer, validator,  data: str, prompt: str):
    """ Function for validating the input data

    Parameters
    ----------
    renderer : a GS renderer object
    validator: a validator object
    data : input data stored as encoded bytes data
    prompt : an input prompt

    Returns
    -------
    a set of rendered images, computed float score
    """
    # print("[INFO] Start validating the input 3D data.")
    print(f"[INFO] Input prompt: {prompt}")
    t1 = time()

    images = renderer.render_gaussian_splatting_views(data, 15, 5.0)
    score = validator.validate(images, prompt)

    t2 = time()
    print(f"[INFO] Validation took: {t2 - t1} sec\n\n")

    return images, score


def load_config():
    """ Function that loads the data from config file

    Returns
    -------
    loaded config data as a dictionary
    """
    with open("test_runner_config.yml", "r") as file:
        config_data = yaml.safe_load(file)
    assert config_data != {}
    return config_data


def match_promtps(prompts_in: list, file_names: list):
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
        raise Validator("No prompts were given by either 'prompts_file' or 'prompts' fields in the config! Nothing to be processed.")

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
    renderer = Renderer(config_data["img_width"], config_data["img_height"])
    renderer.init_gaussian_splatting_renderer()

    # preparing validator
    validator = Validator(debug=True)
    validator.preload_scoring_model()

    # preparing HDF5 loader
    hdf5_loader = HDF5Loader()

    assert config_data["iterations"] != "" and int(config_data["iterations"]) > 0

    # running validation loop
    for i in range(int(config_data["iterations"])):
        for j, h5_file in enumerate(h5_files):

            # loading h5 file
            file_name, _ = os.path.splitext(os.path.basename(h5_file))
            file_name = file_name.strip("_pcl")
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
            # prompts preparing
            if len(prompts) == 1:
                prompt = prompts[0]
            else:
                assert len(prompts) == len(h5_files)
                prompt = prompts[j]

            data_io_encoded = base64.b64encode(data_io.getbuffer())
            images = validate(renderer, validator, data_io_encoded, prompt)

            if save_images:
                path = os.path.join(os.curdir, "renders")
                if not os.path.exists(path):
                    os.mkdir(path)

                for ind, img in enumerate(images):
                    img_path = os.path.join(path, file_name + "_image_" + str(ind)+".png")
                    img.save(img_path)

        gc.collect()
        torch.cuda.empty_cache()
