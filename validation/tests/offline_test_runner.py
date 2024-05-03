import glob
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir+"/lib")

import base64
import torch

from lib.rendering_pipeline import Renderer
from lib.validation_pipeline import Validator

from lib.hdf5_loader import HDF5Loader
from time import time


def get_all_h5_file_names(folder_path: str):
    h5_files = glob.glob(os.path.join(folder_path, '**/*.h5'), recursive=True)
    return h5_files


def validate(data: str, prompt: str):
    print("[INFO] Start validating the input 3D data.")
    print(f"[INFO] Input prompt: {prompt}")
    t1 = time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nUsing device:', device, "\n")

    renderer = Renderer(512, 512)
    renderer.init_gaussian_splatting_renderer()
    images = renderer.render_gaussian_splatting_views(data, 20, 5.0)

    validator = Validator()
    validator.preload_scoring_model()
    score = validator.validate(images, prompt)

    t2 = time()
    print(f"[INFO] Score: {score}")
    print(f"[INFO] Validation took: {t2 - t1} sec")


if __name__ == '__main__':
    h5_files = get_all_h5_file_names(parentdir + "/h5_files")
    print(h5_files)
    prompts = ["a peace of jewelry", "a ghost", "a turtle", "goblin with a weapon",]

    hdf5_loader = HDF5Loader()
    for h5_file, prompt in zip(h5_files, prompts):
        file_name, _ = os.path.splitext(os.path.basename(h5_file))
        file_name = file_name.split("_")[0]
        file_path = os.path.abspath(h5_file)
        file_path = os.path.dirname(file_path)
        # print(file_path, " ; ", file_name)

        data_dict = hdf5_loader.load_point_cloud_from_h5(file_name, file_path)
        data_io = hdf5_loader.pack_point_cloud_to_io_buffer(data_dict["points"],
                                                            data_dict["normals"],
                                                            data_dict["features_dc"],
                                                            data_dict["features_rest"],
                                                            data_dict["opacities"],
                                                            data_dict["scale"],
                                                            data_dict["rotation"],
                                                            data_dict["sh_degree"])

        data_io_encoded = base64.b64encode(data_io.getbuffer())
        validate(data_io_encoded, prompt)
