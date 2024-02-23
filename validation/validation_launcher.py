import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/mining")

import base64
import numpy as np
import ValidationTextTo3DModel as validation

import mining.DreamGaussianLib.HDF5Loader as HDF5Loader
from neurons.protocol import TextTo3D
from time import time


if __name__ == "__main__":
    loader = HDF5Loader.HDF5Loader()

    t1 = time()
    Validator = validation.ValidateTextTo3DModel(512, 512, 10)

    data_dict = loader.load_point_cloud_from_h5(
        "frog", "/home/tesha/Documents/Python/404_dream_gaussian/mining/logs/output7"
    )
    pcl_buffer = loader.pack_point_cloud_to_io_buffer(
        data_dict["points"],
        data_dict["normals"],
        data_dict["features_dc"],
        data_dict["features_rest"],
        data_dict["opacities"],
        data_dict["scale"],
        data_dict["rotation"],
        data_dict["sh_degree"],
    )

    pcl_buffer_encode = base64.b64encode(pcl_buffer.getbuffer())
    pcl_buffer = base64.b64decode(pcl_buffer_encode)

    # prompt = "A Golden Poison Dart Frog"
    prompt = "Yellow Black Poison Dart Frog"
    synapse = TextTo3D(prompt_in=prompt, mesh_out=pcl_buffer_encode)

    print("[INFO] Start validating the input 3D data.")
    print("[INFO] Input prompt: ", synapse.prompt_in)

    t3 = time()

    Validator.init_gaussian_splatting_renderer()
    scores = Validator.score_response_gs_input(prompt, [synapse, synapse], save_images=False, cam_rad=4)

    t4 = time()

    print("[INFO] The validation score is: ", scores)
    print("[INFO] Validation took: ", (t4 - t3) / 60.0, " min")
