import uvicorn
import os
import sys
import inspect
import argparse
from fastapi import FastAPI

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + "/mining")

import ValidationTextTo3DModel as validation
import mining.DreamGaussianLib.HDF5Loader as HDF5Loader

from neurons.protocol import TextTo3D
from time import time

import base64


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    args, extras = parser.parse_known_args()
    return args, extras


app = FastAPI()
args, _ = get_args()

#################################
# For testing
#################################
# loader = HDF5Loader.HDF5Loader()
# pcl_data = loader.load_point_cloud_from_h5("bear", "/home/tesha/Documents/Python/404_dream_gaussian/mining/logs/output8")
# pcl_buffer = loader.pack_point_cloud_to_io_buffer(pcl_data['points'], pcl_data['normals'], pcl_data['features_dc'], pcl_data['features_rest'],
#                                                   pcl_data['opacities'], pcl_data['scale'], pcl_data['rotation'], pcl_data['sh_degree'])
# pcl_buffer_encode = base64.b64encode(pcl_buffer.getbuffer())
#
# prompt = "an image of the bear"
# message = TextTo3D(prompt_in=prompt, mesh_out=pcl_buffer_encode)


@app.get("/validate")
# def validate_gs_result(save_images: bool = False):
def validate_gs_result(message: TextTo3D, save_images: bool = False):
    print("[INFO] Start validating the input 3D data.")
    print("[INFO] Input prompt: ", message.prompt_in)

    t1 = time()

    Validator = validation.ValidateTextTo3DModel(512, 512, 10)
    Validator.init_gaussian_splatting_renderer()
    scores = Validator.score_response_gs_input([message], save_images=save_images, cam_rad=4)

    t2 = time()
    print("[INFO] Score: ", scores)
    print("[INFO] Validation took: ", t2 - t1, " sec")
    return {"score": scores.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)
