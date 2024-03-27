import argparse
import os
import inspect

from omegaconf import OmegaConf
from time import time

import DreamGaussianLib.GaussianProcessor as GaussianProcessor
import DreamGaussianLib.ModelsPreLoader as MLoader
import DreamGaussianLib.HDF5Loader as HDF5Loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--fname", required=True, help="name of the file that will be used for saving")
    args, extras = parser.parse_known_args()

    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    outpath = os.path.join(currentdir, opt.outdir, opt.save_path)
    os.makedirs(outpath, exist_ok=True)

    total_start = time()
    print("\n\n[INFO] Preloading models:")
    t1 = time()
    models = MLoader.preload_model(opt, "cuda")
    t2 = time()
    print("[INFO] It took: ", (t2 - t1) / 60.0, " min.")

    t3 = time()
    print("\n\n[INFO] Start training Gaussian Splatting.")
    gaussian_processor = GaussianProcessor.GaussianProcessor(opt)
    (
        xyz,
        normals,
        f_dc,
        f_rest,
        opacities,
        scale,
        rotation,
        sh_degree,
    ) = gaussian_processor.train(models, opt.iters)
    t4 = time()
    print("[INFO] Finished: ", (t4 - t3) / 60.0, " min.\n")

    hdf5_loader = HDF5Loader.HDF5Loader()
    hdf5_loader.save_point_cloud_to_h5(
        xyz,
        normals,
        f_dc,
        f_rest,
        opacities,
        scale,
        rotation,
        sh_degree,
        args.fname,
        outpath,
    )
