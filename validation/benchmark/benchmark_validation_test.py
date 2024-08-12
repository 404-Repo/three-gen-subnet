import gc
from time import time

import numpy as np
import pandas as pd
import torch
from benchmark_utils.benchmark_loader import BenchmarkLoader
from benchmark_utils.benchmark_runner import BenchmarkRunner
from loguru import logger


def main() -> None:
    benchmark_loader = BenchmarkLoader()

    # loading config file
    config_data = benchmark_loader.load_config()

    # save images enable/disable
    save_images = config_data["save_images"]
    verbose_output = config_data["verbose_output"]
    debug_output = config_data["debug_output"]

    if not (config_data["img_width"] and config_data["img_width"] > 0):
        raise ValueError("Image width must be a positive number")
    if not (config_data["img_height"] and config_data["img_height"] > 0):
        raise ValueError("Image height must be a positive number")
    if not (config_data["iterations"] and int(config_data["iterations"]) > 0):
        raise ValueError("Number of iterations must be a positive integer")

    # getting all file names
    files = benchmark_loader.get_all_data_files(config_data["data_folder"])
    logger.info(f" Files to process: {files}")

    # reading prompts either from the file or from config
    prompts = benchmark_loader.load_prompts(config_data)
    logger.info(f" Prompts used for generating files: {prompts}")

    benchmark_runner = BenchmarkRunner(config_data["views"], verbose=verbose_output, debug=debug_output)

    file_names_list = []
    scores_list = []
    iterations_list = []

    # running validation_lib loop
    for i in range(int(config_data["iterations"])):
        images, scores, _, file_names = benchmark_runner.run_validation_benchmark(
            config_data, prompts, files, save_images
        )
        if save_images:
            for j in range(len(file_names)):
                benchmark_runner.save_rendered_images(images[j], "renders", file_names[j])
            save_images = False

        file_names_list = file_names
        scores_list.append(scores)
        iterations_list.append("iteration_" + str(i))

        gc.collect()
        torch.cuda.empty_cache()

    data = {"Iteration": iterations_list}
    scores_list_final = np.array(scores_list).squeeze().transpose()
    for i, scores in enumerate(scores_list_final):
        data[file_names_list[i]] = scores  # type: ignore

    df = pd.DataFrame(data)
    df.to_csv("statistics.csv", float_format="%.5f")


if __name__ == "__main__":
    t1 = time()
    main()
    t2 = time()
    logger.info(f"Total time spent: {(t2 - t1)/60.0} min")
