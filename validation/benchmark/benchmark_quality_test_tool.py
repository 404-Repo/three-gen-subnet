from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_utils.benchmark_loader import BenchmarkLoader
from benchmark_utils.benchmark_plotter import Plotter
from benchmark_utils.benchmark_runner import BenchmarkRunner
from loguru import logger


def main() -> None:
    benchmark_loader = BenchmarkLoader()

    # loading config file
    config_data = benchmark_loader.load_config()
    template_file = config_data["template_path"]
    debug_output = config_data["debug_output"]
    save_images = config_data["save_images"]
    save_previews = config_data["save_previews"]
    generate_raw_template = config_data["generate_raw_template"]
    evaluate_validation = config_data["evaluate_validation"]

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

    benchmark_runner = BenchmarkRunner(config_data["views"], debug=debug_output)
    logger.info(f" Validating input data: {len(files)} files.")
    images, preview_images, score_sets, _, file_names = benchmark_runner.run_validation_benchmark(
        config_data, prompts, files, save_images, save_previews
    )

    scores = [scores.final_score for scores in score_sets]
    data = {"files": file_names, "prompt": prompts}
    data["final_score"] = [score.final_score for score in score_sets]
    data["quality_score"] = [score.quality_score for score in score_sets]
    data["clip_score"] = [score.clip_score for score in score_sets]
    data["ssim_score"] = [score.ssim_score for score in score_sets]
    data["lpis_score"] = [score.lpips_score for score in score_sets]
    data["sharpness"] = [score.sharpness_score for score in score_sets]

    df = pd.DataFrame(data)
    data_folder_path = Path(config_data["data_folder"])
    df.to_csv(data_folder_path.parent.name + ".csv", float_format="%.5f")

    plotter = Plotter()
    dataset_name = Path(config_data["data_folder"]).name
    plotter.plot_line_chart([np.array(data["final_score"])], "ply_file", "final_score", dataset_name)
    plotter.plot_line_chart([np.array(data["quality_score"])], "ply_file", "quality_score", dataset_name)
    plotter.plot_line_chart([np.array(data["clip_score"])], "ply_file", "clip_score", dataset_name)
    plotter.plot_line_chart([np.array(data["sharpness"])], "ply_file", "sharpness", dataset_name)

    logger.info(" Done. \n")

    if generate_raw_template:
        logger.info(" Creating a raw benchmark template for the current data.")
        benchmark_runner.generate_raw_evaluation_template(
            files, images, prompts, scores, "benchmark_output", template_file
        )
        logger.info(" Done.")

    elif evaluate_validation:
        logger.info(f" Evaluating validation using reference in data from file: {template_file}")
        benchmark_runner.run_evaluation_benchmark(files, scores, template_file)
        logger.info(" Done.")

    else:
        logger.warning(
            " Please enable one of the options in the config file: "
            "'generate_raw_template' or 'evaluate_validation'. Nothing to do."
        )

    if save_previews:
        logger.info("Saving preview images.")
        benchmark_runner.save_rendered_images(
            preview_images, "benchmark_output/previews", file_names, save_previews=True
        )
        logger.info("Done.")


if __name__ == "__main__":
    main()
