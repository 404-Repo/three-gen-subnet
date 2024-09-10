from benchmark_utils.benchmark_loader import BenchmarkLoader
from benchmark_utils.benchmark_runner import BenchmarkRunner
from loguru import logger


def main() -> None:
    benchmark_loader = BenchmarkLoader()

    # loading config file
    config_data = benchmark_loader.load_config()
    template_file = config_data["template_path"]
    verbose_output = config_data["verbose_output"]
    debug_output = config_data["debug_output"]
    save_images = config_data["save_images"]
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

    benchmark_runner = BenchmarkRunner(config_data["views"], verbose=verbose_output, debug=debug_output)
    logger.info(f" Validating input data: {len(files)} files.")
    images, scores, _, file_names = benchmark_runner.run_validation_benchmark(config_data, prompts, files, save_images)
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


if __name__ == "__main__":
    main()
