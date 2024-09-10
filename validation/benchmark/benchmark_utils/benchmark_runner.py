import shutil
from pathlib import Path
from time import time

import pandas as pd
import torch
import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from validation.validation_lib.io.hdf5 import HDF5Loader
from validation.validation_lib.io.ply import PlyLoader
from validation.validation_lib.memory import enough_gpu_mem_available
from validation.validation_lib.rendering.rendering_pipeline import RenderingPipeline
from validation.validation_lib.validation.validation_pipeline import ValidationPipeline


class BenchmarkRunner:
    def __init__(self, views: int, verbose: bool = True, debug: bool = True):
        self._gs_renderer = RenderingPipeline(views)
        self._validator = ValidationPipeline(verbose=verbose, debug=debug)
        self._validator.preload_model()
        self._hdf5_loader = HDF5Loader()
        self._ply_loader = PlyLoader()

        self._high_quality = [0.8, 1.0]
        self._medium_quality = [0.6, 0.7999]
        self._low_quality = [0.0, 0.5999]

    def validate(
        self,
        data_dict: dict,
        img_width: int,
        img_height: int,
        prompt: str,
        cam_rad: float,
        data_ver: int,
    ) -> tuple[list[torch.Tensor], float, float]:
        """Function for validating the input data

        Parameters
        ----------
        data_dict: unpacked data that will be rendered
        img_width: the width of the rendering images
        img_height: the height of the rendering images
        prompt : an input prompt
        cam_rad: the radius of the camera orbit
        data_ver: version of the input data format: 0 - corresponds to dream gaussian
                                                    1 - corresponds to new data format (default)

        Returns
        -------
        images: a set of rendered images,
        score: computed float score
        dt: execution time
        """
        logger.info(f" Input prompt: {prompt}")

        t1 = time()
        images = self._gs_renderer.render_gaussian_splatting_views(
            data_dict, img_width, img_height, cam_rad, data_ver=data_ver
        )
        t2 = time()

        score = self._validator.validate(images, prompt)
        t3 = time()

        dt = t3 - t1

        logger.info(f" Rendering took: {t2 - t1} sec")
        logger.info(f" Validation took: {t3 - t2} sec")
        logger.info(f" Validation took [total]: {dt} sec\n\n")

        images_cpu = [img.detach().cpu() for img in images]
        del images

        return images_cpu, score, dt

    def run_validation_benchmark(
        self, config_data: dict, prompts: list[str], files: list[Path], return_images: bool = False
    ) -> tuple[list[list[torch.Tensor]], list[list[float]], list[list[float]], list[str]]:
        """
        Function for running validation over provided dataset

        Parameters
        ----------
        config_data: data loaded from the benchmark_config.yml to the dictionary
        prompts: a list with prompts
        files: list of paths to the ply files
        return_images: enable/disable returning of the images

        Returns
        -------
        rendered_images: list with rendered images if it was requested
        scores: list of scores per model
        timings: list of timings per validation
        file_names: list of ply file names
        """
        rendered_images = []
        scores = []
        timings = []
        file_names = []
        for j, data_file in enumerate(files):
            # loading file
            data_path = Path(data_file)
            file_name = data_path.stem
            file_ext = data_path.suffix
            file_path = data_path.parent
            logger.info(f" File Path: {file_name}")

            if file_ext == ".h5":
                data_dict = self._hdf5_loader.from_file(file_name, file_path)
            elif file_ext == ".ply":
                data_dict = self._ply_loader.from_file(file_name, file_path)
            else:
                continue

            # prompts preparing
            if len(prompts) == 1:
                prompt = prompts[0]
            elif len(prompts) != len(files):
                raise ValueError("The number of prompts must either be 1 or equal to the number of files.")
            else:
                prompt = prompts[j]

            if not enough_gpu_mem_available(data_dict):
                raise RuntimeError("Not enough GPU memory to process the current data.")

            images, score, dt = self.validate(
                data_dict,
                int(config_data["img_width"]),
                int(config_data["img_height"]),
                prompt,
                config_data["cam_rad"],
                config_data["data_ver"],
            )

            if return_images:
                rendered_images.append(images)

            scores.append([score])
            timings.append([dt])
            file_names.append(file_name)
        return rendered_images, scores, timings, file_names

    def run_evaluation_benchmark(self, files: list[Path], scores: list[list[float]], template_path: str) -> None:
        """
        Function for evaluating the quality of the validation

        Parameters
        ----------
        files: list of paths to the ply files
        scores: list of validation scores per input ply file
        template_path: path to the template with reference data against which current validator will be evaluated
        """
        logger.info(f" Evaluating validation results against reference data from file: {template_path}")
        if len(scores) != len(files):
            raise ValueError("The number of scores must be equal to the number of files.")

        data_frame = pd.read_csv(template_path)
        assigned_tags = []
        positive_detection = 0
        false_detection = 0
        for i in tqdm.trange(len(files)):
            tag = self._assign_tag(scores[i][0])
            assigned_tags.append(tag)

            if (
                ((tag == "hq") and (data_frame.loc[i].at["quality"] == "hq"))
                or ((tag == "mq") and (data_frame.loc[i].at["quality"] == "mq"))
                or ((tag == "lq") and (data_frame.loc[i].at["quality"] == "lq"))
            ):
                positive_detection += 1
            else:
                false_detection += 1

        accuracy = accuracy_score(data_frame["quality"], assigned_tags)
        precision = precision_score(data_frame["quality"], assigned_tags, labels=["hq", "mq", "lq"], average=None)
        recall = recall_score(data_frame["quality"], assigned_tags, labels=["hq", "mq", "lq"], average=None)
        f1score = f1_score(data_frame["quality"], assigned_tags, labels=["hq", "mq", "lq"], average=None)

        logger.info("Results:")
        logger.info(f" Positive Detection: {positive_detection} / {len(files)}")
        logger.info(f" False Detection: {false_detection} / {len(files)}")
        logger.info(f" Accuracy: {round(accuracy, 3)}")
        logger.info(
            f" Precision: < hq: {round(precision[0], 3)} > ; "
            f"< mq: {round(precision[1], 3)} > ; "
            f"< lq: {round(precision[2], 3)} >"
        )
        logger.info(
            f" Recall: < hq: {round(recall[0], 3)} > ; "
            f"< mq: {round(recall[1], 3)} > ; "
            f"< lq: {round(recall[2], 3)} >"
        )
        logger.info(
            f" F1 Score: < hq: {round(f1score[0], 3)} > ; "
            f"< mq: {round(f1score[1], 3)} > ; "
            f"< lq: {round(f1score[2], 3)} >"
        )

    def save_rendered_images(self, images: list[torch.Tensor], folder_name: Path | str, gs_file_name: str) -> None:
        """
        Function for saving rendered images

        Parameters
        ----------
        images: list of images per model
        folder_name: name for the folder where images will be saved
        gs_file_name: name of the ply file for which input set of images was rendered
        """
        path = Path.cwd() / folder_name
        path.mkdir(exist_ok=True)
        path = path / gs_file_name
        path.mkdir()

        self._gs_renderer.save_rendered_images(images, gs_file_name, path)

    def generate_raw_evaluation_template(
        self,
        files: list[Path],
        images: list[list[torch.Tensor]],
        prompts: list[str],
        scores: list[list[float]],
        output_folder: str,
        template_name: str = "quality_assessment.csv",
    ) -> None:
        """
        Function for generating raw evaluation template

        Parameters
        ----------
        files: list of paths to the ply files
        images: list with rendered images
        prompts: list with prompts used for generating input ply files
        scores: list with generated scores
        output_folder: path to the folder where to store sorted ply files
        template_name: the name of the template file that will be saved
        """
        output_folder_path = Path.cwd() / output_folder
        output_folder_path.mkdir(parents=True, exist_ok=True)

        if output_folder_path.exists():
            shutil.rmtree(output_folder_path)
            output_folder_path.mkdir()

        output_imgs_highq_folder_path = output_folder_path / "high_quality" / "images"
        output_imgs_highq_folder_path.mkdir(parents=True, exist_ok=True)

        output_plys_highq_folder_path = output_folder_path / "high_quality" / "plys"
        output_plys_highq_folder_path.mkdir(parents=True, exist_ok=True)

        output_imgs_medq_folder_path = output_folder_path / "medium_quality" / "images"
        output_imgs_medq_folder_path.mkdir(parents=True, exist_ok=True)

        output_plys_medq_folder_path = output_folder_path / "medium_quality" / "plys"
        output_plys_medq_folder_path.mkdir(parents=True, exist_ok=True)

        output_imgs_lowq_folder_path = output_folder_path / "low_quality" / "images"
        output_imgs_lowq_folder_path.mkdir(parents=True, exist_ok=True)

        output_plys_lowq_folder_path = output_folder_path / "low_quality" / "plys"
        output_plys_lowq_folder_path.mkdir(parents=True, exist_ok=True)

        file_names = []
        file_tags = []
        logger.info("Sorting input data into three categories: high, medium, low. Rendering & saving images.")
        for i in tqdm.trange(len(scores)):
            file_path = files[i]
            file_name = file_path.stem
            file_ext = file_path.suffix
            file_names.append(file_name)

            tag = self._assign_tag(scores[i][0])
            file_tags.append(tag)

            if tag == "hq":
                file_destination = output_plys_highq_folder_path / (file_name + file_ext)
                shutil.copy2(file_path, file_destination)
                if images:
                    self.save_rendered_images(images[i], output_imgs_highq_folder_path, file_name)
            elif tag == "mq":
                file_destination = output_plys_medq_folder_path / (file_name + file_ext)
                shutil.copy2(file_path, file_destination)
                if images:
                    self.save_rendered_images(images[i], output_imgs_medq_folder_path, file_name)
            else:
                file_destination = output_plys_lowq_folder_path / (file_name + file_ext)
                shutil.copy2(file_path, file_destination)
                if images:
                    self.save_rendered_images(images[i], output_imgs_lowq_folder_path, file_name)

        logger.info(f"Creating raw template: {template_name}")
        data = {"files": file_names, "prompt": prompts, "scores": scores, "quality": file_tags}

        df = pd.DataFrame(data)
        df.to_csv(template_name, float_format="%.5f")

    def _assign_tag(self, score: float) -> str:
        """
        Function for assigning one of the three tags per model score:
        hq - high quality
        mq - medium quality
        lq - low quality

        Parameters
        ----------
        score: float number with validation score

        Returns
        -------
        tag: string with assigned tag
        """
        if score >= self._high_quality[0]:
            tag = "hq"
        elif (score >= self._medium_quality[0]) and (score < self._medium_quality[1]):
            tag = "mq"
        else:
            tag = "lq"
        return tag
