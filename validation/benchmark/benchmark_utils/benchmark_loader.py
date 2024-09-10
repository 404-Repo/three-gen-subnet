import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class BenchmarkLoader:
    """Loader of the input data for the benchmark processing."""

    def __init__(self, config_file: str = "benchmark_config.yml"):
        """
        Parameters
        ----------
        config_file: path to the config file
        """
        self._config_file = config_file

    def load_config(self) -> Any:
        """
        Function that loads the data from config file

        Returns
        -------
        config_data: loaded config data as a dictionary
        """
        logger.info(f"Loading config file: {self._config_file}")
        config_path = Path(self._config_file)
        with config_path.open(encoding="utf-8") as file:
            config_data = yaml.safe_load(file)
        if not config_data:
            raise ValueError("Config file is empty or invalid")
        return config_data

    @staticmethod
    def load_prompts(config_data: dict) -> list[str]:
        """
        Function for loading prompts from the specified txt file

        Parameters
        ----------
        config_data: config file stored as a dictionary with essential fields

        Returns
        -------
        prompts_in: a list with preloaded prompts
        """
        logger.info("Loading prompts.")
        if config_data["prompts_file"] != "":
            with Path.open(config_data["prompts_file"]) as file:
                prompts_in = [line.rstrip() for line in file]
        elif len(config_data["prompts"]) > 0:
            prompts_in = config_data["prompts"]
        else:
            raise ValueError(
                "No prompts were given by either 'prompts_file' or 'prompts' fields in the config! "
                "Nothing to be processed."
            )

        return prompts_in

    @staticmethod
    def get_all_data_files(folder_path: str) -> list[Path]:
        """

        Parameters
        ----------
        folder_path: path to the folder with input data

        Returns
        -------
        files: list with paths of the loaded files
        """
        logger.info(f"Loading files for processing from folder: {folder_path}")
        folder = Path(folder_path)
        h5_files = list(folder.rglob("*.h5"))
        ply_files = list(folder.rglob("*.ply"))
        files = h5_files + ply_files

        def sort_key(name: Path) -> int | float:
            # Use regular expression to find the numeric part
            match = re.search(r"(\d+)", str(name))
            # Convert the numeric part to an integer for sorting
            return int(match.group(1)) if match else float("inf")

        files = sorted(files, key=sort_key)

        if len(files) == 0:
            raise RuntimeWarning(f"No files were found in <{folder_path}>. Nothing to process!")

        return files
