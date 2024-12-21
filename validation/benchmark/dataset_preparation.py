import os
import re
import shutil
from pathlib import Path

from loguru import logger


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
    logger.info("Loading prompts.")
    if config_data["prompts_file"] != "":
        prompts_dataset_path = Path(config_data["prompts_file"])
        with prompts_dataset_path.open() as file:
            prompts_in = [line.rstrip() for line in file]
    elif len(config_data["prompts"]) > 0:
        prompts_in = config_data["prompts"]
    else:
        raise ValueError(
            "No prompts were given by either 'prompts_file' or 'prompts' fields in the config! "
            "Nothing to be processed."
        )

    return prompts_in


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
    ply_files = list(folder.rglob("*.ply"))
    files = ply_files

    test_match = re.search(r"(\d+)", str(files[0].stem))
    digit_checker = int(test_match.group(1)) if test_match else False

    if not digit_checker:
        sorted_files = sorted(files, key=lambda f: f.name)
    else:
        sorted_files = sorted(files, key=lambda f: int(re.findall(r"\d+", f.name)[0]))

    if len(sorted_files) == 0:
        raise RuntimeWarning(f"No files were found in <{folder_path}>. Nothing to process!")

    return sorted_files


def get_folders_names(path: str) -> list[str]:
    folders_names = sorted(os.listdir(path))
    return folders_names


def save_prompts(file_name: str, prompts_list: list, mode: str = "a") -> None:
    """
    Function for saving the prompts stored in the prompts list

    Parameters
    ----------
    file_name: a string with the name of the file that will be loaded
    prompts_list: a list with strings (generated prompts)
    mode: mode for writing the file: 'a', 'w'

    """
    with Path(file_name).open(mode) as file:
        for p in prompts_list:
            file.write(f"{p}")


def main() -> None:
    dataset_path = Path()
    ply_folder = Path()
    use_folders_names = False
    rename_files = True

    prompts = []

    if use_folders_names:
        folders_names = get_folders_names(dataset_path.as_posix())
        output_dir = dataset_path.parent / "models"
        output_dir.mkdir(exist_ok=True, parents=True)

        for folder_name in folders_names:
            logger.info(f"Processing folder: {folder_name}")
            file_name = folder_name.replace("_", " ") + "\n"

            folder_path = dataset_path / folder_name
            ply_files = list(folder_path.rglob("*.ply"))

            for ply_file in ply_files:
                shutil.copy(ply_file.as_posix(), output_dir)
                prompts.append(file_name)

            if rename_files:
                ply_file_path = (ply_folder / file_name).as_posix() + ".ply"
                Path.rename(output_dir, ply_file_path)

        save_prompts("prompts_dataset.txt", prompts)

    else:
        files = get_all_data_files(ply_folder.as_posix())
        for file in files:
            file_path = Path(file)
            new_ply_file_name = file_path.name.replace(" ", "_")
            new_ply_file_name = new_ply_file_name.replace("-", "_")
            new_ply_path = file_path.parent / new_ply_file_name
            Path.rename(file_path, new_ply_path)

            file_name = file_path.name.replace("_", " ").split(".")[0] + "\n"
            prompts.append(file_name)

    prompts = sorted(prompts)
    prompts_file_path = dataset_path / "prompts_dataset.txt"
    save_prompts(prompts_file_path.as_posix(), prompts)


if __name__ == "__main__":
    main()
