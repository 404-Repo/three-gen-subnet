import argparse
import gc
import io
import json
import traceback
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pybase64
import torch
from engine.data_structures import TimeStat, ValidationRequest, ValidationResponse
from engine.io.ply.loader import PlyLoader
from engine.rendering.renderer import Renderer
from engine.validation_engine import ValidationEngine
from execution_tree import ExecutionTree
from pydantic import BaseModel
from pyinstrument import Profiler
from pyinstrument.renderers import JSONRenderer
from rich import print
from server.pipeline import decode_and_validate_txt

from benchmark.execution_tree_visualizer import ExecutionTreeVisualizer


class BenchmarkValidationResult(BaseModel):
    """
    Statistics of the repeated validation profiling.
    """

    try_cnt: int
    validation_results: list[tuple[ValidationResponse, TimeStat]]
    avg_load_time: float
    avg_image_render_time: float
    avg_validation_time: float
    avg_total_time: float
    execution_tree: dict[str, Any]

    def get_stat_dict(self) -> dict[str, Any]:
        return {
            "avg_load_time": self.avg_load_time,
            "avg_image_render_time": self.avg_image_render_time,
            "avg_validation_time": self.avg_validation_time,
            "avg_total_time": self.avg_total_time,
        }


class BenchmarkValidation:
    """
    Repeatedly validates the same .spz file and returns profiling statistics.
    """

    BENCHMARK_METHOD_NAME: str = "benchmark_validation"  # Change this name if the method name is changed
    RESULT_DIR: Path = Path(__file__).resolve().parent / "results"
    _MAX_PRINT_LEVEL: int = 6  # Max level to print execution tree.
    _SAMPLE_SPZ_MODEL_FILE: Path = Path(__file__).resolve().parent / "data/robocop with hammer in one hand.spz"

    @staticmethod
    def run(try_cnt: int) -> None:
        """
        Runs validation try_cnt times, performs benchmarking
        and saves statistics results to the files in result directory.
        """
        if try_cnt <= 0:
            raise ValueError(f"{try_cnt} must be > 0.")

        print(f"[green]Running benchmark validation with {try_cnt} iterations...[/green]")
        print("[green]1. Preparing validator... (1 from 5)[/green]")
        validator = ValidationEngine()
        validator.load_pipelines()
        renderer = Renderer()
        ply_data_loader = PlyLoader()
        gc.collect()
        torch.cuda.empty_cache()

        print("[green]2. Preparing request data... (2 from 5)[/green]")
        with BenchmarkValidation._SAMPLE_SPZ_MODEL_FILE.open("rb") as file:
            data = pybase64.b64encode_as_string(file.read())
            prompt = BenchmarkValidation._SAMPLE_SPZ_MODEL_FILE.stem
        request_data = ValidationRequest(
            prompt=prompt,
            data=data,
            compression=2,
        )
        print("[green]3. Running validation... (3 from 5)[/green]")
        profiler = Profiler()
        with profiler:
            validation_results = BenchmarkValidation.benchmark_validation(
                request=request_data,
                validator=validator,
                ply_data_loader=ply_data_loader,
                renderer=renderer,
                try_cnt=try_cnt,
            )

        print("[green]4. Analyzing benchmark results... (4 from 5)[/green]")
        execution_tree_str = profiler.output(renderer=JSONRenderer())
        execution_tree_dict = json.loads(execution_tree_str)
        execution_tree = ExecutionTree.from_pyinstrument_output(
            execution_tree_dict, BenchmarkValidation.BENCHMARK_METHOD_NAME, try_cnt
        )
        result = BenchmarkValidation._get_result(
            validation_results=validation_results, execution_tree=execution_tree_dict
        )

        print("[green]5. Saving results... (5 from 5)[/green]")
        BenchmarkValidation._save_result(result=result, execution_tree=execution_tree)
        BenchmarkValidation._print_stat(result=result)
        ExecutionTreeVisualizer.print(root=execution_tree.root, max_level=BenchmarkValidation._MAX_PRINT_LEVEL)
        print(f"[green]\n\nYou can find results in {BenchmarkValidation.RESULT_DIR.absolute()}\n\n[/green]")

    @staticmethod
    def benchmark_validation(
        *,
        validator: ValidationEngine,
        ply_data_loader: PlyLoader,
        renderer: Renderer,
        request: ValidationRequest,
        try_cnt: int,
    ) -> list[tuple[ValidationResponse, TimeStat]]:
        """
        Does try_cnt validation runs and returns profiling statistics.
        Method also saves statistics results to the files in temp directory as zip archive benchmark_validation.zip.
        """
        results: list[tuple[ValidationResponse, TimeStat]] = []
        for _ in range(try_cnt):
            validation_result = decode_and_validate_txt(
                request=request,
                validator=validator,
                renderer=renderer,
                ply_data_loader=ply_data_loader,
            )
            results.append(validation_result)

        return results

    @staticmethod
    def _get_result(
        *, validation_results: list[tuple[ValidationResponse, TimeStat]], execution_tree: dict[str, Any]
    ) -> BenchmarkValidationResult:
        """
        Calculates statistics and returns benchmark validation result.
        """
        time_stats: list[TimeStat] = [time_stat for response, time_stat in validation_results]

        avg_load_time = sum([stat.loading_data_time for stat in time_stats]) / len(time_stats)
        avg_image_render_time = sum([stat.image_rendering_time for stat in time_stats]) / len(time_stats)
        avg_validation_time = sum([stat.validation_time for stat in time_stats]) / len(time_stats)
        avg_total_time = sum([stat.total_time for stat in time_stats]) / len(time_stats)
        return BenchmarkValidationResult(
            validation_results=validation_results,
            execution_tree=execution_tree,
            try_cnt=len(validation_results),
            avg_load_time=avg_load_time,
            avg_image_render_time=avg_image_render_time,
            avg_validation_time=avg_validation_time,
            avg_total_time=avg_total_time,
        )

    @staticmethod
    def _save_result(*, result: BenchmarkValidationResult, execution_tree: ExecutionTree) -> None:
        """
        Saves statistics results to the files in an archive in temp directory:
        - time stats for each iteration in a csv file;
        - aggregated statistics in a csv file;
        - execution tree in a json file;
        - each method execution time in a csv file;
        """
        BenchmarkValidation.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Save time stats for each iteration.
            time_stats = [time_stat.model_dump() for response, time_stat in result.validation_results]
            time_stats_df = pd.DataFrame(time_stats)
            iteration_csv_buffer = io.BytesIO()
            time_stats_df.to_csv(iteration_csv_buffer, index=False)
            iteration_csv_buffer.seek(0)
            zip_file.writestr(
                f"{BenchmarkValidation.BENCHMARK_METHOD_NAME}_iteration_stats.csv",
                iteration_csv_buffer.getvalue(),
            )

            # Save aggregated statistics.
            aggregated_stat = result.get_stat_dict()
            aggregated_stat_csv_buffer = io.StringIO()
            aggregated_stat_df = pd.DataFrame([aggregated_stat])
            aggregated_stat_df.to_csv(aggregated_stat_csv_buffer, index=False)
            aggregated_stat_csv_buffer.seek(0)
            zip_file.writestr(
                f"{BenchmarkValidation.BENCHMARK_METHOD_NAME}_aggregated_stat.csv",
                aggregated_stat_csv_buffer.getvalue(),
            )

            # Save execution tree.
            execution_tree_json_str = json.dumps(result.execution_tree)
            execution_tree_json_buffer = execution_tree_json_str.encode("utf-8")
            zip_file.writestr(
                f"{BenchmarkValidation.BENCHMARK_METHOD_NAME}_execution_tree.json",
                execution_tree_json_buffer,
            )

            # Save execution tree methods in .csv table with times.
            execution_tree_df = execution_tree.to_df()
            execution_tree_csv_buffer = io.BytesIO()
            execution_tree_df.to_csv(execution_tree_csv_buffer, index=False)
            execution_tree_csv_buffer.seek(0)
            zip_file.writestr(
                f"{BenchmarkValidation.BENCHMARK_METHOD_NAME}_function_times.csv",
                execution_tree_csv_buffer.getvalue(),
            )
        zip_file_path = BenchmarkValidation.RESULT_DIR / f"{BenchmarkValidation.BENCHMARK_METHOD_NAME}.zip"
        zip_buffer.seek(0)
        with zip_file_path.open("wb") as f:
            f.write(zip_buffer.getvalue())

    @staticmethod
    def _print_stat(*, result: BenchmarkValidationResult) -> None:
        """
        Prints aggregated statistics in a readable format.
        """
        print("[green]\n\nBENCHMARK STAT RESULT\n[/green]")
        print(f"[green]{'-'*100}[/green]")
        for key, value in result.get_stat_dict().items():
            if isinstance(value, float):
                print(f"{key:<30}: [#AAAA00]{value:.3f}s[/#AAAA00]")
            else:
                print(f"{key:<30}: [#AAAA00]{value}[/#AAAA00]")


if __name__ == "__main__":
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--try-cnt", type=int, required=True)
        args = arg_parser.parse_args()
        try_cnt = args.try_cnt

        BenchmarkValidation.run(try_cnt=try_cnt)

    except Exception as e:
        print(f"[red]Error occurred in benchmark validation: {e}\n [/red]")
        traceback.print_exc()
        exit(1)
