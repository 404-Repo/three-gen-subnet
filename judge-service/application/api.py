import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from application.judge import JudgeModel


api_router = APIRouter()
duel_judge_vlm = JudgeModel()
single_thread_executor = ThreadPoolExecutor(max_workers=1)


def get_judge_model() -> JudgeModel:
    return duel_judge_vlm


def get_executor() -> ThreadPoolExecutor:
    return single_thread_executor


class DuelResponse(BaseModel):
    winner: int = Field(..., description="1 if the first image wins, 2 for the second and 0 if it's a draw.")
    explanation: str = Field(..., description="Explanation of the winner selection.")


@api_router.post(
    "/duel/",
    response_model=DuelResponse,
    summary="Compare and score two generations",
    description="Compare and score two generations based on their renders. ",
)
async def duel(
    prompt: Annotated[str, Form()],
    preview1: Annotated[UploadFile, File()],
    preview2: Annotated[UploadFile, File()],
    judge_model: Annotated[JudgeModel, Depends(get_judge_model)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
) -> DuelResponse:
    try:
        img_data = await preview1.read()
        preview1_image = Image.open(io.BytesIO(img_data))

        img_data = await preview2.read()
        preview2_image = Image.open(io.BytesIO(img_data))

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, perform_duel, prompt, preview1_image, preview2_image, judge_model)
    except ValueError as e:
        error_msg = str(e)
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg) from e
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg) from e


def perform_duel(
    prompt: str,
    preview1: Image.Image,
    preview2: Image.Image,
    judge_model: JudgeModel,
) -> DuelResponse:
    logger.info(f"Performing duel for the prompt `{prompt}`")

    start_time = time.time()

    duel_result = judge_model.judge(preview1, preview2, prompt)

    elapsed = time.time() - start_time
    logger.info(
        f"Duel for prompt `{prompt}` finished in {elapsed:.2f}s. Winner: {duel_result.winner}"
    )

    return DuelResponse(
        winner=duel_result.winner,
        explanation=f"{duel_result.prompt_matching}\n\n{duel_result.quality}",
    )
