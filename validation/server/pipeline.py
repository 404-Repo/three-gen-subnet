import base64
import io
import time

import numpy as np
import pybase64
import pyspz
import torch
from engine.data_structures import GaussianSplattingData, RenderRequest, TimeStat, ValidationRequest, ValidationResponse
from engine.io.ply import PlyLoader
from engine.rendering.renderer import Renderer
from engine.utils.gs_data_checker_utils import add_white_background, is_input_data_valid
from engine.validation_engine import ValidationEngine
from fastapi import HTTPException
from loguru import logger
from PIL import Image


VIEWS_NUMBER = 16
THETA_ANGLES = np.linspace(0, 360, num=VIEWS_NUMBER)
PHI_ANGLES = np.full_like(THETA_ANGLES, -15.0)
FRONT_VIEW_IDX = 1
GRID_VIEW_INDICES = [1, 5, 9, 13]
IMG_WIDTH = 518
IMG_HEIGHT = 518
GRID_VIEW_GAP = 5


def decode_and_validate_txt(
    request: ValidationRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    validator: ValidationEngine,
) -> tuple[ValidationResponse, TimeStat]:
    """
    Validates text-to-3D generation results by scoring how well the 3D asset matches the input prompt.
    """
    if request.prompt is None:
        return ValidationResponse(score=0.0), TimeStat()

    t1 = time.time()
    assets = decode_assets(request)
    gs_data, gs_rendered_images, time_stat = prepare_input_data(
        assets,
        renderer,
        ply_data_loader,
        validator,
        views_number=VIEWS_NUMBER,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        theta_angles=THETA_ANGLES,
        phi_angles=PHI_ANGLES,
    )

    if gs_data is None:
        return ValidationResponse(score=0.0), time_stat

    t2 = time.time()
    validation_result = validate_txt(request.prompt, gs_rendered_images, validator)
    time_stat.validation_time = time.time() - t2

    response = finalize_results(
        request,
        validation_result,
        gs_rendered_images,
    )
    time_stat.total_time = time.time() - t1

    return response, time_stat


def decode_and_validate_img(
    request: ValidationRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    validator: ValidationEngine,
) -> tuple[ValidationResponse, TimeStat]:
    """
    Validates 2D-to-3D generation results by scoring how well the 3D asset matches the input prompt.
    """
    if request.prompt_image is None:
        return ValidationResponse(score=0.0), TimeStat()

    t1 = time.time()
    assets = decode_assets(request)
    gs_data, gs_rendered_images, time_stat = prepare_input_data(
        assets,
        renderer,
        ply_data_loader,
        validator,
        views_number=VIEWS_NUMBER,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        theta_angles=THETA_ANGLES,
        phi_angles=PHI_ANGLES,
    )

    if gs_data is None:
        return ValidationResponse(score=0.0), time_stat

    t2 = time.time()
    image_data = pybase64.b64decode(request.prompt_image)
    prompt_image = Image.open(io.BytesIO(image_data))
    if len(prompt_image.getbands()) == 4:
        prompt_image = add_white_background(prompt_image)
    prompt_image = prompt_image.convert("RGB")
    np_img = np.asarray(prompt_image)
    torch_prompt_image = torch.from_numpy(np_img)
    validation_results = validate_img(torch_prompt_image, gs_rendered_images, validator)
    time_stat.validation_time = time.time() - t2

    response = finalize_results(
        request,
        validation_results,
        gs_rendered_images,
    )
    time_stat.total_time = time.time() - t1

    return response, time_stat


def decode_and_render_grid(
    request: RenderRequest,
    ply_data_loader: PlyLoader,
    renderer: Renderer,
    validator: ValidationEngine,
) -> io.BytesIO:
    assets = decode_assets(request)
    gs_data, gs_rendered_images, time_stat = prepare_input_data(
        assets,
        renderer,
        ply_data_loader,
        validator,
        views_number=4,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        theta_angles=THETA_ANGLES[GRID_VIEW_INDICES],
        phi_angles=PHI_ANGLES[GRID_VIEW_INDICES],
    )

    if not gs_data:
        raise RuntimeError("Invalid splats data")

    preview = combine_images4(images=gs_rendered_images, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, gap=GRID_VIEW_GAP)
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def decode_assets(request: ValidationRequest | RenderRequest) -> bytes:
    if request.compression not in (0, 2):
        raise HTTPException(status_code=400, detail=f"Unsupported compression: {request.compression}")

    t1 = time.time()
    assets = pybase64.b64decode(request.data, validate=True)
    t2 = time.time()
    logger.info(
        f"Assets decoded. Size: {len(request.data)} -> {len(assets)}. "
        f"Time taken: {t2 - t1:.4f} sec. Prompt: {request.prompt}."
    )

    if request.compression == 2:  # SPZ compression.
        compressed_size = len(assets)
        assets = pyspz.decompress(assets, include_normals=False)
        logger.info(
            f"Decompressed. Size: {compressed_size} -> {len(assets)}. "
            f"Time taken: {time.time() - t2:.4f} sec. Prompt: {request.prompt}."
        )

    return assets


def prepare_input_data(
    assets: bytes,
    renderer: Renderer,
    ply_data_loader: PlyLoader,
    validator: ValidationEngine,
    views_number: int,
    img_width: int,
    img_height: int,
    theta_angles: np.ndarray,
    phi_angles: np.ndarray,
) -> tuple[GaussianSplattingData | None, list[torch.Tensor], TimeStat]:
    """Loads PLY asset data and renders validation images."""

    time_stat = TimeStat()

    t1 = time.time()
    pcl_buffer = io.BytesIO(assets)
    gs_data: GaussianSplattingData = ply_data_loader.from_buffer(pcl_buffer)
    t2 = time.time()
    time_stat.loading_data_time = t2 - t1
    logger.info(f"Loading data took: {time_stat.loading_data_time:.4f} sec.")

    if not is_input_data_valid(gs_data):
        return None, [], time_stat

    gs_data_gpu = gs_data.send_to_device(validator.device)
    images = renderer.render_gs(
        gs_data_gpu,
        views_number=views_number,
        img_width=img_width,
        img_height=img_height,
        theta_angles=theta_angles,
        phi_angles=phi_angles,
    )
    t3 = time.time()
    time_stat.image_rendering_time = t3 - t2
    logger.info(f"Image Rendering took: {time_stat.image_rendering_time:.4f} sec.")
    return gs_data_gpu, images, time_stat


def validate_txt(
    prompt: str,
    images: list[torch.Tensor],
    validator: ValidationEngine,
) -> ValidationResponse:
    """
    Validates text-to-3D generation results by scoring how well the 3D asset matches the input prompt.
    """

    t1 = time.time()
    val_res: ValidationResponse = validator.validate_text_to_gs(prompt, images)
    logger.info(f"Score: {val_res.score}. Prompt: {prompt}")
    logger.info(f"Validation took: {time.time() - t1:4f} sec.")
    return val_res


def validate_img(
    prompt_image: torch.Tensor,
    images: list[torch.Tensor],
    validator: ValidationEngine,
) -> ValidationResponse:
    """
    Validates 2D-to-3D generation results by scoring how well the 3D asset matches the input prompt.
    """

    t1 = time.time()
    val_res: ValidationResponse = validator.validate_image_to_gs(prompt_image, images)
    logger.info(f" Score: {val_res.score}. Prompt: provided image.")
    logger.info(f" Validation took: {time.time() - t1} sec.")
    return val_res


def finalize_results(
    validation_request: ValidationRequest,
    validation_results: ValidationResponse,
    rendered_images: list[torch.Tensor],
) -> ValidationResponse:
    """Function that finalizes results"""

    if validation_results.score > validation_request.preview_score_threshold:
        if validation_request.generate_single_preview:
            preview = Image.fromarray(rendered_images[FRONT_VIEW_IDX].detach().cpu().numpy())
            validation_results.preview = save_image_as_encoded_png(preview)
        if validation_request.generate_grid_preview:
            selected_images = [rendered_images[i] for i in GRID_VIEW_INDICES]
            preview = combine_images4(
                images=selected_images, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, gap=GRID_VIEW_GAP
            )
            validation_results.grid_preview = save_image_as_encoded_png(preview)

    return validation_results


def save_image_as_encoded_png(rendered_image: Image.Image) -> str:
    buffer = io.BytesIO()
    rendered_image.save(buffer, format="PNG")
    encoded_png = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_png


def combine_images4(
    images: list[torch.Tensor], img_width: int, img_height: int, gap: int, resize_factor: float = 1.0
) -> Image.Image:
    row_width = img_width * 2 + gap
    column_height = img_height * 2 + gap

    combined_image = Image.new("RGB", (row_width, column_height), color="black")

    pil_images = [Image.fromarray(img.detach().cpu().numpy()) for img in images]

    combined_image.paste(pil_images[0], (0, 0))
    combined_image.paste(pil_images[1], (img_width + gap, 0))
    combined_image.paste(pil_images[2], (0, img_height + gap))
    combined_image.paste(pil_images[3], (img_width + gap, img_height + gap))

    w, h = combined_image.size
    if resize_factor != 1.0:
        combined_image = combined_image.resize(
            (int(w * resize_factor), int(h * resize_factor)), Image.Resampling.LANCZOS
        )

    return combined_image
