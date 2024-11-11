import os

import cv2
import tqdm
import rembg
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

from .GaussianSplattingModel import GaussianModel
from .rendering.gs_camera import OrbitCamera
from .rendering.gs_renderer import GaussianRenderer


class GaussianProcessor:
    def __init__(self, opt: OmegaConf, prompt: str):
        self._opt = opt
        self._W = opt.W
        self._H = opt.H
        self._fixed_cam = None

        self._mode = "image"
        self._seed = "random"
        self._last_seed = 0

        self._buffer_image = np.ones((self._W, self._H, 3), dtype=np.float32)
        self._need_update = True

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = GaussianModel(opt.sh_degree)
        self._render1 = GaussianRenderer()

        self._optimizer = None
        self._bg_remover = None
        self._guidance_sd = None
        self._guidance_zero123 = None

        self._enable_sd = False
        self._enable_zero123 = False

        # input image
        self._input_image = None
        self._input_mask = None
        self._input_img_torch = None
        self._input_mask_torch = None
        self._overlay_input_img = False
        self._overlay_input_img_ratio = 0.5
        self._step = 0
        self._train_steps = 1   # steps per rendering loop

        # load input data from cmdline
        if self._opt.input is not None:
            self._load_image_prompt(self._opt.input)

        # override prompt from cmdline
        if self._opt.prompt is not None and prompt == "":
            self._prompt = self._opt.prompt
        else:
            self._prompt = prompt

        if self._opt.negative_prompt is not None:
            self._negative_prompt = self._opt.negative_prompt

        # initialize gaussians to a blob
        self._model.initialize(num_pts=self._opt.num_pts)

    def _set_torch_seed(self):
        try:
            seed = int(self._seed)
        except Exception:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        self._last_seed = seed

    def _prepare_training_model(self, models: list[str]):
        self._step = 0

        # setting up training
        self._model.training_setup(self._opt)

        # do not do progressive sh-level
        self._model.active_sh_degree = self._model.max_sh_degree
        self._optimizer = self._model.optimizer

        self._fixed_cam = OrbitCamera(
            self._opt.ref_size,
            self._opt.ref_size,
            self._opt.fovy
        )

        if self._opt.mvdream or self._opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            self._fixed_cam.compute_transform_orbit(self._opt.elevation, 90, self._opt.radius)
        else:
            self._fixed_cam.compute_transform_orbit(self._opt.elevation, 0, self._opt.radius)

        self._enable_sd = self._opt.lambda_sd > 0 and self._prompt != ""
        self._enable_zero123 = self._opt.lambda_zero123 > 0 and self._input_image is not None

        if self._guidance_sd is None and self._enable_sd:
            self._guidance_sd = models[0]

        if self._guidance_zero123 is None and self._enable_zero123:
            if len(models) > 1:
                self._guidance_zero123 = models[1]
            else:
                self._guidance_zero123 = models[0]

        # input image
        if self._input_image is not None:
            self._input_img_torch = (
                torch.from_numpy(self._input_image).permute(2, 0, 1).unsqueeze(0).to(self._device)
            )

            self._input_img_torch = F.interpolate(
                self._input_img_torch,
                (self._opt.ref_size, self._opt.ref_size),
                mode="bilinear",
                align_corners=False,
            )

            self._input_mask_torch = (
                torch.from_numpy(self._input_mask).permute(2, 0, 1).unsqueeze(0).to(self._device)
            )

            self._input_mask_torch = F.interpolate(
                self._input_mask_torch,
                (self._opt.ref_size, self._opt.ref_size),
                mode="bilinear",
                align_corners=False,
            )

        # prepare embeddings
        with torch.no_grad():
            if self._enable_sd:
                if self._opt.imagedream:
                    self._guidance_sd.get_image_text_embeds(
                        self._input_img_torch,
                        [self._prompt],
                        [self._negative_prompt],
                    )
                else:
                    self._guidance_sd.get_text_embeds([self._prompt], [self._negative_prompt])

            if self._enable_zero123:
                self._guidance_zero123.get_img_embeds(self._input_img_torch)

    def get_gs_data(self, return_rgb_colors: bool = False):
        means3D = self._model.get_xyz
        rotations = self._model.get_rotation
        scales = self._model.get_scaling
        opacity = self._model.get_opacity.squeeze(1)
        if return_rgb_colors:
            features = self._model.get_features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
            SH_C0 = 0.28209479177387814
            rgbs = (0.5 + SH_C0 * features)
        else:
            rgbs = self._model.get_features

        gs_data = [means3D, rotations, scales, opacity, rgbs]
        return gs_data

    def _train_step(self, i: int):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self._train_steps):
            self._step += 1
            step_ratio = min(1, self._step / self._opt.iters)

            # update lr
            self._model.update_learning_rate(self._step)
            loss = 0

            # known view
            if self._input_img_torch is not None and not self._opt.imagedream:
                cur_cam = self._fixed_cam

                # getting gaussian data
                gs_data = self.get_gs_data()

                # rendering images
                image, mask, _, _ = self._render1.render(
                    cur_cam.world_to_camera_transform.unsqueeze(0),
                    cur_cam.intrinsics.unsqueeze(0),
                    (cur_cam.image_width, cur_cam.image_height),
                    cur_cam.z_near,
                    cur_cam.z_far,
                    gs_data,
                    sh_degree=self._model.active_sh_degree
                )

                # rgb loss
                image = image.pemute(0, 3, 1, 2)

                loss = loss + 10000 * (step_ratio if self._opt.warmup_rgb_loss else 1) * F.mse_loss(
                    image, self._input_img_torch
                )

                # mask loss
                mask = mask.permute(0, 3, 1, 2)
                loss = loss + 1000 * (step_ratio if self._opt.warmup_rgb_loss else 1) * F.mse_loss(
                    mask, self._input_mask_torch
                )

            # novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []

            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(
                min(self._opt.min_ver, self._opt.min_ver - self._opt.elevation),
                -80 - self._opt.elevation,
            )

            max_ver = min(
                max(self._opt.max_ver, self._opt.max_ver - self._opt.elevation),
                80 - self._opt.elevation,
            )

            cur_cam = OrbitCamera(render_resolution, render_resolution, self._opt.fovy)
            cur_cam_i = OrbitCamera(render_resolution, render_resolution, self._opt.fovy)

            for _ in range(self._opt.batch_size):
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                cur_cam.compute_transform_orbit(
                    self._opt.elevation + ver,
                    hor,
                    self._opt.radius + radius
                )

                pose = cur_cam.camera_to_world_tr
                poses.append(pose)

                bg_color = torch.tensor(
                    [1, 1, 1] if np.random.rand() > self._opt.invert_bg_prob else [0, 0, 0],
                    dtype=torch.float32,
                    device=self._device,
                )

                # loading gaussian data
                gs_data = self.get_gs_data()

                # rendering images
                image, _, _, meta = self._render1.render(
                    cur_cam.world_to_camera_transform.unsqueeze(0),
                    cur_cam.intrinsics.unsqueeze(0),
                    (cur_cam.image_width, cur_cam.image_height),
                    cur_cam.z_near,
                    cur_cam.z_far,
                    gs_data,
                    bg_color=bg_color,
                    sh_degree=self._model.active_sh_degree
                )

                image = image.permute(0, 3, 1, 2)
                images.append(image)

                # enable mvdream training
                if self._opt.mvdream or self._opt.imagedream:
                    for view_i in range(1, 4):
                        cur_cam_i.compute_transform_orbit(
                            self._opt.elevation + ver,
                            hor + 90 * view_i,
                            self._opt.radius + radius
                        )
                        pose_i = cur_cam_i.camera_to_world_tr
                        poses.append(pose_i)

                        # loading gaussian data
                        gs_data_i = self.get_gs_data(return_rgb_colors=True)

                        # rendering images
                        image_i, _, _, _ = self._render1.render(
                            cur_cam_i.world_to_camera_transform.unsqueeze(0),
                            cur_cam_i.intrinsics.unsqueeze(0),
                            (cur_cam_i.image_width, cur_cam_i.image_height),
                            cur_cam_i.z_near,
                            cur_cam_i.z_far,
                            gs_data_i,
                            bg_color=bg_color
                        )

                        image_i = image_i.permute(0, 3, 1, 2)
                        images.append(image_i)

                images = torch.cat(images, dim=0)
                poses = torch.stack(poses, dim=0).to(self._device)

                # guidance loss
                if self._enable_sd:
                    if self._opt.mvdream or self._opt.imagedream:
                        loss = loss + self._opt.lambda_sd * self._guidance_sd.train_step(
                            images,
                            poses,
                            step_ratio=step_ratio if self._opt.anneal_timestep else None,
                        )
                    else:
                        loss = loss + self._opt.lambda_sd * self._guidance_sd.train_step(
                            images,
                            step_ratio=step_ratio if self._opt.anneal_timestep else None,
                        )

                if self._enable_zero123:
                    loss = loss + self._opt.lambda_zero123 * self._guidance_zero123.train_step(
                        images,
                        vers,
                        hors,
                        radii,
                        step_ratio=step_ratio if self._opt.anneal_timestep else None,
                        default_elevation=self._opt.elevation,
                    )

                # optimize step
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

                # densify and prune
                if ((self._step >= self._opt.density_start_iter)
                        and (self._step <= self._opt.density_end_iter)):

                    viewspace_point_tensor, visibility_filter, radii = (
                        meta["means2d"],
                        meta["radii"] > 0,
                        meta["radii"],
                    )

                    self._model.max_radii2D[visibility_filter.squeeze(0)] = torch.max(
                        self._model.max_radii2D[visibility_filter.squeeze(0)],
                        radii[visibility_filter],
                    )
                    self._model.add_densification_stats(viewspace_point_tensor, visibility_filter.squeeze(0))

                    if self._step % self._opt.densification_interval == 0:
                        self._model.densify_and_prune(
                            self._opt.densify_grad_threshold,
                            min_opacity=0.01,
                            extent=4,
                            max_screen_size=1,
                        )

                    if self._step % self._opt.opacity_reset_interval == 0:
                        self._model.reset_opacity()

        ender.record()
        torch.cuda.synchronize()

        self._need_update = True

    @torch.no_grad()
    def _test_step(self):
        # ignore if no need to update
        if not self._need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        gs_data = self.get_gs_data(return_rgb_colors=True)

        # should update image
        if self._need_update:
            # render image
            cur_cam = OrbitCamera(self._W, self._H, self._opt.fovy)
            cur_cam.compute_transform_orbit(0, 0, self._opt.radius)

            image, alpha, depth, _ = self._render1.render(
                cur_cam.world_to_camera_transform.unsqueeze(0),
                cur_cam.intrinsics.unsqueeze(0),
                (cur_cam.image_width, cur_cam.image_height),
                cur_cam.z_near,
                cur_cam.z_far,
                gs_data)

            if self._mode == "alpha":
                buffer_image = alpha.repeat(3, 1, 1)

            elif self._mode == "depth":
                buffer_image = depth.repeat(3, 1, 1)
                buffer_image = (buffer_image - buffer_image.min()) / (
                        buffer_image.max() - buffer_image.min() + 1e-20
                )
            else:
                buffer_image = image.permute(0, 3, 1, 2)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self._H, self._W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self._buffer_image = (
                buffer_image.permute(1, 2, 0).contiguous().clamp(0, 1).contiguous().detach().cpu().numpy()
            )

            # display input_image
            if self._overlay_input_img and self._input_image is not None:
                self._buffer_image = (
                        self._buffer_image * (1 - self._overlay_input_img_ratio)
                        + self._input_image * self._overlay_input_img_ratio
                )

            self._need_update = False

        ender.record()
        torch.cuda.synchronize()

    def _load_image_prompt(self, file: str):
        # load image
        logger.info(f"Load image from {file}...")

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self._bg_remover is None:
                self._bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self._bg_remover)

        img = cv2.resize(img, (self._W, self._H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self._input_mask = img[..., 3:]

        # white bg
        self._input_image = img[..., :3] * self._input_mask + (1 - self._input_mask)

        # bgr to rgb
        self._input_image = self._input_image[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            logger.info(f"Load prompt from {file_prompt}...")
            with open(file_prompt, "r") as f:
                self._prompt = f.read().strip()

    def train(self, models: list, iters=500):
        if iters > 0:
            self._prepare_training_model(models)
            for i in tqdm.trange(iters):
                self._train_step(i)

            # do a last prune
            self._model.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        return self.get_gs_model_data()

    def get_gs_model_data(self):
        return self._model.get_model_data()

    def get_gs_model(self):
        return self._model
