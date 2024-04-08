import numpy as np
import cv2
import tqdm
import rembg
import os

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from .CameraUtils import orbit_camera, OrbitCamera
from .GaussianSplattingModel import Renderer, MiniCam


class GaussianProcessor:
    def __init__(self, opt: OmegaConf, prompt: str = ""):
        self.__opt = opt
        self.__W = opt.W
        self.__H = opt.H
        self.__cam = OrbitCamera(self.__W, self.__H, r=self.__opt.radius, fovy=self.__opt.fovy)
        self.__fixed_cam = None

        self.__mode = "image"
        self.__seed = "random"
        self.__last_seed = 0

        self.__buffer_image = np.ones((self.__W, self.__H, 3), dtype=np.float32)
        self.__need_update = True

        # models
        self.__device = torch.device("cuda")
        self.__optimizer = None
        self.__bg_remover = None

        self.__guidance_sd = None
        self.__guidance_zero123 = None
        self.__enable_sd = False
        self.__enable_zero123 = False

        # renderer
        self.__renderer = Renderer(sh_degree=self.__opt.sh_degree)
        self.__gaussian_scale_factor = 1

        # input image
        self.__input_image = None
        self.__input_mask = None
        self.__input_img_torch = None
        self.__input_mask_torch = None
        self.__overlay_input_img = False
        self.__overlay_input_img_ratio = 0.5
        self.__step = 0
        self.__train_steps = 1  # steps per rendering loop

        # input text
        self.__prompt = ""
        self.__negative_prompt = ""

        # load input data from cmdline
        if self.__opt.input is not None:
            self._load_image_prompt(self.__opt.input)

        # override prompt from cmdline
        if self.__opt.prompt is not None and prompt == "":
            self.__prompt = self.__opt.prompt
        else:
            self.__prompt = prompt

        if self.__opt.negative_prompt is not None:
            self.__negative_prompt = self.__opt.negative_prompt

        # override if provide a checkpoint
        if self.__opt.load is not None:
            self.__renderer.initialize(self.__opt.load)
        else:
            # initialize gaussians to a blob
            self.__renderer.initialize(num_pts=self.__opt.num_pts)

    def _set_torch_seed(self):
        try:
            seed = int(self.__seed)
        except Exception:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.__last_seed = seed

    def _prepare_training_model(self, models: list):
        self.__step = 0

        # setting up training
        self.__renderer.gaussians.training_setup(self.__opt)

        # do not do progressive sh-level
        self.__renderer.gaussians.active_sh_degree = self.__renderer.gaussians.max_sh_degree
        self.__optimizer = self.__renderer.gaussians.optimizer

        # default camera
        if self.__opt.mvdream or self.__opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.__opt.elevation, 90, self.__opt.radius)
        else:
            pose = orbit_camera(self.__opt.elevation, 0, self.__opt.radius)

        self.__fixed_cam = MiniCam(
            pose,
            self.__opt.ref_size,
            self.__opt.ref_size,
            self.__cam.fovy,
            self.__cam.fovx,
            self.__cam.near,
            self.__cam.far,
        )

        self.__enable_sd = self.__opt.lambda_sd > 0 and self.__prompt != ""
        self.__enable_zero123 = self.__opt.lambda_zero123 > 0 and self.__input_image is not None

        if self.__guidance_sd is None and self.__enable_sd:
            self.__guidance_sd = models[0]

        if self.__guidance_zero123 is None and self.__enable_zero123:
            if len(models) > 1:
                self.__guidance_zero123 = models[1]
            else:
                self.__guidance_zero123 = models[0]

        # input image
        if self.__input_image is not None:
            self.__input_img_torch = (
                torch.from_numpy(self.__input_image).permute(2, 0, 1).unsqueeze(0).to(self.__device)
            )
            self.__input_img_torch = F.interpolate(
                self.__input_img_torch,
                (self.__opt.ref_size, self.__opt.ref_size),
                mode="bilinear",
                align_corners=False,
            )
            self.__input_mask_torch = (
                torch.from_numpy(self.__input_mask).permute(2, 0, 1).unsqueeze(0).to(self.__device)
            )
            self.__input_mask_torch = F.interpolate(
                self.__input_mask_torch,
                (self.__opt.ref_size, self.__opt.ref_size),
                mode="bilinear",
                align_corners=False,
            )

        # prepare embeddings
        with torch.no_grad():
            if self.__enable_sd:
                if self.__opt.imagedream:
                    self.__guidance_sd.get_image_text_embeds(
                        self.__input_img_torch,
                        [self.__prompt],
                        [self.__negative_prompt],
                    )
                else:
                    self.__guidance_sd.get_text_embeds([self.__prompt], [self.__negative_prompt])

            if self.__enable_zero123:
                self.__guidance_zero123.get_img_embeds(self.__input_img_torch)

    def _train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.__train_steps):
            self.__step += 1
            step_ratio = min(1, self.__step / self.__opt.iters)

            # update lr
            self.__renderer.gaussians.update_learning_rate(self.__step)
            loss = 0

            # known view
            if self.__input_img_torch is not None and not self.__opt.imagedream:
                cur_cam = self.__fixed_cam
                out = self.__renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * (step_ratio if self.__opt.warmup_rgb_loss else 1) * F.mse_loss(
                    image, self.__input_img_torch
                )

                # mask loss
                mask = out["alpha"].unsqueeze(0)  # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 * (step_ratio if self.__opt.warmup_rgb_loss else 1) * F.mse_loss(
                    mask, self.__input_mask_torch
                )

            # novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []

            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(
                min(self.__opt.min_ver, self.__opt.min_ver - self.__opt.elevation),
                -80 - self.__opt.elevation,
            )
            max_ver = min(
                max(self.__opt.max_ver, self.__opt.max_ver - self.__opt.elevation),
                80 - self.__opt.elevation,
            )

            for _ in range(self.__opt.batch_size):
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.__opt.elevation + ver, hor, self.__opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.__cam.fovy,
                    self.__cam.fovx,
                    self.__cam.near,
                    self.__cam.far,
                )

                bg_color = torch.tensor(
                    [1, 1, 1] if np.random.rand() > self.__opt.invert_bg_prob else [0, 0, 0],
                    dtype=torch.float32,
                    device="cuda",
                )
                out = self.__renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.__opt.mvdream or self.__opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(
                            self.__opt.elevation + ver,
                            hor + 90 * view_i,
                            self.__opt.radius + radius,
                        )
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(
                            pose_i,
                            render_resolution,
                            render_resolution,
                            self.__cam.fovy,
                            self.__cam.fovx,
                            self.__cam.near,
                            self.__cam.far,
                        )

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.__renderer.render(cur_cam_i, bg_color=bg_color, convert_SHs_python=True)

                        image = out_i["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                        images.append(image)

                images = torch.cat(images, dim=0)
                poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.__device)

                # guidance loss
                if self.__enable_sd:
                    if self.__opt.mvdream or self.__opt.imagedream:
                        loss = loss + self.__opt.lambda_sd * self.__guidance_sd.train_step(
                            images,
                            poses,
                            step_ratio=step_ratio if self.__opt.anneal_timestep else None,
                        )
                    else:
                        loss = loss + self.__opt.lambda_sd * self.__guidance_sd.train_step(
                            images,
                            step_ratio=step_ratio if self.__opt.anneal_timestep else None,
                        )

                if self.__enable_zero123:
                    loss = loss + self.__opt.lambda_zero123 * self.__guidance_zero123.train_step(
                        images,
                        vers,
                        hors,
                        radii,
                        step_ratio=step_ratio if self.__opt.anneal_timestep else None,
                        default_elevation=self.__opt.elevation,
                    )
                # optimize step
                loss.backward()
                self.__optimizer.step()
                self.__optimizer.zero_grad()

                # densify and prune
                if (self.__step >= self.__opt.density_start_iter) and (self.__step <= self.__opt.density_end_iter):
                    viewspace_point_tensor, visibility_filter, radii = (
                        out["viewspace_points"],
                        out["visibility_filter"],
                        out["radii"],
                    )
                    self.__renderer.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.__renderer.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    self.__renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if self.__step % self.__opt.densification_interval == 0:
                        self.__renderer.gaussians.densify_and_prune(
                            self.__opt.densify_grad_threshold,
                            min_opacity=0.01,
                            extent=4,
                            max_screen_size=1,
                        )

                    if self.__step % self.__opt.opacity_reset_interval == 0:
                        self.__renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()

        self.__need_update = True

    @torch.no_grad()
    def _test_step(self):
        # ignore if no need to update
        if not self.__need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.__need_update:
            # render image

            cur_cam = MiniCam(
                self.__cam.pose,
                self.__W,
                self.__H,
                self.__cam.fovy,
                self.__cam.fovx,
                self.__cam.near,
                self.__cam.far,
            )
            out = self.__renderer.render(cur_cam, self.__gaussian_scale_factor)
            buffer_image = out[self.__mode]  # [3, H, W]

            if self.__mode in ["depth", "alpha"]:
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.__mode == "depth":
                    buffer_image = (buffer_image - buffer_image.min()) / (
                        buffer_image.max() - buffer_image.min() + 1e-20
                    )

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.__H, self.__W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            self.__buffer_image = (
                buffer_image.permute(1, 2, 0).contiguous().clamp(0, 1).contiguous().detach().cpu().numpy()
            )

            # display input_image
            if self.__overlay_input_img and self.__input_image is not None:
                self.__buffer_image = (
                    self.__buffer_image * (1 - self.__overlay_input_img_ratio)
                    + self.__input_image * self.__overlay_input_img_ratio
                )

            self.__need_update = False

        ender.record()
        torch.cuda.synchronize()
        # print("[INFO] Training time (log): ", t, " ms")

    def _load_image_prompt(self, file: str):
        # load image
        print(f"[INFO] load image from {file}...")
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.__bg_remover is None:
                self.__bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.__bg_remover)

        img = cv2.resize(img, (self.__W, self.__H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.__input_mask = img[..., 3:]

        # white bg
        self.__input_image = img[..., :3] * self.__input_mask + (1 - self.__input_mask)

        # bgr to rgb
        self.__input_image = self.__input_image[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f"[INFO] load prompt from {file_prompt}...")
            with open(file_prompt, "r") as f:
                self.__prompt = f.read().strip()

    def get_gs_model_data(self):
        return self.__renderer.gaussians.get_model_data()

    def get_gs_model(self):
        return self.__renderer.gaussians

    def train(self, models: list, iters=500):
        if iters > 0:
            self._prepare_training_model(models)
            for i in tqdm.trange(iters):
                self._train_step()

            # do a last prune
            self.__renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        return self.get_gs_model_data()
