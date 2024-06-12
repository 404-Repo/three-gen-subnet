import numpy as np
import cv2
import tqdm
import rembg
import os
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf

from .CameraUtils import orbit_camera, OrbitCamera
from .GaussianSplattingModel import Renderer, MiniCam

import trimesh
from mesh_renderer import RendererMesh
from mesh import safe_normalize

from grid_put import mipmap_linear_grid_put_2d

# from kiui.lpips import LPIPS

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


        #####################
        # add to process main2
        #####################
        # self custom 
        self.opt = opt


        
        
        
        # input image
        
        self.__input_image = None
        self.input_img = self.__input_image
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


        # models
        self.device = torch.device("cuda")
        
        
        ##################################################################
        # renderer
        self.opt = opt
        self.renderer = RendererMesh(self.opt).to(self.device)
        
        
                
        # MESH 
        ## mesh renderer
        self.__mesh_renderer = None
        ## optimization
        self.__mesh_optimizer = None
        
        ## cam 
        self.__mesh_fixed_cam = None
        
        # input image
        self.__input_img_torch_channel_last  = None
        
        
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

        # training
        print("__train_steps: ", self.__train_steps)
        print("ssssssssssssssssssssssssstart training")
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

                vers.append(ver) # CHECK
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

                # enable mvdream training #CHECK
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
        torch.cuda.synchronize() #CHECK
        t = starter.elapsed_time(ender) 

        self.__need_update = True  #CHECK  y hệt main 1 ở đây ==> hết main1
        
        ## do a last prune
        # self.__renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        ## save to logs folder 
        # self.save_model(mode='model')
        # self.save_model(mode='geo+tex', texture_size=1024)

        
        #--------------------------------------------------------------------------------------------------------------------------------------------
        # REFINE PHASE
        #--------------------------------------------------------------------------------------------------------------------------------------------
        
        
        
        
        # # mesh renderer
        # self.__mesh_renderer = RendererMesh(self.__opt).to(self.device)
        
        # self.__mesh_optimizer = torch.optim.Adam(self.renderer.get_params())

        # self.__mesh_fixed_cam = (pose, self.cam.perspective)
        
        # self.__input_img_torch_channel_last =  self.__input_img_torch[0].permute(1,2,0).contiguous()
        
        
        # # concat variable 
        # self.step = 0 
        # self.train_steps = self.__train_steps
        # self.opt = self.__opt
        # self.input_img_torch = self.__input_img_torch
        # self.fixed_cam = self.__fixed_cam
        # self.renderer = self.__mesh_renderer
        # self.optimizer = self.__mesh_optimizer
        # self.cam = self.__mesh_fixed_cam
        # self.input_img_torch_channel_last = self.__input_img_torch_channel_last
        # self.guidance_sd = self.__guidance_sd 
        # self.guidance_zero123 = self.__guidance_zero123
        # self.enable_sd = self.__enable_sd
        # self.enable_zero123 = self.__enable_zero123
        # self.device = self.__device
        
        
        # for _ in range(self.train_steps):

        #     self.step += 1
        #     step_ratio = min(1, self.step / self.opt.iters_refine)
        
        #     loss = 0

        #     ### known view
        #     if self.input_img_torch is not None and not self.opt.imagedream:

        #         ssaa = min(2.0, max(0.125, 2 * np.random.random()))
        #         out = self.renderer.render(*self.fixed_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

        #         # rgb loss
        #         image = out["image"] # [H, W, 3] in [0, 1]

        #         valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()
        #         loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)

    
        #     ### novel view (manual batch)
        #     render_resolution = 512
        #     images = []
        #     poses = []
        #     vers, hors, radii = [], [], []
    
        #     # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        #     min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        #     max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
        #     for _ in range(self.opt.batch_size):

        #         # render random view
        #         ver = np.random.randint(min_ver, max_ver)
        #         hor = np.random.randint(-180, 180)
        #         radius = 0

        #         vers.append(ver)
        #         hors.append(hor)
        #         radii.append(radius)

        #         pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
        #         poses.append(pose)

        #         # random render resolution
        #         ssaa = min(2.0, max(0.125, 2 * np.random.random()))
        #         out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

        #         image = out["image"] # [H, W, 3] in [0, 1]
        #         image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

        #         images.append(image)

        #         # enable mvdream training
        #         if self.opt.mvdream or self.opt.imagedream:
        #             for view_i in range(1, 4):
        #                 pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
        #                 poses.append(pose_i)

        #                 out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

        #                 image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
        #                 images.append(image)

        #     images = torch.cat(images, dim=0)
        #     poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
        

        #     # import kiui
        #     # kiui.lo(hor, ver)
        #     # kiui.vis.plot_image(image)

        #     # guidance loss
        #     strength = step_ratio * 0.15 + 0.8
        #     if self.enable_sd:
        #         if self.opt.mvdream or self.opt.imagedream:
        #             # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
        #             refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
        #             refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
        #             loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
        #         else:
        #             # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
        #             refined_images = self.guidance_sd.refine(images, strength=strength).float()
        #             refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
        #             loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)

        #     if self.enable_zero123:
        #         # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
        #         refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
        #         refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
        #         loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
        #         # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

        #     # optimize step
        #     loss.backward()
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()


        # ender.record()
        # torch.cuda.synchronize()
        # t = starter.elapsed_time(ender)

        # self.need_update = True

        # if self.gui:
        #     dpg.set_value("_log_train_time", f"{t:.4f}ms")
        #     dpg.set_value(
        #         "_log_train_log",
        #         f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
        #     )

        # # dynamic train steps (no need for now)
        # # max allowed train time per-frame is 500 ms
        # # full_t = t / self.train_steps * 16
        # # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        # #     self.train_steps = train_steps
                

###################################################################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.__opt.outdir, exist_ok=True)
        if mode == 'geo':
            path = os.path.join(self.__opt.outdir, self.__opt.save_path + '_mesh.ply')
            mesh = self.__renderer.gaussians.extract_mesh(path, self.__opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.__opt.outdir, self.__opt.save_path + '_mesh.' + self.__opt.mesh_format)
            mesh = self.__renderer.gaussians.extract_mesh(path, self.__opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.__opt.force_cuda_rast and (not self.__opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.__cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.__cam.fovy,
                    self.__cam.fovx,
                    self.__cam.near,
                    self.__cam.far,
                )
                
                cur_out = self.__renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.__cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            # save model
            path = os.path.join(self.__opt.outdir, self.__opt.save_path + '_model.ply')
            self.__renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")






#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
###################################################################################################################
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

        
        ## do a last prune
        self.__renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)

        # # save to logs folder 
        self.save_model(mode='model')
        self.save_model(mode='geo+tex', texture_size=1024)

        
        return self.get_gs_model_data()

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    def prepare_train(self):

        self.step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)

        self.fixed_cam = (pose, self.__cam.perspective)
        

        self.enable_sd = self.opt.lambda_sd > 0 and self.__prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.__guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.__guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.__guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.__guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.__guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.__guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.__guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")
        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
            self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.__guidance_sd.get_image_text_embeds(self.input_img_torch, [self.__prompt], [self.__negative_prompt])
                else:
                    self.__guidance_sd.get_text_embeds([self.__prompt], [self.__negative_prompt])

            if self.enable_zero123:
                self.__guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()


        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)

            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:

                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(*self.fixed_cam, self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                # rgb loss
                image = out["image"] # [H, W, 3] in [0, 1]
                valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()
                loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)

            ### novel view (manual batch)
            render_resolution = 512
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(pose, self.__cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                images.append(image)

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        out_i = self.renderer.render(pose_i, self.__cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                        image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # import kiui
            # kiui.lo(hor, ver)
            # kiui.vis.plot_image(image)

            # guidance loss
            strength = step_ratio * 0.15 + 0.8
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                    refined_images = self.__guidance_sd.refine(images, poses, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
                else:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                    refined_images = self.__guidance_sd.refine(images, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)

            if self.enable_zero123:
                # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                refined_images = self.__guidance_zero123.refine(images, vers, hors, radii, strength=strength, default_elevation=self.opt.elevation).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
                # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    
    
    
    def train_2(self, models: list, iters=50):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self._train_step()

        return self.get_gs_model_data()
    
    
