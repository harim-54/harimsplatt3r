import json
import os
import sys

import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb

# Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
sys.path.append('src/pixelsplat_src')
sys.path.append('src/mast3r_src')
sys.path.append('src/mast3r_src/dust3r')
# L21은 거리 간의 유클리드 거리를 계산하는 학습 손실 함수로 DUSt3R 모델의 장점(카메라 파라미터 없이 3D 복원 가능)을 낸다.
from src.mast3r_src.dust3r.dust3r.losses import L21
# 신뢰도 기반의 손실함수? ConfLoss
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.scannetpp.scannetpp as scannetpp
import src.mast3r_src.mast3r.model as mast3r_model
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import workspace

# harim
from utils.geometry import constrain_to_ray, solve_sim3_alignment



class MAST3RGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        # Save the config
        self.config = config

        # The encoder which we use to predict the 3D points and Gaussians,
        # trained as a modified MAST3R model. The model's configuration is
        # primarily defined by the pretrained checkpoint that we load, see
        # MASt3R's README.md
        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed='RoPE100',
            patch_embed_cls='ManyAR_PatchEmbed',
            img_size=(512, 512),
            head_type='gaussian_head',
            output_mode='pts3d+gaussian+desc24',
            depth_mode=('exp', -mast3r_model.inf, mast3r_model.inf),
            conf_mode=('exp', 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, 'sh_degree') else 1
        )
        self.encoder.requires_grad_(False)
        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)

        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()

        # Loss criteria
        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS('vgg', spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS('vgg')

        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()


    def forward(self, view1, view2):

        # Freeze the encoder and decoder
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        # Train the downstream heads
        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

# harim : intrinsic 보정 연결
        if self.config.get('use_intrinsics', False):
    # 1. view1, view2에 들어있는 K 행렬(intrinsics)을 가져옵니다.

            K1 = view1['K'] 
            K2 = view2['K']
    
    # 2. utils/geometry.py에 만든 함수 호출
            pred1['pts3d'] = geometry.constrain_to_ray(pred1['pts3d'], K1)
            pred2['pts3d'] = geometry.constrain_to_ray(pred2['pts3d'], K2)
    
    # 3. 가우시안의 중심점(means)도 보정된 pts3d로 동기화
            pred1['means'] = pred1['pts3d']
            pred2['means'] = pred2['pts3d']
# harim 여기까지

        if self.config.get('use_gn_refine', False):
            B = pred1['pts3d'].shape[0]  # 배치 사이즈 확인
            device = pred1['pts3d'].device
            
            s_opts, R_opts, t_opts = [], [], []

            for b in range(B):
                # 각 배치별로 데이터를 꺼내어 GN 수행 (N, 3) 형태로 변환
                # .reshape(-1, 3)은 (H, W, 3)을 (H*W, 3)으로 펼쳐줍니다.
                p_src = pred2['pts3d'][b].reshape(-1, 3)
                p_tgt = pred1['pts3d'][b].reshape(-1, 3)
                conf = pred1['conf'][b].reshape(-1)

                init_guess = (torch.tensor(1.0, device=device),pred2['R'][b], pred2['t'][b])

                s_opt, R_opt, t_opt = geometry.refine_pose_gn(
                    pts_src = p_src,
                    pts_tgt = p_tgt,
                    confidence = conf,
                    initial_guess = init_guess
                )

                s_opts.append(s_opt)
                R_opts.append(R_opt)
                t_opts.append(t_opt)

            pred2['s'] = torch.stack(s_opts)
            pred2['R'] = torch.stack(R_opts)
            pred2['t'] = torch.stack(t_opts)

        pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
        pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])

        learn_residual = True
        
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1['sh'])
            new_sh2 = torch.zeros_like(pred2['sh'])
            new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1['original_img'], 'b c h w -> b h w c'))
            new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2['original_img'], 'b c h w -> b h w c'))
            pred1['sh'] = pred1['sh'] + new_sh1
            pred2['sh'] = pred2['sh'] + new_sh2

        # Update the keys to make clear that pts3d and means are in view1's frame
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        pred2['means_in_other_view'] = pred2.pop('means')

        return pred1, pred2

    def training_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)

# harim
# [수정 포인트: Global Alignment 연결]
# 1. MASt3R-SLAM의 Backend로부터 현재 프레임의 전역 포즈(T_WC)를 가져옴
        T_WCs = batch.get('T_WC')

# 2. 로컬 가우시안 중심(means)을 전역 좌표로 변환
# means_global = T_WC * means_local
        if T_WCs is not None:
            T_WC1 = T_WCs[0]
            T_WC2 = T_WCs[1]

            pred1['means'] = geometry.transform_points(pred1['means'], T_WC1)
            pred2['means_in_other_view'] = geometry.transform_points(pred2['means_in_other_view'], T_WC2)

# 3. 회전(Rotation) 성분도 전역으로 업데이트
            pred1['rotations'] = geometry.transform_rotations(pred1['rotations'], T_WC1)
            pred2['rotations'] = geometry.transform_rotations(pred2['rotations'], T_WC2)
        else:
            pass

        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        # Log losses
        self.log_metrics('train', loss, mse, lpips)
        return loss

    def validation_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']

        # Predict using the encoder/decoder and calculate the loss
        pred1, pred2 = self.forward(view1, view2)
        color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False
        )

        # Log losses
        self.log_metrics('val', loss, mse, lpips)
        return loss

    def test_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch['context']
        num_targets = len(batch['target'])

        # Predict using the encoder/decoder and calculate the loss
        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
# harim : global alignment 추가
        with self.benchmarker.time("global_alignment"):
            # A. Symmetric 매칭점 추출 (기존 pred 데이터 활용)
            pts1 = pred1['pts3d'].reshape(-1, 3)
            pts2 = pred2['means_in_other_view'].reshape(-1, 3)

            conf = pred1['conf'].reshape(-1) 

            # B. 초기 Sim3 최적화 (SVD 기반)
            s, R, t = geometry.solve_sim3_alignment(pts1, pts2)
            
            # C. Gauss-Newton 정밀 최적화
            # 노션의 solve_GN 로직을 호출하여 s, R, t를 더 정밀하게 다듬습니다.
            refined_s, refined_R, refined_t = geometry.refine_pose_gn(
                pts_src = pts2,
                pts_tgt = pts1,
                initial_guess = (s, R, t),
                confidence = conf
                )

            # D. 가우시안 중심(means)을 전역 좌표계로 변환
            # 이제 pred1, pred2의 위치는 단순한 view1 기준이 아니라 정렬된 상태가 됩니다.
            pred2['means_in_other_view'] = refined_s * (pred2['means_in_other_view'] @ refined_R.t()) + refined_t
# harim 코드 수정 끝

        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _ = self.decoder(batch, pred1, pred2, (h, w))

        # Calculate losses
        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips, ssim = self.calculate_loss(
            batch, view1, view2, pred1, pred2, color, mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True
        )

        # Log losses
        self.log_metrics('test', loss, mse, lpips, ssim=ssim)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss(self, batch, view1, view2, pred1, pred2, color, mask, apply_mask=True, average_over_mask=True, calculate_ssim=False):

        target_color = torch.stack([target_view['original_img'] for target_view in batch['target']], dim=1)
        predicted_color = color

        if apply_mask:
            assert mask.sum() > 0, "There are no valid pixels in the mask!"
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w')

        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        # Calculate the total loss
        loss = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss

        # MAST3R Loss
        if self.config.loss.mast3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss

        # Masked SSIM
        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }

        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.data.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [self.config.opt.epochs // 2], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def run_experiment(config):

    # Set the seed
    L.seed_everything(config.seed, workers=True)

    # Set up loggers
    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir,
            name=config.name
        )
        loggers.append(csv_logger)
    if config.loggers.use_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project='splatt3r',
            name=config.name,
            save_dir=config.save_dir,
            config=omegaconf.OmegaConf.to_container(config),
        )
        if wandb.run is not None:
            wandb.run.log_code(".")
        loggers.append(wandb_logger)

    # Set up profiler
    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler = None

    # Model
    print('Loading Model')
    model = MAST3RGaussians(config)
    if config.use_pretrained:
        ckpt = torch.load(config.pretrained_mast3r_path)
        _ = model.encoder.load_state_dict(ckpt['model'], strict=False)
        del ckpt

    # Training Datasets
    print(f'Building Datasets')
    train_dataset = scannetpp.get_scannet_dataset(
        config.data.root,
        'train',
        config.data.resolution,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch,
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    val_dataset = scannetpp.get_scannet_test_dataset(
        config.data.root,
        alpha=0.5,
        beta=0.5,
        resolution=config.data.resolution,
        use_every_n_sample=100,
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    # Training
    print('Training')
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            export.SaveBatchData(save_dir=config.save_dir),
        ],
        check_val_every_n_epoch=1,
        default_root_dir=config.save_dir,
        devices=config.devices,
        gradient_clip_val=config.opt.gradient_clip_val,
        log_every_n_steps=10,
        logger=loggers,
        max_epochs=config.opt.epochs,
        profiler=profiler,
        strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
    )
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

    # Testing
    original_save_dir = config.save_dir
    results = {}
    for alpha, beta in ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5), (0.3, 0.3)):

        test_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=alpha,
            beta=beta,
            resolution=config.data.resolution,
            use_every_n_sample=10
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )

        masking_configs = ((True, False), (True, True))
        for apply_mask, average_over_mask in masking_configs:

            new_save_dir = os.path.join(
                original_save_dir,
                f'alpha_{alpha}_beta_{beta}_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}'
            )
            os.makedirs(new_save_dir, exist_ok=True)
            model.config.save_dir = new_save_dir

            L.seed_everything(config.seed, workers=True)

            # Training
            trainer = L.Trainer(
                accelerator="gpu",
                benchmark=True,
                callbacks=[export.SaveBatchData(save_dir=config.save_dir),],
                default_root_dir=config.save_dir,
                devices=config.devices,
                log_every_n_steps=10,
                strategy="ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto",
            )

            model.lpips_criterion = lpips.LPIPS('vgg', spatial=average_over_mask)
            model.config.loss.apply_mask = apply_mask
            model.config.loss.average_over_mask = average_over_mask
            res = trainer.test(model, dataloaders=data_loader_test)
            results[f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"] = res

            # Save the results
            save_path = os.path.join(original_save_dir, 'results.json')
            with open(save_path, 'w') as f:
                json.dump(results, f)


if __name__ == "__main__":

    # Setup the workspace (eg. load the config, create a directory for results at config.save_dir, etc.)
    config = workspace.load_config(sys.argv[1], sys.argv[2:])
    if os.getenv("LOCAL_RANK", '0') == '0':
        config = workspace.create_workspace(config)

    # Run training
    run_experiment(config)
