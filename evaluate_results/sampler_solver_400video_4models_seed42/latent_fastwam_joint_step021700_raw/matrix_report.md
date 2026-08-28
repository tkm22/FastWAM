# Pixel WAM 400-episode sampler/solver evaluation

Each row is 40 LIBERO tasks x trials 0..9 = 400 closed-loop episodes.
All episodes use joint video/action inference and save rollout and predicted video.

Progress: 6/6 cells, 2400/2400 episodes.

| Schedule | Solver | N | MFE | Success | Rate | gFVD | LPIPS | PSNR | SSIM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | euler | 10 | 10 | 397/400 | 99.25% | 60.461 | 0.1021 | 25.660 | 0.8812 |
| uniform | heun | 10 | 19 | 395/400 | 98.75% | 60.626 | 0.1026 | 25.594 | 0.8804 |
| uniform | midpoint | 10 | 20 | 393/400 | 98.25% | 62.374 | 0.1072 | 25.182 | 0.8745 |
| logit_normal | euler | 10 | 10 | 391/400 | 97.75% | 72.652 | 0.1080 | 25.608 | 0.8783 |
| logit_normal | heun | 10 | 19 | 389/400 | 97.25% | 71.556 | 0.1082 | 25.551 | 0.8777 |
| logit_normal | midpoint | 10 | 20 | 395/400 | 98.75% | 62.051 | 0.1058 | 25.246 | 0.8772 |
