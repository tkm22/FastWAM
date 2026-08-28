import pytest
import torch

from fastwam.utils.video_metrics import (
    frechet_feature_distance,
    video_haar_wavelet_mse,
    video_lpips,
    video_psnr,
    video_ssim,
)


class _MeanSquaredDistance(torch.nn.Module):
    def forward(self, pred, target):
        return (pred - target).square().mean(dim=(1, 2, 3), keepdim=True)


def test_video_metrics_identical_clips():
    video = torch.rand(3, 4, 16, 18)
    assert video_psnr(video, video) == pytest.approx(80.0, abs=1e-4)
    assert video_ssim(video, video) == pytest.approx(1.0, abs=1e-5)
    assert video_lpips(video, video, _MeanSquaredDistance()) == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in video_haar_wavelet_mse(video, video).values())


def test_video_metrics_report_spatial_and_temporal_errors():
    target = torch.zeros(3, 2, 4, 4)
    pred = torch.ones_like(target)
    metrics = video_haar_wavelet_mse(pred, target)
    assert metrics["wavelet_ll_mse"] == pytest.approx(4.0)
    assert metrics["wavelet_lh_mse"] == pytest.approx(0.0)
    assert metrics["wavelet_hl_mse"] == pytest.approx(0.0)
    assert metrics["wavelet_hh_mse"] == pytest.approx(0.0)
    assert metrics["wavelet_temporal_low_mse"] == pytest.approx(2.0)
    assert metrics["wavelet_temporal_high_mse"] == pytest.approx(0.0)


def test_frechet_feature_distance_is_zero_for_identical_features():
    features = torch.tensor(
        [[0.0, 1.0, 2.0], [1.0, -1.0, 0.0], [2.0, 0.5, 1.0]]
    )
    assert frechet_feature_distance(features, features) == pytest.approx(
        0.0, abs=1e-7
    )
