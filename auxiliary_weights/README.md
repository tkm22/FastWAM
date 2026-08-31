# Auxiliary weights

This directory is the single local root for runtime weights that are not
FastWAM model checkpoints. Binary `*.pt` files are ignored by Git and must not
be committed.

- `projections/` contains fitted pixel-to-latent projection weights produced by
  `tools/fit_asym_fastwam_procrustes.py`.
- `metrics/` contains frozen feature extractors used only for evaluation. The
  Kinetics-400 I3D detector used by gFVD is stored as
  `metrics/i3d_torchscript.pt`.

The I3D TorchScript file comes from the official StyleGAN-V FVD implementation:

```text
https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1
sha256 bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2
```
