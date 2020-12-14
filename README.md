# K+1 GANS examples on Gaussians

What we expect Improved GAN and Mode GAN (with dynamic activation maximization) to learn.

![theory](./docs/figures/theory.png)

What Mode GAN learns.

<img src="https://raw.githubusercontent.com/ilyakava/marygan/master/docs/figures/actual.gif" alt="" data-canonical-src="https://raw.githubusercontent.com/ilyakava/marygan/master/docs/figures/actual.gif" width="300" height="300" />

Run this example on cpu with:

```
python main.py --d_loss nll --g_loss activation_maximization --n_real_classes 10 --variance 2.0 --ngpu 0
```

Omit `--ngpu` to run on default GPU.

## Usage

See:

`python main.py`

Don't forget to run `visdom` in a different screen.

## Versioning

Works on torch 1.8.0 with python 3.6.12.

Works on ROCM 3.10 also.

Originally written on torch 1.0.1 python 3.6.8 cuda 10.0, cudnn 7.5.0

