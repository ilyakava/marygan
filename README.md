# K+1 GANS examples on Gaussians

What we expect Improved GAN and Mode GAN (with dynamic activation maximization) to learn.

![theory](./docs/figures/theory.png)

What Mode GAN learns.

<img src="https://raw.githubusercontent.com/ilyakava/marygan/master/docs/figures/actual.gif" alt="" data-canonical-src="https://raw.githubusercontent.com/ilyakava/marygan/master/docs/figures/actual.gif" width="300" height="300" />

Run this example on cpu with:

```
python main.py --d_loss nll --g_loss activation_maximization --n_real_classes 10 --variance 2.0 --ngpu 0
```

## Usage

See:

`python main.py`

## Versioning

- python 3.6.8
- torch '1.0.1'
- cuda/10.0.130 
- cudnn/7.5.0
