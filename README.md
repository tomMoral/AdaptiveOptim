# AdaptiveOptim

Source code for the experiments and figures of the paper ["Adaptive Acceleration of Sparse Coding via Matrix Factorization "](https://arxiv.org/abs/1609.00285). 

### Requirements

 * numpy 1.10+
 * matplotlib 1.8+
 * tensorflow 0.9+
 * scikit-learn 1.16+

All the development was done with python3.4 and might not work for earlier versions.

### Usage

Use the main script `NIPS_figures.py` to launch the experiements. Various option are available from the command line. See `python NIPS_figures.py --help` for more information.

To generate the 4 figures from the paper, use:
```bash
python NIPS_figures.py --data artificial --save_dir layer1
python NIPS_figures.py --data artificial --rho .2 --save_dir layer2
python NIPS_figures.py --data mnist --lmbd .1 -K 100 --save_dir mnist
python NIPS_figures.py --data images --lmbd .05 --save_dir images
```
