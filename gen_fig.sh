#!/usr/bin/sh

python mk_curve.py --exp sparse.05 --save curve_sparse005_large -x 300 -y 5 --noshow --eps 1e-7
python mk_curve.py --exp sparse.05 --save curve_sparse005_seaborn -x 300 -y 5 --seaborn --noshow --eps 1e-7

python mk_curve.py --exp sparse.2 --save curve_sparse02_large -x 1000 -y 5 --noshow
python mk_curve.py --exp sparse.2 --save curve_sparse02_seaborn -x 1000 -y 5 --seaborn --noshow

python mk_curve.py --exp mnist_100_05  -x 500 -y 10 --eps 1e-2 --save curve_mnist_large --noshow
python mk_curve.py --exp mnist_100_05  -x 500 -y 10 --eps 1e-2 --seaborn --save curve_mnist_seaborn --noshow

python mk_curve.py --exp images  -x 500 -y 2 --save curve_images_large --noshow
python mk_curve.py --exp images  -x 500 -y 2 --seaborn --save curve_images_seaborn --noshow

python mk_curve.py --exp adverse -x 1000 -y 100 --rm lfista linear -y 1 --save curve_adverse_large --noshow
python mk_curve.py --exp adverse -x 1000 -y 100 --rm lfista linear -y 1 --seaborn --save curve_adverse_seaborn --noshow

python plot_adverse_dictionary.py --save dictionary --noshow
