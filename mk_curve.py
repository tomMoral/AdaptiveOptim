try:
    import sys
    sys.path.remove("/usr/lib/python3/dist-packages")
except ValueError:
    pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [7.0, 4.0]
print(mpl.rcParams['xtick.labelsize'])


def mk_curve(exp_name='sparse', eps=1e-6, max_iter=600, sym=50, save=None,
             rm=[]):
    curve_cost = np.load('exps/{}/curve_cost.npy'.format(exp_name)).take(0)
    # curve_cost = np.load('save_exp/{}/curve_cost.npy'.format(exp_name)
    #                      ).take(0)
    layer_lvl = [1, 2, 4, 7, 12, 21, 35, 59, 100]
    c_star = min(min(curve_cost['ista']), min(curve_cost['fista']))-eps
    c_star = min(curve_cost['ista'][-1], curve_cost['fista'][-1])-eps

    fig = plt.figure('Curve layer - {}'.format(exp_name))
    fig.clear()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(bottom=.12, top=.99)

    y_max = 0

    for model, name, style in [('lista', 'L-ISTA', 'bo-'),
                               ('lfista', 'L-FISTA', 'c*-'),
                               ('facto', 'FacNet', 'rd-')]:
        if model in rm:
            continue
        cc = np.maximum(curve_cost[model]-c_star, eps)
        y_max = max(y_max, cc[0])
        ll = layer_lvl[:len(cc)]
        ax.loglog(ll, cc, style, label=name)

    for model, name, style in [('ista', 'ISTA', 'g-'),
                               ('fista', 'FISTA', 'ys-'),
                               ('linear', 'Linear', '--og')
                               ]:
        if model in rm:
            continue
        try:
            cc = np.maximum(curve_cost[model]-c_star, eps)
        except KeyError:
            continue
        y_max = max(y_max, cc[0])
        iters = min(max_iter, len(cc))
        makers = np.unique((10**np.arange(0, np.log10(iters-1), 2/9)
                            ).astype(int))-1
        t = range(1, len(cc))
        ax.loglog(t, cc[1:], style,
                  markevery=makers,
                  label=name)

    ax.hlines([eps], 1, max_iter, 'k', '--')

    ax.legend(fontsize='x-large', ncol=2)
    ax.set_xlim((1, max_iter))
    ax.set_ylim((eps/2, sym*y_max))
    ax.set_xlabel('# iteration/layers k', fontsize='x-large')
    ax.set_ylabel('Cost function $F(z) - F(z^*)$', fontsize='x-large')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
    if save:
        plt.savefig('../Loptim/images/{}.pdf'.format(save), dpi=150)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Make figure')
    parser.add_argument('--exp', type=str, default='sparse.05',
                        help='name of the experience')
    parser.add_argument('--save', type=str, default=None,
                        help='save file for the figure')
    parser.add_argument('-x', type=int, default=600,
                        help='iteration maximal on the figure')
    parser.add_argument('-y', type=int, default=50,
                        help='scaling factor for y')
    parser.add_argument('--eps', type=float, default=1e-6,
                        help='scaling factor for y')
    parser.add_argument('--rm', nargs='+', type=str, default=[],
                        help='remove some curves from the plot')

    args = parser.parse_args()

    mk_curve(args.exp, eps=args.eps, max_iter=args.x, sym=args.y,
             save=args.save, rm=args.rm)
    plt.show()
