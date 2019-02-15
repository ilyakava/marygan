import numpy as np

import cvxpy as cvx

import matplotlib.cm as cm
from matplotlib import pyplot

from scipy import stats

import pdb

def rel_H(x, y):
    # relative entropy in Boyd
    return cvx.sum(cvx.kl_div(x, y) + x - y)

def ternary_1D_Gauss():
    n_examples = 500
    f1_samples = np.random.normal(-1, 0.5, n_examples)
    f2_samples = np.random.normal(0.25, 0.5, n_examples)
    # make histograms
    # say from -2,2, 
    hist_range = (-2,2)
    nbins = 100
    f1, bins = np.histogram(f1_samples, nbins, hist_range)
    f1 = f1 / float(n_examples)
    f2, bins = np.histogram(f2_samples, nbins, hist_range)
    f2 = f2 / float(n_examples)
    bins = bins[:-1] # bins are +1 long for some reason
    bin_width = bins[1] - bins[0]

    # random histograms
    f1 = np.random.uniform(0, 1, len(f2))
    f1 = f1 / sum(f1)
    f2 = np.random.uniform(0, 1, len(f2))
    f2 = f2 / sum(f2)

    # the optimization problem:
    f0 = cvx.Variable(nbins)
    h = (f0 + f1 + f2)/3.0
    # loss = cvx.sum(cvx.kl_div((f0+f2)/2, h)) + cvx.sum(cvx.kl_div((f0+f1)/2, h)) + cvx.sum(cvx.kl_div((f1+f2)/2, h))
    # loss = rel_H((f0+f2)/2, h) + rel_H((f0+f1)/2, h) + rel_H((f1+f2)/2, h)
    # loss = rel_H((f1+f2)/2, h)
    loss = rel_H((f0+f2)/2, h) + rel_H((f0+f1)/2, h)
    constraints = [cvx.sum(f0) == 1, f0 >= 0]

    obj = cvx.Minimize(loss)
    prob = cvx.Problem(obj, constraints)

    prob.solve()
    
    mix = (f1+f2)/2.0
    print('distance from mixture: %f' % np.linalg.norm(mix-f0.value))

    pyplot.bar(bins, f1, width=bin_width, alpha=0.5, label='f1')
    pyplot.bar(bins, f2, width=bin_width, alpha=0.5, label='f2')
    pyplot.bar(bins, mix, width=bin_width, alpha=0.5, label='mix')
    pyplot.bar(bins, f0.value, width=bin_width, alpha=0.5, label='f0')
    pyplot.legend(loc='upper right')
    pyplot.show()

def solve_mary(*hists):
    hists = list(hists)
    nbins = len(hists[0])
    f0 = cvx.Variable(nbins)
    constraints = [cvx.sum(f0) == 1, f0 >= 0]
    hists.insert(0,f0)
    M = len(hists)
    h = f0
    for i in range(1,M):
        h += hists[i]
    
    loss = 0
    for fi in hists:
        loss += rel_H((h - fi)/(M-1), h/M)

    # just the first term
    # loss = rel_H((h - f0)/(M-1), h/M)

    obj = cvx.Minimize(loss)
    prob = cvx.Problem(obj, constraints)

    prob.solve()

    return f0.value

def normal_hist(mean, var, n_examples, nbins, hist_range):
    f1_samples = np.random.normal(mean, var, n_examples)
    # make histograms
    # say from -2,2, 
    f1, bins = np.histogram(f1_samples, nbins, hist_range)
    f1 = f1 / float(n_examples)
    bins = bins[:-1] # bins are +1 long for some reason
    return [f1, bins]

def quaternary_eg():
    n_examples = 5000
    hist_range = (-2,2)
    nbins = 100
    bin_width = (hist_range[1] - hist_range[0])/float(nbins)

    f1, bins = normal_hist(-1,0.5,n_examples,nbins,hist_range)
    f2, bins = normal_hist(0,0.5,n_examples,nbins,hist_range)
    f3, bins = normal_hist(1,0.5,n_examples,nbins,hist_range)

    f0 = solve_mary(f1,f2,f3)
    mix = (f1 + f2 + f3) / 3.0

    print('distance from mixture: %f' % np.linalg.norm(mix-f0))

    # pdb.set_trace()

    pyplot.bar(bins, mix, width=bin_width, alpha=0.5, label='mix')
    pyplot.bar(bins, f0, width=bin_width, alpha=0.5, label='f0')
    pyplot.legend(loc='upper right')
    pyplot.show()

    # you can see in this case that f0 is multimodal, not a literal mixture,
    # when the datasets are closer together

def two_dim_plot():
    """Take 2 2d gaussians and solves our loss for them
    """
    mu1 = (2,1)
    mu2 = (2,3)
    subtitle = "mu_data1 = {}, mu_data2 = {}, var = 1".format(mu1,mu2)
    x_range = (0,4)
    y_range = (mu1[1]-2,mu2[1]+2)
    nbins = (40,60)
    tbins = np.prod(nbins)
    n_examples = 10000000

    samp1x = np.random.normal(mu1[0], 1, n_examples)
    samp1y = np.random.normal(mu1[1], 1, n_examples)
    samp2x = np.random.normal(mu2[0], 1, n_examples)
    samp2y = np.random.normal(mu2[1], 1, n_examples)
    
    sampx = np.concatenate([samp1x, samp2x])
    sampy = np.concatenate([samp1y, samp2y])

    # normed=True does not mean it sums to 1
    Hmix, xedges, yedges = np.histogram2d(sampx, sampy, nbins, [x_range, y_range], normed=True)
    Hmix /= Hmix.sum()

    H1, xedges, yedges = np.histogram2d(samp1x, samp1y, nbins, [x_range, y_range], normed=True)
    H1 /= H1.sum()
    H2, xedges, yedges = np.histogram2d(samp2x, samp2y, nbins, [x_range, y_range], normed=True)
    H2 /= H2.sum()

    # fix orientation
    Hmix = Hmix.T
    H1 = H1.T
    H2 = H2.T
    nbinsT = (60,40)


    f1 = H1.reshape((tbins,))
    f2 = H2.reshape((tbins,))
    fmix = Hmix.reshape((tbins,))

    f0 = solve_mary(f1,f2)
    H0 = f0.reshape(nbinsT)


    _, f1_bins = np.histogram(f1, 5)
    _, f2_bins = np.histogram(f2, 5)
    fr_bins = np.mean([f1_bins, f2_bins],axis=0)
    fr_bins = fr_bins[:-1] # this last one is too close to the peak
    # _, f0_bins = np.histogram(f0, 5)

    fmix_bins = stats.mstats.mquantiles(fmix, [i/20.0 for i in range(7,20,3)])
    f0_bins = stats.mstats.mquantiles(f0, [i/20.0 for i in range(7,20,3)])
    fcomp_bins = np.mean([fmix_bins, f0_bins],axis=0)
    
    # plot 1
    fig, (ax, ax2, ax3) = pyplot.subplots(1, 3, sharey=True)
    ax.set_title('Contours of P_data1 and P_data2 over P_fake heatmap\n%s' % subtitle)

    im = ax.imshow(H0, cmap=cm.gray, extent=x_range+y_range)
    CS2 = ax.contour(H2, fr_bins, cmap='flag', extent=x_range+y_range)
    CS1 = ax.contour(H1, fr_bins, cmap='flag', extent=x_range+y_range)

    labs = {}
    for b in fr_bins:
        with_zero = '%.1E' % b
        labs[b] = with_zero[:-2] + with_zero[-1]
    ax.clabel(CS1, fr_bins[1:], inline=1, fmt=labs, fontsize=10)
    CBI = fig.colorbar(im, shrink=0.5, ax=ax)

    # plot 2
    ax2.set_title('Contours of (P_data1 + P_data2)/2 over P_fake heatmap\n%s' % subtitle)

    im = ax2.imshow(H0, cmap=cm.gray, extent=x_range+y_range)
    CS1 = ax2.contour(Hmix, fcomp_bins, cmap='flag', extent=x_range+y_range)

    labs = {}
    for b in fcomp_bins:
        with_zero = '%.1E' % b
        labs[b] = with_zero[:-2] + with_zero[-1]
    ax2.clabel(CS1, fcomp_bins, inline=1, fmt=labs, fontsize=10)
    CBI = fig.colorbar(im, shrink=0.5, ax=ax2)

    # plot 3
    ax3.set_title('Contours of P_fake over (P_data1 + P_data2)/2 heatmap\n%s' % subtitle)

    im = ax3.imshow(Hmix, cmap=cm.gray, extent=x_range+y_range)
    CS1 = ax3.contour(H0, fcomp_bins, cmap='flag', extent=x_range+y_range)

    ax3.clabel(CS1, fcomp_bins, inline=1, fmt=labs, fontsize=10)
    CBI = fig.colorbar(im, shrink=0.5, ax=ax3)

    fig.show()
    pdb.set_trace()



def ternary_eg():
    n_examples = 5000
    hist_range = (0,6)
    nbins = 100
    bin_width = (hist_range[1] - hist_range[0])/float(nbins)

    f1, bins = normal_hist(2,1,n_examples,nbins,hist_range)
    f2, bins = normal_hist(4,1,n_examples,nbins,hist_range)

    f0 = solve_mary(f1,f2)
    mix = (f1 + f2) / 2.0

    print('distance from mixture: %f' % np.linalg.norm(mix-f0))

    # pdb.set_trace()

    pyplot.bar(bins, mix, width=bin_width, alpha=0.5, label='mix')
    pyplot.bar(bins, f0, width=bin_width, alpha=0.5, label='f0')
    pyplot.legend(loc='upper right')
    pyplot.show()

    # pdb.set_trace()
    # you can see in this case that f0 is multimodal, not a literal mixture,
    # when the datasets are closer together

    pyplot.bar(bins, f1/2, width=bin_width, alpha=0.5, label='f1')
    pyplot.bar(bins, f2/2, width=bin_width, alpha=0.5, label='f2')
    pyplot.bar(bins, f0, width=bin_width, alpha=0.5, label='f0')
    pyplot.legend(loc='upper right')
    pyplot.show()

if __name__ == '__main__':
    two_dim_plot()