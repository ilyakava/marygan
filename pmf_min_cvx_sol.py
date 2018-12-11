import numpy as np

import cvxpy as cvx

from matplotlib import pyplot

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

def ternary_eg():
    n_examples = 5000
    hist_range = (-2,2)
    nbins = 100
    bin_width = (hist_range[1] - hist_range[0])/float(nbins)

    f1, bins = normal_hist(-1/2.0,0.5,n_examples,nbins,hist_range)
    f2, bins = normal_hist(1/2.0,0.5,n_examples,nbins,hist_range)

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
    ternary_eg()