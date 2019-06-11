"""
Usage: python random_plot.py ./figure.pdf \
    ./configs/0030_0001_samples.csv ./configs/0030_0001_samples.csv ./configs/0030_0001_samples.csv ./configs/0030_0001_samples.csv \
    ./configs/0030_0001_cov.csv ./configs/0030_0001_cov.csv ./configs/0030_0001_cov.csv ./configs/0030_0001_cov.csv
"""
import sys
import numpy as np
import scipy
import scipy.stats
import pandas as pd

output_filename = sys.argv[1]
sample_filenames = sys.argv[2:6]
cov_filenames = sys.argv[6:10]

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
plt.switch_backend('PDF')
#plt.switch_backend('Agg')
#Use tex
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage[helvet]{sfmath}\usepackage{helvet}']

#Basic measurements
nrows = 3
ncols = 4
points = 10         # Font size
fig_w_in = 5.5      # Plot width (inches)
panel_wh_ratio = 0.9
panel_lm_in = 0.55
panel_rm_in = 0.05
panel_tm_in = 0.2
panel_bm_in = 0.45
fig_lm_in = 0.
fig_rm_in = 0.
fig_tm_in = 0.
fig_bm_in = 0.
y_labelpad = 5

panel_w_in = (fig_w_in - fig_lm_in - fig_rm_in)/ncols
panel_h_in = panel_wh_ratio * panel_w_in
fig_h_in = nrows*panel_h_in + fig_tm_in + fig_bm_in

panel_w_s  = panel_w_in  / fig_w_in
panel_h_s  = panel_h_in  / fig_h_in

panel_lm_s = panel_lm_in / fig_w_in
panel_rm_s = panel_rm_in / fig_w_in
panel_tm_s = panel_tm_in / fig_h_in
panel_bm_s = panel_bm_in / fig_h_in

fig_lm_s = fig_lm_in / fig_w_in
fig_rm_s = fig_rm_in / fig_w_in
fig_tm_s = fig_tm_in / fig_h_in
fig_bm_s = fig_bm_in / fig_h_in

pt_w_s = 1/72/fig_w_in
pt_h_s = 1/72/fig_w_in
char_w_s = pt_w_s*points
char_h_s = pt_h_s*points

def bottom_margin(row):
    return (nrows - (row + 1))*panel_h_s + panel_bm_s + fig_bm_s
def left_margin(col):
    return col*panel_w_s + panel_lm_s + fig_lm_s
def rect(row, col):
    return [left_margin(col), bottom_margin(row), panel_w_s - panel_lm_s - panel_rm_s, panel_h_s - panel_tm_s - panel_bm_s]
def label(ax, s):
    lmbm, rmtm = ax.get_position().get_points()
    lm, bm = lmbm
    rm, tm = rmtm
    w = rm - lm
    h = tm - bm
    ax.figure.text(lm-3.3*char_w_s, tm+char_h_s, r'\textbf{' + s + r'}')

fig = plt.figure(figsize=(fig_w_in, fig_h_in))

def set_ylabel_coords(ax, yshift=0):
    lmbm, rmtm = ax.get_position().get_points()
    lm, bm = lmbm
    rm, tm = rmtm
    w = rm - lm
    h = tm - bm
    ax.yaxis.set_label_coords(lm-2.5*char_w_s, bm+h/2+h*yshift, transform = ax.figure.transFigure)

z = scipy.stats.norm(0, 1)
lim = 4
titles = ["C=3", "C=10", "C=30", "C=100"]
for i in range(4):
    ax=fig.add_axes(rect(0, i))
    df = pd.read_csv(sample_filenames[i])
    ax.hist(np.array(df.r0), bins=50, range=(-lim, lim), density=True)
    xs = np.linspace(-lim, lim, 100)
    ax.plot(xs, z.pdf(xs), linewidth=1)
    ax.set_ylim(0, 0.7)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(titles[i], pad=-5)
    ax.set_xlim(-lim, lim)
    ax.set_xticks([-lim, 0, lim])
    ax.set_xlabel('output')
    if i == 0:
        label(ax, 'A')
        ax.set_ylabel("pdf")
        set_ylabel_coords(ax)


for i in range(4):
    ax=fig.add_axes(rect(1, i))
    df = pd.read_csv(sample_filenames[i])

    xs, ys = scipy.stats.probplot(np.array(df.r0), dist=z, fit=False)

    ax.plot(xs, ys, linewidth=1)
    ax.plot([-lim, lim], [-lim, lim], linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-lim, 0, lim])
    ax.set_yticks([-lim, 0, lim])
    ax.set_xlabel('limiting q.')
    if i == 0:
        label(ax, 'B')
        ax.set_ylabel('sampled q.')
        set_ylabel_coords(ax)

lim = 2
for i in range(4):
    ax=fig.add_axes(rect(2, i))
    df = pd.read_csv(cov_filenames[i])

    est = np.array(df.est)
    true = np.array(df.true)

    ax.plot([0, lim], [0, lim], color='tab:orange', linewidth=1)
    ax.scatter(true, est, 0.3, color='tab:blue')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('limiting cov.')
    ax.set_xlim(0, 1, lim)
    ax.set_ylim(0, 1, lim)
    ax.set_xticks([0, 1, lim])
    ax.set_yticks([0, 1, lim])
    if i == 0:
        label(ax, 'C')
        ax.set_ylabel('sampled cov.')
        set_ylabel_coords(ax, yshift=-0.05)

fig.savefig(output_filename)
