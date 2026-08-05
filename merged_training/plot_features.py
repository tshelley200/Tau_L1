import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
import os

import matplotlib.pylab as pylab
import matplotlib as mpl

from utils import *

plt.style.use(hep.style.ROOT)
params = {'legend.fontsize': 'medium',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)
mpl.rcParams['lines.linewidth'] = 5

data_dir = "../data/merged/"
plot_dir = "../plots/features_barrel/"

if(not(os.path.exists(plot_dir))): os.system("mkdir " + plot_dir)
if(not(os.path.exists(plot_dir + "pt_corr/"))): os.system("mkdir " + plot_dir + "pt_corr/")

pt_cut = 15.
pt_norm = True
endcap = False

if(endcap):
    eta_min = 1.5
    eta_max = 2.4
else:
    eta_min = 0.0
    eta_max = 1.5

data = create_train_test_data(data_dir, train=True, pt_cut = pt_cut, pt_norm = pt_norm,
        track_extra_inputs = True, calo_extra_inputs = True, 
        eta_min = eta_min)

X_train, y_train_jetID, y_train_pt_regress = data['X'], data['Y_jetID'], data['Y_pt_corr']

pt_train = data['reco_pt']

sig_mask = y_train_jetID > 0.9



if(endcap): calo_feat_names = hgcal_feat_names
else: calo_feat_names = ecal_feat_names + hcal_feat_names

feat_names_all = feat_names + track_feat_names + calo_feat_names

print(X_train.shape)
print(len(feat_names), len(track_feat_names), len(calo_feat_names))

print(np.mean(X_train, axis = (0,1)))
print(np.std(X_train, axis = (0,1)))
eps=1e-8
sig_non_zero_cand = X_train[sig_mask][:,:,0] > eps
bkg_non_zero_cand = X_train[~sig_mask][:,:,0] > eps


#Pt plot
plt.figure()
h, bins, _ = plt.hist(pt_train[sig_mask], density = True, label = "Sig", color = 'r', alpha = 0.5)
plt.hist(pt_train[~sig_mask], bins = bins, density = True, label = "Bkg", color = 'b', alpha = 0.5)
plt.xlabel("Tau Cand Pt")
plt.legend()
plt.savefig(plot_dir +  "pt" + ".png")

#Feature plots
for ifeat in range(len(feat_names_all)):
    plt.figure()
    sig_feat = X_train[sig_mask][sig_non_zero_cand][:,ifeat].reshape(-1)
    bkg_feat = X_train[~sig_mask][bkg_non_zero_cand][:,ifeat].reshape(-1)
    h, bins, _ = plt.hist(sig_feat, density = True, label = "Sig", color = 'r', alpha = 0.5)
    plt.hist(bkg_feat, bins = bins, density = True, label = "Bkg", color = 'b', alpha = 0.5)

    plt.xlabel(feat_names_all[ifeat])
    plt.legend()
    plt.savefig(plot_dir + feat_names_all[ifeat] + ".png")

    pt_correlation = np.corrcoef(X_train[~sig_mask][:,0,ifeat].reshape(-1), pt_train[~sig_mask].reshape(-1))
    print("%i %s %.3f %.3f %.3f"  % (ifeat, feat_names_all[ifeat], np.mean(bkg_feat), np.std(bkg_feat), pt_correlation[0][1]))
    plt.figure()
    plt.scatter(pt_train[~sig_mask].reshape(-1),  X_train[~sig_mask][:,0,ifeat].reshape(-1), color='b', s = 5)
    plt.xlabel("Tau Cand. Pt")
    plt.ylabel(feat_names_all[ifeat])
    plt.title("Corr Coeff %.2f" % pt_correlation[0][1])
    plt.savefig(plot_dir + "pt_corr/"+ feat_names_all[ifeat] + ".png")

