import os
from utils import *

def plot_rate_data(scores_bkg, scores_sig, bkg_rate = 32e+3, label = ""):
    
    n_bkg_event = scores_bkg.shape[0]
    n_sig_event = scores_sig.shape[0]
    
    ##LOOP to calculate the ROC curves
    tau_score_edges = [round(i,2) for i in np.arange(0, 0.8, 0.01).tolist()]+\
                  [round(i,4) for i in np.arange(0.9, 1, 0.0001)] + [1]

    sig_list = []
    bkg_list = []

    for tau_score_cut in tau_score_edges:
    
        bkg_pass = np.sum(scores_bkg>tau_score_cut)
        sig_pass = np.sum(scores_sig>tau_score_cut)
        
        bkg_list.append(bkg_pass/n_bkg_event)
        sig_list.append(sig_pass/n_sig_event)

    
    bkg_list_scaled = np.array(bkg_list) * bkg_rate
    
    plt.plot(np.array(sig_list), bkg_list_scaled, linewidth=5, label=label)


pt_cut = 50.
#pt_max = 60.
pt_max = 9999.
eta_min = 1.5
eta_max = 5.0
pt_norm = False
#models = ["../models/mar6/deepset_baseline_barrel.h5", "../models/mar6/deepset_pt40_barrel.h5", "../models/mar6/deepset_pt40_ptrel_barrel.h5", "../models/mar6/deepset_pt_rw_barrel.h5"]
#labels = ['Deepset', 'Deepset pt > 40', 'Deepset pt relative', 'Deepset pt flat' ]

#models = ["../models/mar6/deepset_pt_rw_barrel.h5", "../models/mar6/deepset_track_only_barrel.h5", "../models/mar6/cnn_ecal_only_barrel.h5", "../models/mar6/cnn_hcal_only_barrel.h5"]
#labels = ['Deepset baseline', 'Deepset track feats only', 'CNN ECal only', 'CNN HCal only' ]
#model_str = ['deepset', 'deepset_track_only','CNN_ecal_only', 'CNN_hcal_only' ]

#models = ["../models/mar6/deepset_pt_rw_barrel.h5", "../models/mar6/deepset_track_extra_barrel.h5", 
#          "../models/mar6/deepset_cnn_ecal_extra_barrel.h5", "../models/mar6/deepset_cnn_hcal_extra_barrel.h5"]
#labels = ['Deepset baseline', 'Deepset track extra only', 'Deepset + CNN ECal extra', 'Deepset + CNN HCal extra']
#model_str = ['deepset', 'deepset_track_extra','deepset_cnn_ecal_extra', 'deepset_cnn_hcal_extra']

#models = ["../models/mar6/deepset_baseline_endcap.h5", "../models/mar6/deepset_track_only_endcap.h5", 
#          "../models/mar6/deepset_hgcal_only_endcap.h5"]
#labels = ['Deepset baseline', 'Deepset track extra only', 'Deepset  HGcal extra only']
#model_str = ['deepset', 'deepset_track_only','deepset_hgcal_only', ]

models = ["../models/mar6/deepset_baseline_endcap.h5", "../models/mar6/deepset_track_extra_endcap.h5", 
          "../models/mar6/deepset_hgcal_extra_endcap.h5"]
labels = ['Deepset baseline', 'Deepset track extra', 'Deepset  HGcal extra']
model_str = ['deepset', 'deepset_track_extra','deepset_hgcal_extra', ]

data_dir = "../data/merged/"
plot_dir = "../plots/endcap_feat_only_cmp/"
test_frac = 0.3
if(not(os.path.exists(plot_dir))): os.system("mkdir %s" % plot_dir)


data_no_norm = create_train_test_data(data_dir, train=False, pt_cut=pt_cut, pt_max = pt_max, eta_min=eta_min, eta_max=eta_max, test_frac=test_frac, pt_norm = False)
if(np.any(pt_norm)):
    data_pt_norm = create_train_test_data(data_dir, train=False, pt_cut=pt_cut, pt_max = pt_max, eta_min=eta_min, eta_max=eta_max, test_frac=test_frac, pt_norm = True)
else:
    data_pt_norm = None

_, y_test_jetID, y_test_pT_regress, sig_eff, bkg_eff  = data_no_norm['X'], data_no_norm['Y_jetID'], data_no_norm['Y_pt_corr'], data_no_norm['sig_eff'], data_no_norm['bkg_eff']

scores=[]

for i in range(len(models)):
    data = data_pt_norm if 'pt_norm' in model_str else data_no_norm
    dense = False
    

    X_test = data['X']
    if('track_extra' in model_str[i]):
        X_test = np.concatenate((X_test, data['track_extra']), axis = -1)
    if('calo_extra' in model_str[i]):
        X_test = np.concatenate((X_test, data['ecal_extra'], data['hcal_extra']), axis = -1)
    if('ecal_extra' in model_str[i]):
        X_test_calo = data['ecal_extra'].reshape(-1, 10, 3, 15)
        X_test = [X_test, X_test_calo]

    if('hcal_extra' in model_str[i]):
        X_test_calo = data['hcal_extra'].reshape(-1, 10, 3, 3)
        X_test = [X_test, X_test_calo]
    if('hgcal_extra' in model_str[i]):
        X_test_calo = data['ecal_extra'].reshape(X_test.shape[0], 10, -1)
        X_test = np.concatenate((X_test, X_test_calo), axis = -1)
    if('dense' in model_str[i]):
        X_test = X_test.reshape(X_test.shape[0], -1)

        scaler = joblib.load(models[i].replace(".h5", "_scaler.pkl"))
        X_test = scaler.transform(X_test)
        dense = True

    if('track_only' in model_str[i]):
        X_test = data['track_extra']

    if('ecal_only' in model_str[i]):
        X_test = data['ecal_extra'].reshape(-1, 10, 3, 15)
    if('hcal_only' in model_str[i]):
        X_test = data['hcal_extra'].reshape(-1, 10, 3, 3)
    if('hgcal_only' in model_str[i]):
        X_test = data['ecal_extra'].reshape(X_test.shape[0], 10, -1)
        #X_test = X_test[:,:,:len(hgcal_feat_names)]


    model = load_model(models[i]) if dense else load_qmodel(models[i])
    #print(model_str[i], X_test.shape)
    preds = model.predict(X_test, batch_size = 256)

    scores.append(preds)


plt.figure()
for i in range(len(labels)):
    preds = scores[i]
    label = labels[i]

    with np.errstate(divide='ignore', invalid='ignore'):
        fpr, tpr, _ = roc_curve(y_test_jetID, preds)
        sic = tpr / np.sqrt(fpr)

        random_tpr = np.linspace(0, 1, 300)
        random_sic = random_tpr / np.sqrt(random_tpr)

    auc_ = auc(fpr, tpr)
    print("%s AUC %.3f" % (label, auc_))

    plt.plot(fpr, tpr, label="%s (AUC = %.3f)" % (label, auc_))

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.savefig(plot_dir + "roc_performance.png")
	

#Rate plot
plt.figure()
plt.plot([],[], 'none', label=r'(MinBias $p_T$ >  % d GeV, $0 < |\eta| < 1.5 $)'%int(pt_cut))

min_bias_rate = 32e+3
min_bias_selected_rate = min_bias_rate * bkg_eff

for i in range(len(labels)):
    preds = scores[i]
    plot_rate_data(preds[y_test_jetID == 0], preds[y_test_jetID ==1], bkg_rate = min_bias_selected_rate, label = labels[i])


hep.cms.text("Phase 2 Simulation")
hep.cms.lumitext("PU 200 (14 TeV)")

plt.ylabel(r'$Single \tau_h$ Trigger Rate [kHz]')
plt.xlabel(r'$\tau_h$ [$p_T^{gen} > % d$ GeV]' %int(pt_cut))

plt.ylim([0.1, 1e3])
plt.yscale('log')
#plt.legend(loc='best',fontsize=20)
plt.legend(loc='upper left')
plt.savefig(plot_dir + "rate_comparison.png")


