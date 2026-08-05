#Import stuff
import uproot
import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import copy
import time

import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sklearn


import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot



from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

import optparse
import importlib
import pathlib
from keras import optimizers
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qlayers import QDense, QActivation
from qkeras import QConv1D
from qkeras.utils import load_qmodel
from keras.layers import *
from qkeras import *

import hist
from hist import Hist

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import cm
import mplhep as hep
plt.style.use(hep.style.ROOT)
import matplotlib as mpl
import random


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
        'axes.labelsize': 'x-large',
        'axes.titlesize':'x-large',
        'xtick.labelsize':'medium',
        'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

#line thickness
mpl.rcParams['lines.linewidth'] = 5


feat_names = ["Cand_pt", "Cand_Eta", "Cand_Phi", "IsPhoton", "IsElectron", "IsMuon", "IsNeutral Had", "IsChargedHad"]

track_feat_names = ["PuppiWeight", "Vx", "Vy", "Vz", "d0", "z0", "TrkMVA1", "TrkMVA2", "TrkMVA3", "CaloEta", "CaloPhi", "chi2", "nStubs", "nPar", "hOverE", "pfClusterPt", "egVsPionMVAOut", "egVsPUMVAOut"]

hgcal_feat_names = ["ShowerLength", "CoreShowerLength", "FirstLayer", "MaxLayer", "eMax", "sigmaEtaEtaMax", "sigmaPhiPhiMax", "sigmaEtaEtaTot", "sigmaPhiPhiTot",
                    "sigmaZZ", "sigmaRRTot", "sigmaRRMax", "sigmaRRMean", "zBaryCenter", "layer10percent", "layer90percent", "triggerCells67percent", "triggerCells90percent",
                    "mipPt", "sumPt"] 

ecal_feat_names = []
hcal_feat_names = []
for i in range(3):
    for j in range(15):
        if(j<3): hcal_feat_names.append("HCalCell_%i_%i" % (i, j))
        ecal_feat_names.append("ECalCell_%i_%i" % (i,j))


#Create training set
def create_train_test_data(dir_path, test_frac = 0.2, train = True, track_extra_inputs = False, calo_extra_inputs = False, 
        eta_min = 0., eta_max = 2.4, pt_cut = 50.0, pt_max = 9999.0, pt_norm = False):

    #Might have to change the version for other ntuple files
    sig = uproot.open(dir_path+"/sig.root")
    bkg = uproot.open(dir_path+"/bkg.root")
    qcd = uproot.open(dir_path+"/qcd.root")

    test_index_sig = int((1.0 - test_frac) * len(sig['ntuplePupSingle']['tree']['m_inputs'].array()))
    test_index_bkg = int((1.0 - test_frac) * len(bkg['ntuplePupSingle']['tree']['m_inputs'].array()))

    print(test_index_sig, test_index_bkg)
    n_parts = 10
    n_feats = 8
    n_track_extra = 18 # 10x18
    n_ecal_extra = 45 # 10x45, 10x9x5
    n_hcal_extra = 9 # 10x90

    #indices of features to scale by pt
    pt_norm_idxs = [0]
    offset = n_feats

    #To fix
    #if(track_extra_inputs):
    #    pt_norm_idxs.append(offset + track_feat_names.index("pfClusterPt"))
    #    offset+=n_track_extra
    #if(calo_extra_inputs):
    #    if(eta_min > 1.0): # kinda hacky, want to select hgcal features
    #        pt_norm_idxs += [offset + hgcal_feat_names.index("sumPt"), offset + hgcal_feat_names.index("mipPt")]
    #    else:
    #        
    #        #one ecal cell is buggy...
    #        new_idxs = list(range(offset, offset + n_ecal_extra + n_hcal_extra))
    #        new_idxs.remove(offset + ecal_feat_names.index("ECalCell_1_7"))
    #        new_idxs.remove(offset + ecal_feat_names.index("ECalCell_0_13"))
    #        pt_norm_idxs +=  new_idxs




    eps = 1e-8

    if train:

        truth_pt_sig = np.asarray(sig['ntuplePupSingle']['tree']['genpt1'].array()[:test_index_sig])
        truth_pt_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['genpt1'].array()[:test_index_bkg])
        truth_pt_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['genpt1'].array())

        reco_pt_sig = np.asarray(sig['ntuplePupSingle']['tree']['pt'].array()[:test_index_sig])
        reco_pt_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['pt'].array()[:test_index_bkg])
        reco_pt_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['pt'].array())


        reco_eta_sig = np.asarray(sig['ntuplePupSingle']['tree']['eta'].array()[:test_index_sig])
        reco_eta_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['eta'].array()[:test_index_bkg])
        reco_eta_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['eta'].array())

        deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[:test_index_sig]

        sig_input = np.asarray(sig['ntuplePupSingle']['tree']['m_inputs'].array())[:test_index_sig].reshape(-1, n_parts, n_feats)
        bkg_input = np.asarray(bkg['ntuplePupSingle']['tree']['m_inputs'].array())[:test_index_bkg].reshape(-1, n_parts, n_feats)
        qcd_input = np.asarray(qcd['ntuplePupSingle']['tree']['m_inputs'].array()).reshape(-1, n_parts, n_feats)

        sig_track_extra = np.asarray(sig['ntuplePupSingle']['tree']['extra_inputs'].array())[:test_index_sig].reshape(-1, n_parts, n_track_extra)
        bkg_track_extra = np.asarray(bkg['ntuplePupSingle']['tree']['extra_inputs'].array())[:test_index_bkg].reshape(-1, n_parts, n_track_extra)
        qcd_track_extra = np.asarray(qcd['ntuplePupSingle']['tree']['extra_inputs'].array()).reshape(-1, n_parts, n_track_extra)

        #Remove strange outliers
        sig_track_extra[:,:, 16:18] = np.clip(sig_track_extra[:,:, 16:18], -2.0, 2.0)
        bkg_track_extra[:,:, 16:18] = np.clip(bkg_track_extra[:,:, 16:18], -2.0, 2.0)
        qcd_track_extra[:,:, 16:18] = np.clip(qcd_track_extra[:,:, 16:18], -2.0, 2.0)

        sig_ecal_extra = np.asarray(sig['ntuplePupSingle']['tree']['ecal_shape'].array())[:test_index_sig].reshape(-1, n_parts, n_ecal_extra)
        bkg_ecal_extra = np.asarray(bkg['ntuplePupSingle']['tree']['ecal_shape'].array())[:test_index_bkg].reshape(-1, n_parts, n_ecal_extra)
        qcd_ecal_extra = np.asarray(qcd['ntuplePupSingle']['tree']['ecal_shape'].array()).reshape(-1, n_parts, n_ecal_extra)


        sig_hcal_extra = np.asarray(sig['ntuplePupSingle']['tree']['hcal_shape'].array())[:test_index_sig].reshape(-1, n_parts, n_hcal_extra)
        bkg_hcal_extra = np.asarray(bkg['ntuplePupSingle']['tree']['hcal_shape'].array())[:test_index_bkg].reshape(-1, n_parts, n_hcal_extra)
        qcd_hcal_extra = np.asarray(qcd['ntuplePupSingle']['tree']['hcal_shape'].array()).reshape(-1, n_parts, n_hcal_extra)

        if(pt_norm):
            for idx in pt_norm_idxs:
                sig_input[:,:,idx] /= reco_pt_sig.reshape(-1,1) + eps
                bkg_input[:,:,idx] /= reco_pt_bkg.reshape(-1,1) + eps
                qcd_input[:,:,idx] /= reco_pt_qcd.reshape(-1,1) + eps


    else:

        truth_pt_sig = np.asarray(sig['ntuplePupSingle']['tree']['genpt1'].array()[test_index_sig:])
        truth_pt_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['genpt1'].array()[test_index_bkg:])
        truth_pt_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['genpt1'].array())

        reco_pt_sig = np.asarray(sig['ntuplePupSingle']['tree']['pt'].array()[test_index_sig:])
        reco_pt_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['pt'].array()[test_index_bkg:])
        reco_pt_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['pt'].array())

        reco_eta_sig = np.asarray(sig['ntuplePupSingle']['tree']['eta'].array()[test_index_sig:])
        reco_eta_bkg = np.asarray(bkg['ntuplePupSingle']['tree']['eta'].array()[test_index_bkg:])
        reco_eta_qcd = np.asarray(qcd['ntuplePupSingle']['tree']['eta'].array())

        deltaR_sig = sig['ntuplePupSingle']['tree']['gendr1'].array()[test_index_sig:]


        sig_input = np.asarray(sig['ntuplePupSingle']['tree']['m_inputs'].array())[test_index_sig:].reshape(-1, n_parts, n_feats)
        bkg_input = np.asarray(bkg['ntuplePupSingle']['tree']['m_inputs'].array())[test_index_bkg:].reshape(-1, n_parts, n_feats)
        qcd_input = np.asarray(qcd['ntuplePupSingle']['tree']['m_inputs'].array()).reshape(-1, n_parts, n_feats)


        sig_track_extra = np.asarray(sig['ntuplePupSingle']['tree']['extra_inputs'].array())[test_index_sig:].reshape(-1, n_parts, n_track_extra)
        bkg_track_extra = np.asarray(bkg['ntuplePupSingle']['tree']['extra_inputs'].array())[test_index_bkg:].reshape(-1, n_parts, n_track_extra)
        qcd_track_extra = np.asarray(qcd['ntuplePupSingle']['tree']['extra_inputs'].array()).reshape(-1, n_parts, n_track_extra)


        sig_track_extra[:,:, 16:18] = np.clip(sig_track_extra[:,:, 16:18], -2.0, 2.0)
        bkg_track_extra[:,:, 16:18] = np.clip(bkg_track_extra[:,:, 16:18], -2.0, 2.0)
        qcd_track_extra[:,:, 16:18] = np.clip(qcd_track_extra[:,:, 16:18], -2.0, 2.0)


        sig_ecal_extra = np.asarray(sig['ntuplePupSingle']['tree']['ecal_shape'].array())[test_index_sig:].reshape(-1, n_parts,  n_ecal_extra)
        bkg_ecal_extra = np.asarray(bkg['ntuplePupSingle']['tree']['ecal_shape'].array())[test_index_bkg:].reshape(-1, n_parts,  n_ecal_extra)
        qcd_ecal_extra = np.asarray(qcd['ntuplePupSingle']['tree']['ecal_shape'].array()).reshape(-1, n_parts, n_ecal_extra)


        sig_hcal_extra = np.asarray(sig['ntuplePupSingle']['tree']['hcal_shape'].array())[test_index_sig:].reshape(-1, n_parts, n_hcal_extra)
        bkg_hcal_extra = np.asarray(bkg['ntuplePupSingle']['tree']['hcal_shape'].array())[test_index_bkg:].reshape(-1, n_parts, n_hcal_extra)
        qcd_hcal_extra = np.asarray(qcd['ntuplePupSingle']['tree']['hcal_shape'].array()).reshape(-1, n_parts, n_hcal_extra)

            
        if(pt_norm):
            for idx in pt_norm_idxs:
                sig_input[:,:,idx] /= reco_pt_sig.reshape(-1,1) + eps
                bkg_input[:,:,idx] /= reco_pt_bkg.reshape(-1,1) + eps
                qcd_input[:,:,idx] /= reco_pt_qcd.reshape(-1,1) + eps


    selection_sig = (reco_pt_sig > pt_cut) & (reco_pt_sig < pt_max) & (abs(deltaR_sig) < 0.4) & (abs(reco_eta_sig) > eta_min) & (abs(reco_eta_sig) < eta_max)
    selection_bkg = (reco_pt_bkg > pt_cut) & (reco_pt_bkg < pt_max) & (abs(reco_eta_bkg) > eta_min) & (abs(reco_eta_bkg) < eta_max)
    selection_qcd = (reco_pt_qcd > pt_cut) & (reco_pt_qcd < pt_max) & (abs(reco_eta_qcd) > eta_min) & (abs(reco_eta_qcd) < eta_max)


    sig_eff = np.mean(selection_sig)
    bkg_eff = np.mean(selection_bkg)
    print("sig eff %.3f" % sig_eff)
    print("bkg eff %.3f" % bkg_eff)

    y_sig_pt = truth_pt_sig[selection_sig]
    y_bkg_pt = truth_pt_bkg[selection_bkg]
    y_qcd_pt = truth_pt_qcd[selection_qcd]

    #Inputs: pt, eta, phi, particle id(one hot encoded)
    X_sig = np.nan_to_num(np.asarray(sig_input[selection_sig]))
    y_sig = np.full(X_sig.shape[0], 1.)
    sig_pt = np.asarray(reco_pt_sig[selection_sig])

    X_bkg = np.nan_to_num(np.asarray(bkg_input)[selection_bkg])
    y_bkg = np.full(X_bkg.shape[0], 0.)
    bkg_pt = np.asarray(reco_pt_bkg[selection_bkg])

    X_qcd = np.nan_to_num(np.asarray(qcd_input)[selection_qcd])
    y_qcd = np.full(X_qcd.shape[0], 0.)
    qcd_pt = np.asarray(reco_pt_qcd[selection_qcd])

    print("Nsig, Nbkg, Nqcd: %i %i %i" %(y_sig.shape[0], y_bkg.shape[0], y_qcd.shape[0]))

    bkg_pt_1 = np.ones(y_bkg_pt.shape[0])
    qcd_pt_1 = np.ones(y_qcd_pt.shape[0])

    if(train):
        X_train = np.concatenate([X_sig, X_bkg, X_qcd])
        y_train_jetID = np.concatenate([y_sig, y_bkg, y_qcd])
        y_train_pt_corr = np.concatenate([y_sig_pt / sig_pt,bkg_pt_1, qcd_pt_1])
        pt_train = np.concatenate([sig_pt, bkg_pt, qcd_pt])

        track_extra_train = np.concatenate( (sig_track_extra[selection_sig], bkg_track_extra[selection_bkg], qcd_track_extra[selection_qcd]))
        ecal_extra_train = np.concatenate( (sig_ecal_extra[selection_sig], bkg_ecal_extra[selection_bkg], qcd_ecal_extra[selection_qcd]))
        hcal_extra_train = np.concatenate( (sig_hcal_extra[selection_sig], bkg_hcal_extra[selection_bkg], qcd_hcal_extra[selection_qcd]))
    else:
        X_train = np.concatenate([X_sig, X_bkg])
        y_train_jetID = np.concatenate([y_sig, y_bkg])
        y_train_pt_corr = np.concatenate([y_sig_pt / sig_pt,bkg_pt_1])
        pt_train = np.concatenate([sig_pt, bkg_pt ])

        track_extra_train = np.concatenate( (sig_track_extra[selection_sig], bkg_track_extra[selection_bkg]))
        ecal_extra_train = np.concatenate( (sig_ecal_extra[selection_sig], bkg_ecal_extra[selection_bkg] ))
        hcal_extra_train = np.concatenate( (sig_hcal_extra[selection_sig], bkg_hcal_extra[selection_bkg]))

    X_train[abs(X_train) > 1e+9] = 0.
    track_extra_train[abs(track_extra_train) > 1e+9] = 0.
    ecal_extra_train[abs(ecal_extra_train) > 1e+9] = 0.
    hcal_extra_train[abs(hcal_extra_train) > 1e+9] = 0.

    X_train = np.nan_to_num(X_train, posinf = 0., neginf=0.)
    track_extra_train = np.nan_to_num(track_extra_train, posinf = 0., neginf=0.)
    ecal_extra_train = np.nan_to_num(ecal_extra_train, posinf = 0., neginf=0.)
    hcal_extra_train = np.nan_to_num(hcal_extra_train, posinf = 0., neginf=0.)

    assert not np.any(np.isnan(X_train)) 
    assert not np.any(np.isnan(track_extra_train))
    assert not np.any(np.isnan(ecal_extra_train))
    assert not np.any(np.isnan(hcal_extra_train))
    assert not np.any(np.isnan(y_train_jetID))
    assert not np.any(np.isnan(y_train_pt_corr))

    data_dict = {
            'X': X_train,

            'reco_pt': pt_train,
            'Y_jetID': y_train_jetID,
            'Y_pt_corr' : y_train_pt_corr,
            'sig_eff' : sig_eff,
            'bkg_eff' : bkg_eff,

            'track_extra': track_extra_train,
            'ecal_extra': ecal_extra_train,
            'hcal_extra': hcal_extra_train,
    }



    return data_dict


def class_model(input_shape = None):
    input_size = input_shape[-1]
    drop_rate = 0.2

    inputs = tf.keras.layers.Input(shape=(input_size,), name='model_input')

    main_branch = tf.keras.layers.Dense(32, activation = "ReLU", name='Dense1')(inputs)
    #main_branch = tf.keras.layers.Dense(32, activation = "ReLU", name='Dense2')(main_branch)
    main_branch = keras.layers.Dropout(drop_rate)(main_branch)
    main_branch = tf.keras.layers.Dense(16, activation = "ReLU", name='Dense3')(main_branch)
    main_branch = keras.layers.Dropout(drop_rate)(main_branch)
    main_branch = tf.keras.layers.Dense(16, activation = "ReLU", name='Dense4')(main_branch)
    main_branch = keras.layers.Dropout(drop_rate)(main_branch)
    main_branch = tf.keras.layers.Dense(8, activation = "ReLU", name='Dense5')(main_branch)

    jetID_branch = tf.keras.layers.Dense(1, activation='sigmoid', name='jetID_output')(main_branch)


    model = tf.keras.Model(inputs = inputs, outputs = jetID_branch)

    return model

def class_model_quant(input_shape=None, output_shape=[1], bits=9, bits_int=2, alpha_val=1):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform'
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=input_shape, name='model_input')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)

    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling1D(name='avgpool')(main)

    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('sigmoid', name='jet_id_output')(jet_id)

    #pT regression branch
    #pt_regress = QDense(10, name='Dense_1_pT', **common_args)(main)
    #pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)

    #pt_regress = QDense(1, name='pT_output',
    #                    kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                    bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                    kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    #model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])
    model = tf.keras.Model(inputs = inputs, outputs = jet_id)

    return model

def deepset_cnn(point_shape=None, cnn_shape=None, output_shape=[1], bits=9, bits_int=2, alpha_val=1):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform',
    }

    #Initialize inputs
    cnn_inputs = tf.keras.layers.Input(shape=cnn_shape, name='cnn_input')

    #cnn_main branch
    cnn_main = BatchNormalization(name='norm_input_cnn')(cnn_inputs)

    #First Conv2D
    cnn_main = QConv2D(filters=10, kernel_size=3, strides=(1,1), padding='same', data_format='channels_first', name='Conv2D_1', **common_args)(cnn_main)
    cnn_main = QActivation(activation=quantized_relu(bits), name='relu_11')(cnn_main)

    cnn_main = QConv2D(filters=10, kernel_size=3, strides=(2,2), padding='same', name='Conv2D_2', data_format='channels_first', **common_args)(cnn_main)
    cnn_main = QActivation(activation=quantized_relu(bits), name='relu_21')(cnn_main)

    #Second Conv2D
    cnn_main = QConv2D(filters=10, kernel_size=3, strides=(2,2), padding='same', data_format='channels_first', name='Conv2D_3', **common_args)(cnn_main)
    cnn_main = QActivation(activation=quantized_relu(bits), name='relu_31')(cnn_main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    cnn_main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool1')(cnn_main)
    cnn_main = GlobalAveragePooling2D(name='avgpool1', data_format='channels_first')(cnn_main)

    #Initialize deepset inputs
    inputs = tf.keras.layers.Input(shape=point_shape, name='point_input')

    #Main branch
    main = BatchNormalization(name='norm_input_point')(inputs)

    #First Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    #Second Conv1D
    main = QConv1D(filters=10, kernel_size=1, name='Conv1D_2', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool2')(main)
    main = GlobalAveragePooling1D(name='avgpool2')(main)

    main = keras.layers.Concatenate(name='concat')([cnn_main, main])


    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('sigmoid', name='jet_id_output')(jet_id)

    #pT regression branch
    #pt_regress = QDense(10, name='Dense_1_pT', **common_args)(main)
    #pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)

    #pt_regress = QDense(1, name='pT_output',
    #                    kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                    bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                    kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    #model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])
    model = tf.keras.Model(inputs = [inputs, cnn_inputs], outputs = jet_id)

    return model




def cnn(input_shape=None, output_shape=[1], bits=9, bits_int=2, alpha_val=1):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform',
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=input_shape, name='model_input')

    #Main branch
    main = BatchNormalization(name='norm_input')(inputs)

    #First Conv2D
    main = QConv2D(filters=10, kernel_size=3, strides=(1,1), padding='same', data_format='channels_first', name='Conv2D_1', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_1')(main)

    main = QConv2D(filters=10, kernel_size=3, strides=(2,2), padding='same', name='Conv2D_2', data_format='channels_first', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_2')(main)

    #Second Conv2D
    main = QConv2D(filters=10, kernel_size=3, strides=(2,2), padding='same', data_format='channels_first', name='Conv2D_3', **common_args)(main)
    main = QActivation(activation=quantized_relu(bits), name='relu_3')(main)

    # Linear activation to change HLS bitwidth to fix overflow in AveragePooling
    main = QActivation(activation='quantized_bits(18,8)', name = 'act_pool')(main)
    main = GlobalAveragePooling2D(name='avgpool', data_format='channels_first')(main)

    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('sigmoid', name='jet_id_output')(jet_id)

    #pT regression branch
    #pt_regress = QDense(10, name='Dense_1_pT', **common_args)(main)
    #pt_regress = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress)

    #pt_regress = QDense(1, name='pT_output',
    #                    kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                    bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
    #                    kernel_initializer='lecun_uniform')(pt_regress)

    #Define the model using both branches
    #model = tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress])
    model = tf.keras.Model(inputs = inputs, outputs = jet_id)

    return model




def merged_model(input_size = 80):

    inputs = tf.keras.layers.Input(shape=(input_size,), name='model_input')

    main_branch = tf.keras.layers.Dense(25, activation = "ReLU", name='Dense1')(inputs)
    main_branch = tf.keras.layers.Dense(25, activation = "ReLU", name='Dense2')(main_branch)
    main_branch = tf.keras.layers.Dense(15, activation = "ReLU", name='Dense3')(main_branch)
    main_branch = tf.keras.layers.Dense(15, activation = "ReLU", name='Dense4')(main_branch)
    main_branch = tf.keras.layers.Dense(10, activation = "ReLU", name='Dense5')(main_branch)

    jetID_branch = tf.keras.layers.Dense(1, activation='sigmoid', name='jetID_output')(main_branch)

    pT_branch = tf.keras.layers.Dense(1, name='pT_output')(main_branch)

    model = tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])

    return model

def quantized_merged_model(input_size = 80):

    bits = 9
    bits_int = 2
    alpha_val = 1
    bits_conv = 9


    model = tf.keras.Sequential()

    inputs = tf.keras.layers.Input(shape=(input_size,), name='model_input')

    #kernel_initializer = initializers.lecun_uniform(seed)

    main_branch = QDense(25, name='Dense_1',
            kernel_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(inputs)  # , kernel_regularizer=l1(0.0001)))
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_1')(main_branch)
    main_branch = QDense(25, name='Dense_2',
            kernel_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_2')(main_branch)
    main_branch = QDense(15, name='Dense_3',
            kernel_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)  # , kernel_regularizer=l1(0.0001)))
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_3')(main_branch)
    main_branch = QDense(15, name='Dense_4',
            kernel_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_4')(main_branch)
    main_branch = QDense(10, name='Dense_5',
            kernel_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)
    main_branch = QActivation(activation=quantized_relu(bits), name='relu_5')(main_branch)
    jetID_branch = QDense(1, name='Dense_6',
            kernel_quantizer=quantized_bits(
                16, 6, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                16, 6, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)
    jetID_branch = tf.keras.layers.Activation(activation='sigmoid', name='jetID_output')(jetID_branch)

    pT_branch = QDense(1, name='pT_output',
            kernel_quantizer=quantized_bits(
                16, 6, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                16, 6, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)


    #model.compile(loss='binary_crossentropy', optimizer='adam',
    #              metrics=['binary_accuracy'])

    model = tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])


    return model

def train_weights(y_train, truth_pt_train, class_labels, pt_flat_weighting=True):
    """
    Re-balancing the class weights and then flatten them based on truth pT
    """
    num_samples = y_train.shape[0]

    sample_weights = np.ones(num_samples)

    # Define pT bins
    pt_bins = np.array([
        15, 17, 19, 22, 25, 30, 35, 40, 45, 50,
        60, 76, 97, 122, 154, 195, 246, 311,
        393, 496, 627, 792, np.inf  # Use np.inf to cover all higher values
    ])
    
    # Initialize counts per class per pT bin
    class_pt_counts = {}
    
    # Calculate counts per class per pT bin
    for label, idx in class_labels.items():
        class_mask = y_train == idx
        class_pt_counts[idx], _ = np.histogram(truth_pt_train[class_mask], bins=pt_bins)
    
    # Compute the maximum counts per pT bin over all classes
    max_counts_per_bin = np.zeros(len(pt_bins)-1)
    for bin_idx in range(len(pt_bins)-1):
        counts_in_bin = [class_pt_counts[idx][bin_idx] for idx in class_labels.values()]
        max_counts_per_bin[bin_idx] = max(counts_in_bin)
    
    # Compute weights per class per pT bin
    weights_per_class_pt_bin = {}
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = np.zeros(len(pt_bins)-1)
        for bin_idx in range(len(pt_bins)-1):
            class_count = class_pt_counts[idx][bin_idx]
            if class_count == 0:
                weights_per_class_pt_bin[idx][bin_idx] = 0.
            else:
                weights_per_class_pt_bin[idx][bin_idx] = max_counts_per_bin[bin_idx] / class_count

    # Assign weights to samples
    for idx in class_labels.values():
        class_mask = y_train == idx
        class_truth_pt = truth_pt_train[class_mask]
        sample_indices = np.where(class_mask)[0]
        bin_indices = np.digitize(class_truth_pt, pt_bins) - 1  # Subtract 1 to get 0-based index
        bin_indices[bin_indices == len(pt_bins)-1] = len(pt_bins)-2  # Handle right edge
        sample_weights[sample_indices] = weights_per_class_pt_bin[idx][bin_indices]
    
    # Normalize sample weights
    sample_weights = sample_weights / np.mean(sample_weights)
