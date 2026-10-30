import uproot
import numpy as np
import awkward as ak
from scipy.stats import norm
from scipy.optimize import curve_fit
import os
import copy
import time
import json

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
mpl.rcParams['lines.linewidth'] = 5


# ==================== CONSTANTS ====================
TREE        = "ntuplePupSingle/tree"  # one row per L1 tau candidate
EVT_TREE    = "ntuple/tree"           # one row per gen object; contains EVERY processed event (used for the rate)
N_PARTS     = 10                      # particles per tau candidate
N_BASE      = 8                       # base features per particle (m_inputs)
BARREL_ETA  = 1.5                     # |eta| < 1.5 = barrel (GCT clusters), |eta| >= 1.5 = endcap (HGCal)
TAU_ETA_MAX = 2.5                     # taus are only seeded at |eta| < 2.5 (CMSSW + firmware)
MINBIAS_RATE_KHZ = 32e+3              # rate of filled bunch crossings (kHz), used to turn a fraction into a rate

PARTICLE_MODELS = ["class_model_quant", "class_model", "merged_model", "quantized_merged_model"]  # calo info glued onto particles
IMAGE_MODELS    = ["cnn", "deepset_cnn"]                                                          # calo info as an image
MERGED_MODELS   = ["merged_model", "quantized_merged_model"]                                      # firmware model = quantized_merged_model
FLAT_MODELS     = ["class_model"] + MERGED_MODELS                                                 # take a flattened (10*F,) input


# --- feature names (used for plotting / bookkeeping only) ---
# feat_names: the 8 base features per particle in 'm_inputs': pT, deta, dphi (relative to the seed) & its five categories
feat_names = ["Cand_pt", "Cand_Eta", "Cand_Phi", "IsPhoton", "IsElectron", "IsMuon", "IsNeutral Had", "IsChargedHad"]

# track features per particle in 'extra_inputs'
# OLD layout (18): the last two (e/gamma MVAs) no longer exist in CMSSW >= 15_1_0_pre4 (always 0) -> dropped when reading
track_feat_names_old = ["PuppiWeight", "Vx", "Vy", "Vz", "d0", "z0", "TrkMVA1", "TrkMVA2", "TrkMVA3", "CaloEta", "CaloPhi", "chi2", "nStubs", "nPar", "hOverE", "pfClusterPt", "egVsPionMVAOut", "egVsPUMVAOut"]
# NEW layout (17): slot 14 = H/E (-1 = no EM energy, -2 = no cluster), slot 15 = cluster pT, slot 16 = has_cluster (0/1)
track_feat_names = track_feat_names_old[:16] + ["hasCluster"]
N_TRACK_OLD, N_TRACK_NEW = 18, 17
TRACK_PT_IDX = track_feat_names.index("pfClusterPt")   # cluster pT slot (15) -> divided by tau pT when pt_norm is on

# hgcal_feat_names: HGCal shower features (endcap)
# NOTE: old ntuples store 21 numbers (ecal_shape slots 0-20 for endcap taus) but this list only has 20 names,
# so one name is missing (check the ntupler). The last two slots track the cluster pT (mipPt, sumPt).
hgcal_feat_names = ["ShowerLength", "CoreShowerLength", "FirstLayer", "MaxLayer", "eMax", "sigmaEtaEtaMax", "sigmaPhiPhiMax", "sigmaEtaEtaTot", "sigmaPhiPhiTot",
                    "sigmaZZ", "sigmaRRTot", "sigmaRRMax", "sigmaRRMean", "zBaryCenter", "layer10percent", "layer90percent", "triggerCells67percent", "triggerCells90percent",
                    "mipPt", "sumPt"]
N_HGCAL_OLD = 21          # number of HGCal numbers in the old ecal_shape branch (checked on the VBF test ntuple)
HGCAL_PT_IDXS = [-2, -1]  # mipPt, sumPt = last two HGCal slots -> divided by tau pT when pt_norm is on

# ecal/hcal cell names: ECal 3x15 = 45 cells, HCal 3x3 = 9 cells per particle (barrel)
ecal_feat_names = []
hcal_feat_names = []
for i in range(3):
    for j in range(15):
        if(j<3): hcal_feat_names.append("HCalCell_%i_%i" % (i, j))
        ecal_feat_names.append("ECalCell_%i_%i" % (i,j))
N_ECAL, N_HCAL = 45, 9
ECAL_IMG, HCAL_IMG = (3, 15), (3, 3)
# "one ecal cell is buggy..." (original author): these two cells are NOT divided by pT in pt_norm
ECAL_BUGGY_CELLS = [ecal_feat_names.index("ECalCell_1_7"), ecal_feat_names.index("ECalCell_0_13")]


# ==================== DATA LOADING ====================
def _clean(a, name):
    """Replace NaN / inf / huge (>1e9) values by 0 and say how many were replaced."""
    bad = ~np.isfinite(a) | (np.abs(np.nan_to_num(a)) > 1e+9)
    if bad.any():
        print("  %s: replaced %i NaN/inf/huge values with 0" % (name, int(bad.sum())))
        a = a.copy()
        a[bad] = 0.
    return a


def apply_pt_norm(d):
    """pt_norm: divide every pT-like input by the tau candidate's raw (uncorrected) L1 pT ('pt' branch),
    so inputs become fractions of the tau pT. Rows with pt <= 0 (invalid rows) are left unchanged."""
    pt = d["pt"]
    scale = np.where(pt > 0, 1.0 / np.where(pt > 0, pt, 1.0), 1.0).astype(np.float32)[:, None]

    d["X"][:, :, 0] *= scale                          # particle pT
    d["track_extra"][:, :, TRACK_PT_IDX] *= scale     # linked cluster pT
    ecal_idx = [i for i in range(d["gct_ecal"].shape[-1]) if i not in ECAL_BUGGY_CELLS]
    d["gct_ecal"][:, :, ecal_idx] *= scale[:, :, None]  # ECal cell energies (except the 2 buggy cells)
    d["gct_hcal"] *= scale[:, :, None]                  # HCal cell energies
    d["hgcal"][:, :, HGCAL_PT_IDXS] *= scale[:, :, None]  # HGCal mipPt, sumPt
    return d


def load_sample(path, pt_norm=False):
    """Read ONE ntuple file -> dict of per-row arrays (no cuts, no split).
    Works with both ntuple layouts:
      old: extra_inputs (10x18), ecal_shape (10x45: ECal cells in the barrel, HGCal in the endcap), hcal_shape (10x9)
      new: extra_inputs (10x17), gct_ecal (10x45), gct_hcal (10x9), hgcal (10xN)
    Output always has the same keys: track_extra, gct_ecal, gct_hcal, hgcal (zeros where a block doesn't apply)."""
    tree = uproot.open(path)[TREE]
    keys = set(tree.keys())
    new_calo = "gct_ecal" in keys
    calo_branches = ["gct_ecal", "gct_hcal", "hgcal"] if new_calo else ["ecal_shape", "hcal_shape"]
    optional = [b for b in ["ptcorr"] if b in keys]  # NN-corrected pT from the CMSSW model (for comparison)
    branches = ["run", "lumi", "event", "pt", "eta", "genpt1", "gendr1", "m_inputs", "extra_inputs"] + calo_branches + optional
    a = tree.arrays(branches, library="np")
    n = len(a["pt"])
    print("Loaded %s: %i rows (%s calo layout)" % (os.path.basename(path), n, "new" if new_calo else "old"))

    d = {
        "pt":      a["pt"].astype(np.float32),        # raw L1 tau pT (no NN correction)
        "ptcorr":  a["ptcorr"].astype(np.float32) if "ptcorr" in a else np.full(n, np.nan, np.float32),
        "eta":     a["eta"].astype(np.float32),
        "gen_pt":  a["genpt1"].astype(np.float32),
        "gendr":   a["gendr1"].astype(np.float32),    # -999 if no gen tau matched
        "event_id": np.stack([a["run"], a["lumi"], a["event"]], axis=1).astype(np.uint64),
        "X":       _clean(a["m_inputs"].reshape(n, N_PARTS, N_BASE).astype(np.float32), "m_inputs"),
    }

    # --- track features: old layout (18) -> drop the 2 dead e/gamma MVAs; new layout (17) -> keep all ---
    trk = a["extra_inputs"].reshape(n, N_PARTS, -1).astype(np.float32)
    if trk.shape[-1] == N_TRACK_OLD:
        trk = trk[:, :, :16]
    elif trk.shape[-1] != N_TRACK_NEW:
        raise ValueError("extra_inputs has %i features per particle; expected %i (old) or %i (new)" % (trk.shape[-1], N_TRACK_OLD, N_TRACK_NEW))
    d["track_extra"] = _clean(trk, "extra_inputs")

    # --- calorimeter blocks ---
    if new_calo:
        d["gct_ecal"] = a["gct_ecal"].reshape(n, N_PARTS, -1).astype(np.float32)
        d["gct_hcal"] = a["gct_hcal"].reshape(n, N_PARTS, -1).astype(np.float32)
        d["hgcal"]    = a["hgcal"].reshape(n, N_PARTS, -1).astype(np.float32)
    else:
        # old layout: split by the tau's |eta| (barrel -> GCT cells, endcap -> HGCal numbers)
        ecal = a["ecal_shape"].reshape(n, N_PARTS, N_ECAL).astype(np.float32)
        hcal = a["hcal_shape"].reshape(n, N_PARTS, N_HCAL).astype(np.float32)
        barrel = (np.abs(d["eta"]) < BARREL_ETA)[:, None, None]
        d["gct_ecal"] = np.where(barrel, ecal, 0.).astype(np.float32)
        d["gct_hcal"] = np.where(barrel, hcal, 0.).astype(np.float32)
        d["hgcal"]    = np.where(barrel, 0., ecal[:, :, :N_HGCAL_OLD]).astype(np.float32)
    for k in ["gct_ecal", "gct_hcal", "hgcal"]:
        d[k] = _clean(d[k], k)

    if pt_norm:
        d = apply_pt_norm(d)
    return d


def count_events(path, n_events=None):
    """Total number of processed events in a file (incl. events with no tau candidate), for the trigger rate.
    Priority: n_events argument > unique (run, lumi, event) in EVT_TREE > unique events in the tau tree (warns)."""
    if n_events:
        return int(n_events)
    f = uproot.open(path)
    tree = f[EVT_TREE] if EVT_TREE in f else f[TREE]
    if EVT_TREE not in f:
        print("WARNING: %s has no %s; counting only events with a tau candidate -> rate is OVERestimated" % (path, EVT_TREE))
    a = tree.arrays(["run", "lumi", "event"], library="np")
    ids = np.stack([a["run"], a["lumi"], a["event"]], axis=1).astype(np.uint64)
    return len(np.unique(ids, axis=0))


def split_mask(event_id, split, train_frac, val_frac, test_frac, seed):
    """Random train/val/test split BY EVENT (all candidates of one event land in the same set).
    Same seed + fractions -> same split in training.py, plot.py and predict.py."""
    if train_frac + val_frac + test_frac > 1. + 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must be <= 1")
    _, inv = np.unique(event_id, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    r = np.random.default_rng(seed).random(inv.max() + 1)[inv]  # one random number per event
    lo, hi = {"train": (0., train_frac),
              "val":   (train_frac, train_frac + val_frac),
              "test":  (train_frac + val_frac, train_frac + val_frac + test_frac)}[split]
    return (r >= lo) & (r < hi)


def select_mask(d, is_sig, pt_cut, pt_max, eta_min, eta_max):
    """Selection cuts on raw L1 pT / |eta|; signal must also match a gen tau within dR < 0.4."""
    aeta = np.abs(d["eta"])
    m = (d["pt"] > pt_cut) & (d["pt"] < pt_max) & (aeta >= eta_min) & (aeta < eta_max)
    if is_sig:
        m &= np.abs(d["gendr"]) < 0.4
    return m


# create_train_test_data: reads sig/bkg/qcd ntuples, splits them by event, applies the cuts,
# labels sig=1 / bkg,qcd=0, and returns a dict of arrays.
# split='train' -> train part of sig & bkg + ALL of qcd (qcd is used for training only)
# split='val' / 'test' -> that part of sig & bkg (no qcd)
def create_train_test_data(dir_path, split="train", sig_file="VBFHToTauTau_PU200_1-1_emseed_sig_test.root",
        bkg_file="MinBias_PU200_1-1_emseed_bkg_test.root", qcd_file="QCD_Pt15To3000_PU200_1-1_emseed_qcd_test.root",
        train_frac=0.56, val_frac=0.14, test_frac=0.30, seed=1111,
        eta_min=0., eta_max=TAU_ETA_MAX, pt_cut=15.0, pt_max=9999.0, pt_norm=False,
        train=None, track_extra_inputs=None, calo_extra_inputs=None):
    # old interface (still used by plot_features.py): train=True -> 'train', train=False -> 'test'; the other two are ignored
    if train is not None:
        split = "train" if train else "test"

    samples = [("sig", sig_file, 1.), ("bkg", bkg_file, 0.)]
    if split == "train":
        samples.append(("qcd", qcd_file, 0.))

    keys = ["X", "track_extra", "gct_ecal", "gct_hcal", "hgcal", "pt", "ptcorr", "eta", "gen_pt", "event_id"]
    parts = {k: [] for k in keys + ["label", "sample"]}
    effs = {}
    for code, (name, fname, label) in enumerate(samples):
        d = load_sample(os.path.join(dir_path, fname), pt_norm=pt_norm)
        in_split = np.ones(len(d["pt"]), bool) if name == "qcd" else \
                   split_mask(d["event_id"], split, train_frac, val_frac, test_frac, seed)
        sel = select_mask(d, name == "sig", pt_cut, pt_max, eta_min, eta_max)
        keep = in_split & sel
        effs[name] = sel[in_split].mean() if in_split.any() else 0.  # fraction of candidates passing the cuts
        print("  %s (%s split): %i rows in split, %i pass cuts" % (name, split, in_split.sum(), keep.sum()))
        for k in keys:
            parts[k].append(d[k][keep])
        parts["label"].append(np.full(keep.sum(), label, np.float32))
        parts["sample"].append(np.full(keep.sum(), code, np.int8))  # 0 = sig, 1 = minbias, 2 = qcd

    out = {k: np.concatenate(v) for k, v in parts.items()}
    n_sig, n_bkg = int((out["label"] == 1).sum()), int((out["label"] == 0).sum())
    print("Nsig, Nbkg (%s): %i %i" % (split, n_sig, n_bkg))
    if n_sig == 0 or n_bkg == 0:  # T8
        raise ValueError("No %s candidates pass the cuts in the %s split; loosen the cuts or use more events"
                         % ("signal" if n_sig == 0 else "background", split))

    # pT-correction target = gen pT / raw L1 pT for signal; 1 (placeholder, weight 0 in training) for background
    is_sig = out["label"] == 1
    y_pt = np.ones_like(out["pt"])
    y_pt[is_sig] = out["gen_pt"][is_sig] / out["pt"][is_sig]

    return {
        'X': out["X"],
        'track_extra': out["track_extra"],
        'gct_ecal': out["gct_ecal"],
        'gct_hcal': out["gct_hcal"],
        'hgcal': out["hgcal"],

        'reco_pt': out["pt"],          # raw L1 tau pT
        'ptcorr': out["ptcorr"],       # CMSSW-NN-corrected pT (nan if the branch is missing)
        'gen_pt': out["gen_pt"],
        'eta': out["eta"],
        'event_id': out["event_id"],
        'sample': out["sample"],
        'Y_jetID': out["label"],
        'Y_pt_corr': y_pt,
        'sig_eff': effs["sig"],
        'bkg_eff': effs["bkg"],
    }


# ==================== MODEL INPUTS + FLAG CHECKS ====================
def region_of(eta_min, eta_max):
    """barrel only / endcap only / both (range crosses |eta| = 1.5)."""
    if eta_max <= BARREL_ETA:
        return "barrel"
    if eta_min >= BARREL_ETA:
        return "endcap"
    return "both"


def check_config(cfg):
    """Raise an error for setups that can't work; warn for setups that work but waste inputs."""
    m, calo, hgcal = cfg["model_fn"], cfg["calo_extra"], cfg["hgcal_extra"]
    if cfg["eta_min"] >= cfg["eta_max"]:
        raise ValueError("eta_min must be smaller than eta_max")
    region = region_of(cfg["eta_min"], cfg["eta_max"])

    if m in PARTICLE_MODELS:
        if region == "barrel" and hgcal and not calo:
            raise ValueError("Barrel only + hgcal_extra alone: HGCal is empty in the barrel, so the model gets no calo info. Use --calo_extra.")
        if region == "endcap" and calo and not hgcal:
            raise ValueError("Endcap only + calo_extra alone: GCT ECal/HCal are empty in the endcap, so the model gets no calo info. Use --hgcal_extra.")
        if region == "barrel" and hgcal:
            print("WARNING: barrel only -> hgcal_extra adds all-zero columns (allowed, but useless)")
        if region == "endcap" and calo:
            print("WARNING: endcap only -> calo_extra adds 54 all-zero columns (allowed, but useless)")
        if region == "both" and calo != hgcal:
            print("WARNING: range crosses |eta| = 1.5 with only one calo flag -> taus on the other side get no calo info")
        if m in MERGED_MODELS and (cfg["track_extra"] or calo or hgcal or cfg["pt_norm"] or cfg.get("input_norm")):
            print("WARNING: merged model with extra features / pt_norm / input_norm no longer matches the firmware (8 raw base features, no normalization)")

    else:  # image models: barrel calo info enters as the image
        if region == "endcap":
            raise ValueError("%s needs a calo image, which only exists in the barrel; use eta_max <= 1.5 (or a range crossing 1.5 for deepset_cnn)" % m)
        if m == "cnn" and region == "both":
            raise ValueError("cnn has no particle inputs, so endcap taus would see only an all-zero image; use eta_max <= 1.5")
        if calo:
            print("WARNING: calo_extra is ignored for %s (barrel calo info enters through the --image)" % m)
        if m == "cnn" and (cfg["track_extra"] or hgcal):
            print("WARNING: track_extra / hgcal_extra are ignored for cnn (it only sees the image)")
        if m == "deepset_cnn" and region == "barrel" and hgcal:
            print("WARNING: barrel only -> hgcal_extra adds all-zero columns (allowed, but useless)")
        if m == "deepset_cnn" and region == "both" and not hgcal:
            print("WARNING: range crosses |eta| = 1.5 without hgcal_extra -> endcap taus get no calo info (their image is all zeros)")


def make_image(d, image):
    """Calo image per particle, channels first: ECal (N, 10, 3, 15) or HCal (N, 10, 3, 3). All zeros in the endcap."""
    if image == "ecal":
        return d["gct_ecal"].reshape(-1, N_PARTS, *ECAL_IMG)
    return d["gct_hcal"].reshape(-1, N_PARTS, *HCAL_IMG)


def build_inputs(d, cfg):
    """Glue the requested feature blocks onto each particle and shape them for the model.
    Used by training.py, plot.py and predict.py, so the inputs are always built the same way."""
    m = cfg["model_fn"]
    if m == "cnn":
        return make_image(d, cfg["image"])

    blocks = [d["X"]]                                   # 8 base features
    if cfg["track_extra"]:
        blocks.append(d["track_extra"])                 # +16 (old) / +17 (new)
    if cfg["calo_extra"] and m in PARTICLE_MODELS:
        blocks += [d["gct_ecal"], d["gct_hcal"]]        # +45 +9 (barrel)
    if cfg["hgcal_extra"]:
        blocks.append(d["hgcal"])                       # +21 (old) / +N (new) (endcap)
    X = np.concatenate(blocks, axis=-1)

    if m == "deepset_cnn":
        return [X, make_image(d, cfg["image"])]         # two inputs: particles + image
    if m in FLAT_MODELS:
        return X.reshape(X.shape[0], -1)                # (N, 10*F), particle by particle (same order as the firmware)
    return X                                            # deep set: (N, 10, F)


def model_paths(model_out):
    """Files saved next to a model: its config (json) and, for class_model, its input scaler."""
    base = model_out[:-3] if model_out.endswith(".h5") else model_out
    return base + "_config.json", base + "_scaler.pkl"


def load_config(model_path):
    cfg_path, _ = model_paths(model_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("%s not found: the model needs its config file (written by training.py) to rebuild its inputs" % cfg_path)
    with open(cfg_path) as f:
        return json.load(f)


def split_outputs(preds):
    """Model output -> (ID score (N,), pT factor (N,) or None)."""
    if isinstance(preds, list):
        return preds[0].ravel(), preds[1].ravel()
    return preds.ravel(), None


# ==================== MODELS (output: ID score 0=fake, 1=tau; + pT correction factor if pt_regress) ====================
@tf.keras.utils.register_keras_serializable(package="L1NNTau")
class MaskedAverage(tf.keras.layers.Layer):
    """Deep-set average over particles that SKIPS empty slots (M3).
    A slot is empty when its particle pT (feature 0 of the raw input) is 0, the same rule as the firmware.
    Inputs: [per-particle features (N, 10, C), raw model input (N, 10, F)] -> output (N, C).
    Registered above, so saved models load as long as utils is imported. (Not an hls4ml layer: the deep set
    isn't in the firmware; use --no-mask_empty for a plain average if you ever need to convert it.)
    add_count=True also outputs the number of real particles / 10 (output (N, C+1)): the masked average alone
    no longer tells the network how many particles the candidate has, and that multiplicity matters (taus have few)."""
    def __init__(self, add_count=True, **kwargs):
        super().__init__(**kwargs)
        self.add_count = add_count

    def call(self, inputs):
        feats, raw = inputs
        mask = tf.cast(tf.not_equal(raw[:, :, 0], 0.), feats.dtype)[:, :, None]  # 1 = real particle, 0 = empty
        count = tf.reduce_sum(mask, axis=1)                                       # number of real particles, (N, 1)
        mean = tf.reduce_sum(feats * mask, axis=1) / tf.maximum(count, 1.)        # avoid 0/0 if a candidate has no particles
        if self.add_count:
            return tf.concat([mean, count / float(N_PARTS)], axis=-1)
        return mean

    def get_config(self):  # so add_count is restored when the model is loaded
        return {**super().get_config(), "add_count": self.add_count}


# class_model: plain float dense net on flattened inputs (10*F,) -> 32-16-16-8 -> sigmoid; uses dropout
def class_model(input_shape = None, pt_regress = False):
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

    if pt_regress:  # pT head: correction factor (gen pT / L1 pT)
        pT_branch = tf.keras.layers.Dense(1, name='pT_output')(main_branch)
        return tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])

    model = tf.keras.Model(inputs = inputs, outputs = jetID_branch)

    return model

# class_model_quant: quantized DEEP SET. Same small net applied to each of the 10 particles (Conv1D k=1),
# averaged over particles (order-independent), then 32-16-1 classifier. 9-bit weights for FPGA (hls4ml).
def class_model_quant(input_shape=None, output_shape=[1], bits=9, bits_int=2, alpha_val=1, pt_regress=False, mask_empty=True, add_count=True):

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
    if mask_empty:  # average over real particles only (empty, zero-padded slots are skipped) [+ particle count]
        main = MaskedAverage(add_count=add_count, name='avgpool')([main, inputs])
    else:           # plain average over all 10 slots (old behavior)
        main = GlobalAveragePooling1D(name='avgpool')(main)

    #Now split into jet ID and pt regression

    #jetID branch, 3 layer MLP
    jet_id = QDense(32, name='Dense_1_jetID', **common_args)(main)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_1_jetID')(jet_id)

    jet_id = QDense(16, name='Dense_2_jetID', **common_args)(jet_id)
    jet_id = QActivation(activation=quantized_relu(bits), name='relu_2_jetID')(jet_id)

    jet_id = QDense(output_shape[0], name='Dense_3_jetID', **common_args)(jet_id)
    jet_id = Activation('sigmoid', name='jet_id_output')(jet_id)

    if pt_regress:
        #pT regression branch (shares the deep-set summary 'main' with the ID head)
        pt_regress_out = QDense(10, name='Dense_1_pT', **common_args)(main)
        pt_regress_out = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress_out)

        pt_regress_out = QDense(1, name='pT_output',
                            kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            kernel_initializer='lecun_uniform')(pt_regress_out)

        #Define the model using both branches
        return tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress_out])

    model = tf.keras.Model(inputs = inputs, outputs = jet_id)

    return model

# deepset_cnn: deep set on particles + quantized CNN on calo shower images (ECal 3x15 or HCal 3x3 per particle),
# both branches concatenated -> 32-16-1 classifier. Two inputs: [particles, images].
def deepset_cnn(point_shape=None, cnn_shape=None, output_shape=[1], bits=9, bits_int=2, alpha_val=1, pt_regress=False, mask_empty=True, add_count=True):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform',
    }

    #Initialize inputs
    cnn_inputs = tf.keras.layers.Input(shape=cnn_shape, name='cnn_input')

    #cnn_main branch (axis=1: normalize per particle channel, since the image is channels_first; M4)
    cnn_main = BatchNormalization(axis=1, name='norm_input_cnn')(cnn_inputs)

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
    if mask_empty:  # average over real particles only (empty, zero-padded slots are skipped) [+ particle count]
        main = MaskedAverage(add_count=add_count, name='avgpool2')([main, inputs])
    else:           # plain average over all 10 slots (old behavior)
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

    if pt_regress:
        #pT regression branch
        pt_regress_out = QDense(10, name='Dense_1_pT', **common_args)(main)
        pt_regress_out = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress_out)

        pt_regress_out = QDense(1, name='pT_output',
                            kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            kernel_initializer='lecun_uniform')(pt_regress_out)

        #Define the model using both branches
        return tf.keras.Model(inputs = [inputs, cnn_inputs], outputs = [jet_id, pt_regress_out])

    model = tf.keras.Model(inputs = [inputs, cnn_inputs], outputs = jet_id)

    return model




# cnn: quantized 2D CNN on calo shower images only (input (10, 3, 3) HCal or (10, 3, 15) ECal, channels_first) -> 32-16-1
def cnn(input_shape=None, output_shape=[1], bits=9, bits_int=2, alpha_val=1, pt_regress=False):

    # Define a dictionary for common arguments
    common_args = {
        'kernel_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'bias_quantizer': quantized_bits(bits, bits_int, alpha=alpha_val),
        'kernel_initializer': 'lecun_uniform',
    }

    #Initialize inputs
    inputs = tf.keras.layers.Input(shape=input_shape, name='model_input')

    #Main branch (axis=1: normalize per particle channel, since the image is channels_first; M4)
    main = BatchNormalization(axis=1, name='norm_input')(inputs)

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

    if pt_regress:
        #pT regression branch
        pt_regress_out = QDense(10, name='Dense_1_pT', **common_args)(main)
        pt_regress_out = QActivation(activation=quantized_relu(bits), name='relu_1_pt')(pt_regress_out)

        pt_regress_out = QDense(1, name='pT_output',
                            kernel_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            bias_quantizer=quantized_bits(16, 6, alpha=alpha_val),
                            kernel_initializer='lecun_uniform')(pt_regress_out)

        #Define the model using both branches
        return tf.keras.Model(inputs = inputs, outputs = [jet_id, pt_regress_out])

    model = tf.keras.Model(inputs = inputs, outputs = jet_id)

    return model




# merged_model: float dense net on flattened inputs (default 80 = 10 particles x 8) -> 25-25-15-15-10
# outputs: jet ID score (+ pT correction if pt_regress)
def merged_model(input_size = 80, pt_regress = True, input_norm = False):

    inputs = tf.keras.layers.Input(shape=(input_size,), name='model_input')

    x = inputs
    # M2 (ON HOLD, check with advisor): input_norm=True normalizes the raw inputs first. This ADDS a layer,
    # so the firmware (which has no such layer) would no longer match. Default off = firmware model.
    if input_norm:
        x = BatchNormalization(name='norm_input')(x)

    main_branch = tf.keras.layers.Dense(25, activation = "ReLU", name='Dense1')(x)
    main_branch = tf.keras.layers.Dense(25, activation = "ReLU", name='Dense2')(main_branch)
    main_branch = tf.keras.layers.Dense(15, activation = "ReLU", name='Dense3')(main_branch)
    main_branch = tf.keras.layers.Dense(15, activation = "ReLU", name='Dense4')(main_branch)
    main_branch = tf.keras.layers.Dense(10, activation = "ReLU", name='Dense5')(main_branch)

    jetID_branch = tf.keras.layers.Dense(1, activation='sigmoid', name='jetID_output')(main_branch)

    if not pt_regress:
        return tf.keras.Model(inputs = inputs, outputs = jetID_branch)

    pT_branch = tf.keras.layers.Dense(1, name='pT_output')(main_branch)

    model = tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])

    return model

# quantized_merged_model: 9-bit version of merged_model (ID + pT correction).
# Same structure as the NN in the firmware / CMSSW (80 inputs, 2 outputs) when run with the 8 base features.
def quantized_merged_model(input_size = 80, pt_regress = True, input_norm = False):

    bits = 9
    bits_int = 2
    alpha_val = 1

    inputs = tf.keras.layers.Input(shape=(input_size,), name='model_input')

    x = inputs
    # M2 (ON HOLD, check with advisor): input_norm=True normalizes the raw inputs first. This ADDS a layer,
    # so the firmware (which has no such layer) would no longer match. Default off = firmware model.
    if input_norm:
        x = BatchNormalization(name='norm_input')(x)

    #kernel_initializer = initializers.lecun_uniform(seed)

    main_branch = QDense(25, name='Dense_1',
            kernel_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                bits, bits_int, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(x)  # , kernel_regularizer=l1(0.0001)))
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

    if not pt_regress:
        return tf.keras.Model(inputs = inputs, outputs = jetID_branch)

    pT_branch = QDense(1, name='pT_output',
            kernel_quantizer=quantized_bits(
                16, 6, alpha=alpha_val),
            bias_quantizer=quantized_bits(
                16, 6, alpha=alpha_val),
            kernel_initializer='lecun_uniform')(main_branch)

    model = tf.keras.Model(inputs = inputs, outputs = [jetID_branch, pT_branch])


    return model

# ==================== TRAINING WEIGHTS ====================
def train_weights(y_train, pt, class_labels, mode="balance"):
    """
    Per-candidate training weights, binned in the raw L1 tau pT.
      balance: in every pT bin, sig and bkg get the same total weight (so pT alone can't separate them)
      flat:    balance + every pT bin gets the same total weight (high-pT bins count as much as low-pT ones)
      none:    no pT weighting; only the overall sig/bkg balance (signal weighted by Nbkg/Nsig)
    Weights are rescaled so their mean is 1.
    """
    num_samples = y_train.shape[0]

    if mode == "none":
        sample_weights = np.ones(num_samples)
        nSig = np.sum(y_train)
        nBkg = num_samples - nSig
        print("Sig rw %.2f" % (nBkg/nSig))
        sample_weights[y_train > 0.9] = nBkg / nSig  # weight signals relative to the background
        return sample_weights / np.mean(sample_weights)

    sample_weights = np.ones(num_samples)

    # pT bin edges (start at 0 so every candidate falls in a bin; U7)
    pt_bins = np.array([
        0, 15, 17, 19, 22, 25, 30, 35, 40, 45, 50,
        60, 76, 97, 122, 154, 195, 246, 311,
        393, 496, 627, 792, np.inf  # Use np.inf to cover all higher values
    ])
    n_bins = len(pt_bins) - 1

    # Calculate counts per class per pT bin
    class_pt_counts = {}
    for label, idx in class_labels.items():
        class_mask = y_train == idx
        class_pt_counts[idx], _ = np.histogram(pt[class_mask], bins=pt_bins)

    # Compute the maximum counts per pT bin over all classes
    max_counts_per_bin = np.zeros(n_bins)
    for bin_idx in range(n_bins):
        counts_in_bin = [class_pt_counts[idx][bin_idx] for idx in class_labels.values()]
        max_counts_per_bin[bin_idx] = max(counts_in_bin)

    # Compute weights per class per pT bin
    #   balance: each class totals max_count in the bin -> sig = bkg in every bin
    #   flat:    each class totals 1 in the bin -> sig = bkg AND all bins equal
    weights_per_class_pt_bin = {}
    for idx in class_labels.values():
        weights_per_class_pt_bin[idx] = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            class_count = class_pt_counts[idx][bin_idx]
            if class_count == 0:
                weights_per_class_pt_bin[idx][bin_idx] = 0.
            elif mode == "flat":
                weights_per_class_pt_bin[idx][bin_idx] = 1. / class_count
            else:
                weights_per_class_pt_bin[idx][bin_idx] = max_counts_per_bin[bin_idx] / class_count

    # Assign weights to samples
    for idx in class_labels.values():
        class_mask = y_train == idx
        sample_indices = np.where(class_mask)[0]
        bin_indices = np.digitize(pt[class_mask], pt_bins) - 1  # Subtract 1 to get 0-based index
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)       # Handle edges
        sample_weights[sample_indices] = weights_per_class_pt_bin[idx][bin_indices]

    # rescale so the mean weight is 1 (keeps loss on a normal scale)
    sample_weights = sample_weights / np.mean(sample_weights)

    return sample_weights
