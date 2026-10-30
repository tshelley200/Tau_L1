import os
import re
import argparse
from utils import *

# NN score cuts scanned for the rate curve (fine steps near 1, where the useful working points are)
TAU_SCORE_EDGES = [round(i,2) for i in np.arange(0, 0.8, 0.01).tolist()]+\
                  [round(i,4) for i in np.arange(0.9, 1, 0.0001)] + [1]


def parse_args():
    p = argparse.ArgumentParser(description="ROC, trigger-rate and pT-response plots on the test split")
    # models to compare (each needs the <name>_config.json written by training.py)
    p.add_argument("--models", nargs="+", default=["/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/models/oct_22/sample_class_quant_track_extra_endcap.h5"])
    p.add_argument("--labels", nargs="+", default=["Deepset track extra"])

    p.add_argument("--data_dir", default="/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/L1NNTau_data/")
    p.add_argument("--sig_file", default=None)  # default: same file as in the first model's config
    p.add_argument("--bkg_file", default=None)
    p.add_argument("--plot_dir", default="../plots/")

    # test-set cuts and split; default None = take them from the first model's config (so they match training)
    for name in ["pt_cut", "pt_max", "eta_min", "eta_max", "train_frac", "val_frac", "test_frac"]:
        p.add_argument("--" + name, type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    # total MinBias events processed; default: counted from the 'ntuple/tree' of the bkg file
    p.add_argument("--n_minbias_events", type=int, default=None)
    return p.parse_args()


def rate_curve(score_sig, score_bkg, bkg_event_id, n_events):
    """Signal efficiency (per tau) and trigger rate (per EVENT: an event fires if ANY of its candidates passes)."""
    eff = np.array([(score_sig > t).mean() for t in TAU_SCORE_EDGES])
    if len(score_bkg):
        _, inv = np.unique(bkg_event_id, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        ev_max = np.full(inv.max() + 1, -np.inf)
        np.maximum.at(ev_max, inv, score_bkg)  # highest score in each event
    else:
        ev_max = np.array([])
    rate = np.array([(ev_max > t).sum() for t in TAU_SCORE_EDGES]) / n_events * MINBIAS_RATE_KHZ
    return eff, rate


def cms_labels():
    hep.cms.text("Phase 2 Simulation")
    try:
        hep.cms.lumitext("PU 200 (14 TeV)")
    except AttributeError:  # mplhep >= 1.0 removed lumitext
        pass


def main(args):
    if len(args.models) != len(args.labels):
        raise ValueError("--models and --labels must have the same length")
    cfgs = [load_config(m) for m in args.models]

    # --- test settings: command line, else the first model's config ---
    ref = cfgs[0]
    for k in ["pt_cut", "pt_max", "eta_min", "eta_max", "train_frac", "val_frac", "test_frac", "seed", "sig_file", "bkg_file"]:
        if getattr(args, k) is None:
            setattr(args, k, ref[k])
    for m, c in zip(args.models, cfgs):
        for k in ["train_frac", "val_frac", "test_frac", "seed", "sig_file", "bkg_file"]:
            if c[k] != getattr(args, k):
                print("WARNING: %s was trained with %s = %s (test uses %s); its test set may overlap its training set" % (m, k, c[k], getattr(args, k)))
    os.makedirs(args.plot_dir, exist_ok=True)

    # --- load the test split (sig + MinBias), without and (if needed) with pt_norm ---
    load_kw = dict(sig_file=args.sig_file, bkg_file=args.bkg_file, train_frac=args.train_frac, val_frac=args.val_frac,
                   test_frac=args.test_frac, seed=args.seed, eta_min=args.eta_min, eta_max=args.eta_max,
                   pt_cut=args.pt_cut, pt_max=args.pt_max)
    data = {False: create_train_test_data(args.data_dir, split="test", pt_norm=False, **load_kw)}
    if any(c["pt_norm"] for c in cfgs):
        data[True] = create_train_test_data(args.data_dir, split="test", pt_norm=True, **load_kw)
    test = data[False]
    y, aeta = test['Y_jetID'], np.abs(test['eta'])

    # MinBias events in the test split = all processed events x test fraction (events without a candidate can't fire)
    n_test_events = count_events(os.path.join(args.data_dir, args.bkg_file), args.n_minbias_events) * args.test_frac
    print("MinBias events in the test split: %.0f" % n_test_events)

    # --- regions: always 'all'; barrel and endcap separately if the range crosses |eta| = 1.5 ---
    regions = [("all", np.ones_like(y, bool), args.eta_min, args.eta_max)]
    if region_of(args.eta_min, args.eta_max) == "both":
        regions += [("barrel", aeta < BARREL_ETA, args.eta_min, BARREL_ETA),
                    ("endcap", aeta >= BARREL_ETA, BARREL_ETA, args.eta_max)]

    # --- run every model on the test set ---
    scores, pt_factors = [], []
    for path, label, cfg in zip(args.models, args.labels, cfgs):
        X_test = build_inputs(data[cfg["pt_norm"]], cfg)
        if cfg["model_fn"] == "class_model":
            X_test = joblib.load(model_paths(path)[1]).transform(X_test)
        model = load_qmodel(path, compile=False)
        id_score, pt_factor = split_outputs(model.predict(X_test, batch_size = 256))
        scores.append(id_score)
        pt_factors.append(pt_factor)

        # save test-set predictions for your own plots
        npz = os.path.join(args.plot_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", label) + "_test.npz")
        corr = test['reco_pt'] * pt_factor if pt_factor is not None else np.full_like(id_score, np.nan)
        np.savez(npz, id_score=id_score, pt_factor=pt_factor if pt_factor is not None else np.full_like(id_score, np.nan),
                 corrected_pt=corr, reco_pt=test['reco_pt'], ptcorr_cmssw=test['ptcorr'], gen_pt=test['gen_pt'],
                 eta=test['eta'], label=y, event_id=test['event_id'])
        print("Saved", npz)

    for rname, rmask, lo, hi in regions:
        # --- ROC ---
        plt.figure()
        for score, label in zip(scores, args.labels):
            ym, sm = y[rmask], score[rmask]
            if len(np.unique(ym)) < 2:
                print("Skipping ROC (%s) for %s: needs both sig and bkg" % (rname, label))
                continue
            fpr, tpr, _ = roc_curve(ym, sm)
            auc_ = auc(fpr, tpr)
            print("%s [%s] AUC %.3f" % (label, rname, auc_))
            plt.plot(fpr, tpr, label="%s (AUC = %.3f)" % (label, auc_))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(r"$%g \leq |\eta| < %g$" % (lo, hi))
        plt.legend(loc="best")
        plt.savefig(os.path.join(args.plot_dir, "roc_%s.png" % rname))
        plt.close()

        # --- trigger rate (per event) vs signal efficiency ---
        plt.figure()
        plt.plot([], [], 'none', label=r'MinBias, $p_T^{L1} > %g$ GeV, $%g \leq |\eta| < %g$' % (args.pt_cut, lo, hi))
        bkg = rmask & (y == 0)
        for score, label in zip(scores, args.labels):
            eff, rate = rate_curve(score[rmask & (y == 1)], score[bkg], test['event_id'][bkg], n_test_events)
            plt.plot(eff, rate, linewidth=5, label=label)
        cms_labels()
        plt.ylabel(r'Single $\tau_h$ Trigger Rate [kHz]')
        plt.xlabel(r'$\tau_h$ efficiency ($p_T^{L1} > %g$ GeV)' % args.pt_cut)
        plt.ylim([0.1, 1e3])
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(args.plot_dir, "rate_%s.png" % rname))
        plt.close()

        # --- pT response for signal: L1 pT / gen pT before and after the correction ---
        sig = rmask & (y == 1)
        if any(f is not None for f in pt_factors) and sig.any():
            plt.figure()
            curves = [("L1 (raw)", test['reco_pt'][sig])]
            if np.isfinite(test['ptcorr']).any():
                curves.append(("CMSSW NN (ptcorr)", test['ptcorr'][sig]))
            curves += [(label, test['reco_pt'][sig] * f[sig]) for f, label in zip(pt_factors, args.labels) if f is not None]
            for name, pt_vals in curves:
                r = pt_vals / test['gen_pt'][sig]
                q16, q50, q84 = np.percentile(r, [16, 50, 84])
                plt.hist(r, bins=60, range=(0, 3), histtype='step', density=True, linewidth=3,
                         label="%s (median %.2f, res. %.2f)" % (name, q50, (q84 - q16) / 2 / q50))
            plt.xlabel(r"$p_T / p_T^{gen}$")
            plt.ylabel("a.u.")
            plt.title(r"$%g \leq |\eta| < %g$" % (lo, hi))
            plt.legend(loc="upper right", fontsize="small")
            plt.savefig(os.path.join(args.plot_dir, "pt_response_%s.png" % rname))
            plt.close()

    print("Plots written to", args.plot_dir)


if __name__ == "__main__":
    main(parse_args())
