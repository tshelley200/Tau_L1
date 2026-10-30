"""
Run a trained model over EVERY row of one or more ntuples and save, per input file:
  <out>.root : friend tree 'nn' with the same rows, in the same order, as ntuplePupSingle/tree
               branches: nn_id (ID score), nn_pt_corr (pT correction factor), nn_pt (corrected pT = pt * nn_pt_corr)
  <out>.npz  : the same numbers + pt, ptcorr, genpt1, gendr1, eta, run/lumi/event (for plotting in python)
Rows that are invalid (pt <= 0) or outside the model's |eta| range get -1.

Usage:
  python predict.py --model model.h5 --inputs sig.root bkg.root --out_dir preds/
ROOT:   tree->AddFriend("nn", "preds/sig_nn.root");  tree->Draw("nn.nn_pt / genpt1");
"""
import argparse
from utils import *


def parse_args():
    p = argparse.ArgumentParser(description="Write NN outputs (ID score, corrected pT) as a friend tree + npz")
    p.add_argument("--model", required=True)
    p.add_argument("--inputs", nargs="+", required=True)   # ntuple files
    p.add_argument("--out_dir", default=None)              # default: next to each input file
    p.add_argument("--batch_size", type=int, default=1024)
    return p.parse_args()


def main(args):
    cfg = load_config(args.model)
    model = load_qmodel(args.model, compile=False)
    scaler = joblib.load(model_paths(args.model)[1]) if cfg["model_fn"] == "class_model" else None

    for path in args.inputs:
        d = load_sample(path, pt_norm=cfg["pt_norm"])
        X = build_inputs(d, cfg)
        if scaler is not None:
            X = scaler.transform(X)
        id_score, pt_factor = split_outputs(model.predict(X, batch_size=args.batch_size))

        # valid rows: real candidate (pt > 0) inside the model's |eta| range; everything else -> -1
        aeta = np.abs(d["eta"])
        valid = (d["pt"] > 0) & (aeta >= cfg["eta_min"]) & (aeta < cfg["eta_max"])
        out = {
            "nn_id":      np.where(valid, id_score, -1.).astype(np.float32),
            "nn_pt_corr": np.where(valid, pt_factor, -1.).astype(np.float32) if pt_factor is not None else np.full(len(valid), -1., np.float32),
            "nn_pt":      np.where(valid, d["pt"] * pt_factor, -1.).astype(np.float32) if pt_factor is not None else np.full(len(valid), -1., np.float32),
        }
        if pt_factor is None:
            print("NOTE: model has no pT head (trained without --pt_regress); nn_pt_corr / nn_pt are -1")

        out_dir = args.out_dir or os.path.dirname(os.path.abspath(path))
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, os.path.basename(path).replace(".root", "") + "_nn")

        with uproot.recreate(base + ".root") as f:
            f["nn"] = out  # friend tree: same row order as ntuplePupSingle/tree
        run, lumi, event = d["event_id"].T
        np.savez(base + ".npz", **out, valid=valid, pt=d["pt"], ptcorr=d["ptcorr"], genpt1=d["gen_pt"],
                 gendr1=d["gendr"], eta=d["eta"], run=run, lumi=lumi, event=event)
        print("Saved %s.root (tree 'nn', %i rows, %i valid) and %s.npz" % (base, len(valid), valid.sum(), base))


if __name__ == "__main__":
    main(parse_args())
