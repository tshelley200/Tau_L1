"""
Convert a trained model to an hls4ml (Vitis HLS) project = the C++ "firmware" version of the network,
and check that it gives the same outputs as the Keras model.

Only quantized_merged_model trained with the 8 base features (no extras, no pt_norm / input_norm)
has the same layout as the network in the current firmware / CMSSW (80 inputs -> ID score + pT correction).

Usage:
  python export_hls4ml.py --model models/firmware.h5 --out_dir hls/firmware
  python export_hls4ml.py --model models/firmware.h5 --out_dir hls/firmware --data_dir <ntuple dir>   # + compare outputs
  python export_hls4ml.py ... --build   # also run HLS synthesis (needs Vitis HLS installed and on your PATH)
"""
import argparse
from utils import *
import hls4ml


def parse_args():
    p = argparse.ArgumentParser(description="Export a trained model with hls4ml")
    p.add_argument("--model", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--project_name", default="tau_nn")
    p.add_argument("--part", default="xcvu13p-flga2577-2-e")   # Serenity VU13P (the board in the correlator firmware)
    p.add_argument("--clock_period", type=float, default=2.8)  # ns (360 MHz)
    p.add_argument("--data_dir", default=None)                # if given: compare Keras vs HLS on the test split
    p.add_argument("--build", action="store_true")            # run C synthesis (slow, needs Vitis HLS)
    return p.parse_args()


def main(args):
    cfg = load_config(args.model)

    # --- warn if this model can't simply replace the one in the firmware ---
    firmware_like = (cfg["model_fn"] == "quantized_merged_model" and not cfg["track_extra"] and not cfg["calo_extra"]
                     and not cfg["hgcal_extra"] and not cfg["pt_norm"] and not cfg.get("input_norm", False))
    if not firmware_like:
        print("WARNING: this model does not have the firmware layout (quantized_merged_model, 8 base features, "
              "no normalization). The export works, but the firmware input/packing code would need changes.")
    if cfg["model_fn"] in ["class_model_quant", "deepset_cnn"] and cfg.get("mask_empty", True):
        raise ValueError("This deep set uses the custom MaskedAverage layer, which hls4ml can't convert. "
                         "Retrain with --no-mask_empty to export it.")
    if not cfg["pt_regress"]:
        print("NOTE: model has no pT head; the firmware/CMSSW network expects 2 outputs (ID + pT correction)")

    model = load_qmodel(args.model, compile=False)

    # --- hls4ml config: one setting per layer, fully parallel (Latency strategy, ReuseFactor 1) like the current firmware ---
    hls_cfg = hls4ml.utils.config_from_keras_model(model, granularity="name", backend="Vitis")
    for layer in hls_cfg["LayerName"]:
        hls_cfg["LayerName"][layer]["Strategy"] = "Latency"
        hls_cfg["LayerName"][layer]["ReuseFactor"] = 1
    if "model_input" in hls_cfg["LayerName"]:
        hls_cfg["LayerName"]["model_input"]["Precision"] = {"result": "fixed<16,10>"}  # same input type as the firmware (input_t)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=hls_cfg, backend="Vitis", output_dir=args.out_dir, project_name=args.project_name,
        part=args.part, clock_period=args.clock_period, io_type="io_parallel")
    hls_model.compile()  # builds the C++ emulation library, so hls_model.predict() works
    print("Wrote hls4ml project to %s (firmware C++ in %s/firmware/)" % (args.out_dir, args.out_dir))

    # --- compare Keras vs HLS outputs on the test split ---
    if args.data_dir:
        d = create_train_test_data(args.data_dir, split="test", sig_file=cfg["sig_file"], bkg_file=cfg["bkg_file"],
                                   train_frac=cfg["train_frac"], val_frac=cfg["val_frac"], test_frac=cfg["test_frac"],
                                   seed=cfg["seed"], eta_min=cfg["eta_min"], eta_max=cfg["eta_max"],
                                   pt_cut=cfg["pt_cut"], pt_max=cfg["pt_max"], pt_norm=cfg["pt_norm"])
        X = build_inputs(d, cfg)
        X = [np.ascontiguousarray(x, dtype=np.float32) for x in X] if isinstance(X, list) else np.ascontiguousarray(X, dtype=np.float32)
        k_id, k_pt = split_outputs(model.predict(X, verbose=0))
        h = hls_model.predict(X)
        h_id, h_pt = (h[0].ravel(), h[1].ravel()) if isinstance(h, list) else (np.asarray(h).reshape(len(k_id), -1)[:, 0], None)
        print("ID score: max |Keras - HLS| = %.4f (mean %.4f)" % (np.abs(k_id - h_id).max(), np.abs(k_id - h_id).mean()))
        if k_pt is not None and h_pt is not None:
            print("pT corr.: max |Keras - HLS| = %.4f (mean %.4f)" % (np.abs(k_pt - h_pt).max(), np.abs(k_pt - h_pt).mean()))

    if args.build:
        hls_model.build(csim=False, synth=True)  # resource / latency report in <out_dir>/<project>_prj/


if __name__ == "__main__":
    main(parse_args())
