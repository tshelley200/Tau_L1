from utils import *
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train the L1 NN tau ID (+ optional pT correction)")

    p.add_argument("--seed", type=int, default=1111)

    # ---------------------- input files / output model ----------------------
    p.add_argument("--data_dir",  default="/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/L1NNTau_data/")
    p.add_argument("--sig_file",  default="VBFHToTauTau_PU200_1-1_emseed_sig_test.root")
    p.add_argument("--bkg_file",  default="MinBias_PU200_1-1_emseed_bkg_test.root")
    p.add_argument("--qcd_file",  default="QCD_Pt15To3000_PU200_1-1_emseed_qcd_test.root")
    # output model; its folder is created if needed. A <name>_config.json is saved next to it (used by plot.py / predict.py)
    p.add_argument("--model_out", default="/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/models/oct_22/sample_class_quant_track_extra_endcap.h5")

    #===========================model and its inputs============================================
    p.add_argument("--model_fn", default="class_model_quant",
                   choices=["class_model_quant",       # quantized deep set (default)
                            "class_model",             # plain float dense, flattened inputs + scaler
                            "cnn",                     # quantized 2D CNN on a calo image only (barrel)
                            "deepset_cnn",             # deep set on particles + CNN on a calo image
                            "merged_model",            # float dense, 25-25-15-15-10 (float version of the firmware model)
                            "quantized_merged_model"]) # = the model in the firmware / CMSSW (with the 8 base features)
    # append track features per particle (16 old / 17 new layout). Default: on, except for merged models (firmware uses 8 base features only)
    p.add_argument("--track_extra", action=argparse.BooleanOptionalAction, default=None)
    # append GCT ECal (45) + HCal (9) cells per particle (barrel). Ignored for cnn / deepset_cnn (they use --image)
    p.add_argument("--calo_extra",  action=argparse.BooleanOptionalAction, default=False)
    # append HGCal shower features per particle (endcap)
    p.add_argument("--hgcal_extra", action=argparse.BooleanOptionalAction, default=False)
    # calo image for cnn / deepset_cnn: ECal (10, 3, 15) or HCal (10, 3, 3)
    p.add_argument("--image", default="hcal", choices=["hcal", "ecal"])
    # deep sets (class_model_quant, deepset_cnn): skip empty (zero-padded) particle slots in the average (M3).
    # --no-mask_empty = old plain average over all 10 slots
    p.add_argument("--mask_empty", action=argparse.BooleanOptionalAction, default=True)
    # with --mask_empty: also give the network the number of real particles / 10 (the masked average alone loses it)
    p.add_argument("--add_count", action=argparse.BooleanOptionalAction, default=True)
    # merged models only (M2, on hold): add a normalization layer on the raw inputs. Changes the firmware -> default off
    p.add_argument("--input_norm", action=argparse.BooleanOptionalAction, default=False)

    #===========================pT regression============================================
    # add a pT head that predicts the correction factor gen pT / L1 pT (trained on real taus only)
    p.add_argument("--pt_regress", action=argparse.BooleanOptionalAction, default=False)
    # total loss = gamma * ID loss + (1 - gamma) * pT loss
    p.add_argument("--gamma", type=float, default=0.7)

    #===========================cuts, split and training weights============================================
    # random split BY EVENT; QCD is used for training only. Defaults = old proportions (70% -> 80/20 train/val, 30% test)
    p.add_argument("--train_frac", type=float, default=0.56)
    p.add_argument("--val_frac",   type=float, default=0.14)
    p.add_argument("--test_frac",  type=float, default=0.30)  # held out for plot.py (same seed + fractions there)
    p.add_argument("--pt_cut",     type=float, default=15.)   # raw L1 tau pT > pt_cut
    p.add_argument("--pt_max",     type=float, default=9999.)
    p.add_argument("--eta_min",    type=float, default=1.5)   # |eta| >= eta_min
    p.add_argument("--eta_max",    type=float, default=TAU_ETA_MAX)  # |eta| < eta_max (taus only exist up to 2.5)
    # divide all pT-like inputs by the tau candidate's raw L1 pT ('pt' branch)
    p.add_argument("--pt_norm",    action=argparse.BooleanOptionalAction, default=False)
    # balance: sig = bkg in every pT bin; flat: balance + every pT bin counts equally (more weight at high pT); none: overall sig/bkg balance only
    p.add_argument("--pt_weighting", default="balance", choices=["balance", "flat", "none"])

    #===========================training============================================
    # continue training model_out; the result is saved under a NEW name (<model_out>_cont.h5)
    p.add_argument("--load", action=argparse.BooleanOptionalAction, default=False)

    # ---------------------- training hyperparameters ---------------
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--epochs",     type=int, default=35)
    return p.parse_args()


def new_name(path):
    """model.h5 -> model_cont.h5 (or model_cont2.h5, ... if that exists), so a reloaded model never overwrites the original."""
    base = path[:-3] if path.endswith(".h5") else path
    out, i = base + "_cont.h5", 2
    while os.path.exists(out):
        out, i = base + "_cont%i.h5" % i, i + 1
    return out


def main(args):
    # ------- settings (stored in <model>_config.json so plot.py / predict.py rebuild the same inputs) -------
    if args.track_extra is None:  # default: off for merged models (firmware match), on otherwise
        args.track_extra = args.model_fn not in MERGED_MODELS
    cfg = {k: getattr(args, k) for k in
           ["model_fn", "track_extra", "calo_extra", "hgcal_extra", "image", "mask_empty", "add_count", "input_norm", "pt_regress", "gamma",
            "pt_norm", "pt_weighting", "pt_cut", "pt_max", "eta_min", "eta_max",
            "train_frac", "val_frac", "test_frac", "seed", "sig_file", "bkg_file", "qcd_file"]}
    check_config(cfg)  # raises for impossible setups, warns for useless ones

    model_in = args.model_out
    model_out = new_name(args.model_out) if args.load else args.model_out
    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)  # T6
    cfg_path, scaler_path = model_paths(model_out)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
        except RuntimeError as e:
            print(e)

    # --- seeds ---
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)


    # --- load data: train part of sig/bkg + all qcd; val part of sig/bkg (split by event, cuts applied) ---
    load_kw = dict(sig_file=args.sig_file, bkg_file=args.bkg_file, qcd_file=args.qcd_file,
                   train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
                   eta_min=args.eta_min, eta_max=args.eta_max, pt_cut=args.pt_cut, pt_max=args.pt_max, pt_norm=args.pt_norm)
    train = create_train_test_data(args.data_dir, split="train", **load_kw)
    val   = create_train_test_data(args.data_dir, split="val",   **load_kw)

    # --- build model inputs (glue the requested feature blocks onto each particle / make the calo image) ---
    X_train, X_val = build_inputs(train, cfg), build_inputs(val, cfg)
    if args.model_fn == "class_model":  # float dense net: standardize inputs, save the scaler for plot.py / predict.py
        scaler = StandardScaler().fit(X_train)
        X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
        joblib.dump(scaler, scaler_path)
    shapes = [x.shape for x in X_train] if isinstance(X_train, list) else [X_train.shape]
    print("Model inputs:", shapes)
    cfg["input_shapes"] = [list(s[1:]) for s in shapes]

    # --- per-candidate training weights (pT-binned, see train_weights) ---
    class_labels = {'bkg' : 0, 'sig' : 1}
    y_train, y_val = train['Y_jetID'], val['Y_jetID']
    w_train = train_weights(y_train, train['reco_pt'], class_labels, mode=args.pt_weighting)
    w_val   = train_weights(y_val,   val['reco_pt'],   class_labels, mode=args.pt_weighting)


    # --- build (or reload) the model ---
    if(args.load):
        model = load_qmodel(model_in, compile=False)
        if (len(model.outputs) == 2) != args.pt_regress:
            raise ValueError("%s has %i output(s), but --pt_regress is %s" % (model_in, len(model.outputs), args.pt_regress))
        print("Loaded %s; will save to %s" % (model_in, model_out))
    else:
        if args.model_fn == "deepset_cnn":
            model = deepset_cnn(point_shape=X_train[0].shape[1:], cnn_shape=X_train[1].shape[1:],
                                pt_regress=args.pt_regress, mask_empty=args.mask_empty, add_count=args.add_count)
        elif args.model_fn == "class_model_quant":
            model = class_model_quant(input_shape=X_train.shape[1:], pt_regress=args.pt_regress,
                                      mask_empty=args.mask_empty, add_count=args.add_count)
        elif args.model_fn in MERGED_MODELS:
            model = globals()[args.model_fn](input_size=X_train.shape[1], pt_regress=args.pt_regress, input_norm=args.input_norm)
        else:
            model = globals()[args.model_fn](input_shape=X_train.shape[1:], pt_regress=args.pt_regress)

    # --- compile: ID = binary cross-entropy; pT = mean squared error (real taus only); metrics use the weights ---
    opt = optimizers.Adam()
    if args.pt_regress:
        id_name, pt_name = model.output_names
        model.compile(optimizer = opt,
                loss = {id_name: 'binary_crossentropy', pt_name: 'mean_squared_error'},
                loss_weights = {id_name: args.gamma, pt_name: 1. - args.gamma},
                weighted_metrics = {id_name: ['accuracy', 'AUC']})
        # pT target = gen pT / L1 pT; background rows get weight 0 in the pT loss (they have no real tau pT)
        y_fit = {id_name: y_train, pt_name: train['Y_pt_corr']}
        w_fit = {id_name: w_train, pt_name: w_train * (y_train == 1)}
        val_data = (X_val, {id_name: y_val, pt_name: val['Y_pt_corr']},
                           {id_name: w_val, pt_name: w_val * (y_val == 1)})
    else:
        model.compile(optimizer = opt, loss='binary_crossentropy', weighted_metrics = ['accuracy', 'AUC'])
        y_fit, w_fit, val_data = y_train, w_train, (X_val, y_val, w_val)
    model.summary()

    # --- stop after 10 epochs w/o val_loss improvement; halve LR after 5 ---
    callbacks = [ tf.keras.callbacks.EarlyStopping(verbose = 1, patience=10, restore_best_weights = True, monitor = 'val_loss', min_delta=1e-4),
                  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
                ]

    # --- train (validation = separate val split of sig + MinBias, so val_loss / val_auc are meaningful) ---
    history = model.fit(X_train, y_fit,
            sample_weight = w_fit,
            epochs=args.epochs,
            batch_size = args.batch_size,
            verbose =1,
            validation_data = val_data,
            callbacks = callbacks,
            shuffle = True)  # reshuffles the training set every epoch

    # --- sanity check: a constant ID score means the ID head stopped learning (seen with quantized_merged_model
    #     + pT regression at gamma <= 0.7 on small samples; raw inputs saturate its 9-bit ReLUs) ---
    id_val, _ = split_outputs(model.predict(X_val, batch_size = 1024, verbose = 0))
    if np.std(id_val) < 1e-3:
        print("WARNING: the ID score is constant on the validation set (ID head collapsed). Try a higher --gamma or --pt_norm.")

    # --- save trained model (.h5) + its config ---
    model.save(model_out)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved %s and %s" % (model_out, cfg_path))
    return model, history


if __name__ == "__main__":
    main(parse_args())
