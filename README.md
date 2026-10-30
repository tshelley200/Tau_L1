# L1 NN Tau

This repository trains the small neural network (NN) that the CMS Phase-2 **Level-1 (L1) trigger** uses to
identify hadronically decaying tau leptons (τh).

**Flowchart**
```
ntuples (.root)  ──►  training.py  ──►  model.h5 + model_config.json  ──►  plot.py  ──►  ROC / rate / pT plots
sig/MinBias(Bkg)/QCD                                                     └─►  export_hls4ml.py  ──►  firmware C++
```
---

## Contents
1. [Repository layout](#1-repository-layout)
2. [Installation (Harvard FASRC)](#2-installation-harvard-fasrc)
3. [Input data](#3-input-data)
4. [Quick start: example training](#4-quick-start-your-first-training)
5. [Models](#5-the-models)
6. [All training options](#6-all-training-options)
7. [Running many trainings on SLURM](#7-running-many-trainings-on-slurm)
8. [Making plots](#8-making-plots)
9. [Exporting models to firmware (hls4ml)](#9-exporting-a-model-to-firmware-hls4ml)

---

## 1. Repository layout

| Path | What it is |
|---|---|
| `environment.yml` | Software environment (Python packages) |
| `merged_training/utils.py` | Shared code: data loading, input building, model definitions, training weights |
| `merged_training/training.py` | Trains one model |
| `merged_training/plot.py` | Compares trained models on the test set (ROC, trigger rate, pT response) |
| `merged_training/predict.py` | Runs a model over a full ntuple and writes its outputs as a ROOT friend tree + `.npz` |
| `merged_training/export_hls4ml.py` | Converts a model to an hls4ml (FPGA) project |
| `jobs/` | SLURM scripts to run many trainings at once and to make comparison plots |
| `HLS_IP/`, `hls4ml/` | Older firmware conversion files (kept for reference) |
| `verification/` | Scripts to sanity-check ntuples and set up CMSSW |

---

## 2. Installation (Harvard FASRC)

**Create the environment**
```bash
salloc -p test -c 4 --mem 16G -t 1:00:00
module load python
mamba env create -f environment.yml           
```

**Activate Environment** 
```bash
module load python
source activate tau_nn
```
> TensorFlow must stay at 2.15. Newer versions use Keras 3, which QKeras and this code don't support.

---

## 3. Input data

The ntuples are on lab storage:
```
/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/L1NNTau_data/
```

| File (default name) | Sample | Label | Used for |
|---|---|---|---|
| `VBFHToTauTau_PU200_1-1_emseed_sig_test.root` | VBF H → ττ | 1 (tau) | train / val / test |
| `MinBias_PU200_1-1_emseed_bkg_test.root` | MinBias (Bkg) | 0 (fake) | train / val / test, **trigger rate** |
| `QCD_Pt15To3000_PU200_1-1_emseed_qcd_test.root` | QCD multijet | 0 (fake) | training only |

Other file names can be given with `--sig_file`, `--bkg_file`, `--qcd_file`.

The tree `ntuplePupSingle/tree` has one row per L1 tau candidate. Each candidate has up to
10 particles, ordered by pT; empty particle slots are filled with zeros.

| Branch | Per particle | Contents |
|---|---|---|
| `m_inputs` | 8 | pT, Δη, Δφ (relative to the seed particle), 5 particle-type flags (photon, electron, muon, neutral hadron, charged hadron) |
| `extra_inputs` | 16–17 | Track info (vertex, impact parameters, track quality) + linked calorimeter cluster (η, φ, H/E, pT) |
| `ecal_shape`, `hcal_shape` (old) or `gct_ecal`, `gct_hcal`, `hgcal` (new) | 45 / 9 / 21 | Calorimeter information: barrel ECal 3×15 and HCal 3×3 cells, endcap HGCal shower shape |
| `pt` | per candidate | L1 tau pT **without** NN correction (used for cuts and as "L1 pT") |
| `ptcorr` | per candidate | L1 tau pT corrected by the NN currently in CMSSW (for comparison) |
| `eta`, `genpt1`, `gendr1` | per candidate | η; generator (true) tau pT; distance ΔR to the true tau (−999 if none) |
| `run`, `lumi`, `event` | per candidate | Event ID (used for the per-event split and the trigger rate) |

The code reads both the old and the new calorimeter branch layout automatically. The second tree, `ntuple/tree`, contains every
processed event (including ones without any tau candidate); `plot.py` uses it to normalize the trigger rate.

**How the data is used**
* **Labels:** taus come from the H→ττ file, fakes from MinBias / QCD. A signal candidate is only kept if it lies within ΔR < 0.4 of a true tau.
* **Cuts:** L1 pT > `--pt_cut` (default 15 GeV) and `--eta_min` ≤ |η| < `--eta_max`. Taus only exist up to |η| = 2.5.
* **Split:** events are randomly divided into train / validation / test (default 56 / 14 / 30 %). All candidates of one event land in the same set.

---

## 4. Quick start: training example

```bash
salloc -p gpu_test --gres=gpu:1 -c 4 --mem 32G -t 0:30:00
module load python && source activate tau_nn
cd ~/L1NNTau/merged_training

DATA=/n/holystore01/LABS/iaifi_lab/Lab/phlab-l1nntau/L1NNTau_data

# train a deep set on all |eta| for 3 epochs (just to see that everything works)
python training.py --data_dir $DATA --model_out ~/tau_test/my_first_model.h5 \
                   --eta_min 0 --calo_extra --hgcal_extra --epochs 3

# plot it
python plot.py --data_dir $DATA --models ~/tau_test/my_first_model.h5 --labels "My first model" \
               --plot_dir ~/tau_test/plots/
```
Look at `~/tau_test/plots/roc_all.png`. With 3 epochs the curve won't be great; real trainings use 35 epochs (the default).

---

## 5. Models

Choose one with `--model_fn`. Each model takes the (up to) 10 particles of a tau candidate and outputs an ID score
(plus a pT correction with `--pt_regress`).

| `--model_fn` | Big idea | Input shape | Quantized Option | In firmware? |
|---|---|---|---|---|
| `quantized_merged_model` | **Plain dense network** on all particles flattened into one row, matches current firmware | (80,) = 10 × 8 | 9-bit | **Yes**|
| `merged_model` | Same as above but with normal (float) numbers | (10·F,) | No | No |
| `class_model` | Dense network with dropout; inputs standardized first | (10·F,) | No | No |
| `class_model_quant` | **Deep set**: the same small network looks at each particle, then the results are averaged | (10, F) | 9-bit | No |
| `cnn` | **Image network** on calorimeter cells | (10, 3, 3) HCal or (10, 3, 15) ECal | 9-bit | No |
| `deepset_cnn` | Deep set on the particles **+** CNN on a calorimeter image, combined | [(10, F), image] | 9-bit | No |

F = features per particle (see the table in section 6).

* **Dense (merged) network.** All 10 particles' features are put into one long list (particle 1, then particle 2, …).
  Every number gets its own weight, so the network knows which slot is which: slot 1 is always the hardest particle, and it can
  see which slots are empty. This is the network that runs in the trigger today.
* **Deep set.** A tau candidate is a *set* of particles; their order shouldn't matter. The deep set runs one small network on
  each particle (shared weights), then averages the 10 results into one summary of the candidate, which a small classifier turns
  into a score. Empty slots are skipped in the average, and the number of real particles is added to the summary
  (`--mask_empty`, `--add_count`, both on by default).
* **CNN.** Treats the calorimeter energy deposits around the particles as a tiny picture and scans it with 3×3 filters that learn
  shower shapes (e.g. "narrow core, quiet around it" = tau-like). The calorimeter picture only exists in the **barrel**.
* **Deep set + CNN.** Uses both: particle information through the deep set and the calorimeter picture through the CNN.

**pT head (`--pt_regress`).** Adds a second output that learns the correction factor `true pT / L1 pT` from real taus.
The total loss is `gamma × ID loss + (1 − gamma) × pT loss` (`--gamma`, default 0.7).

---

## 6. All training options

Run `python training.py -h` for the full list. All options:

### Files

| Option | Default | Meaning |
|---|---|---|
| `--data_dir` | lab data folder | Folder with the ntuples |
| `--sig_file`, `--bkg_file`, `--qcd_file` | see section 3 | File names inside `--data_dir` |
| `--model_out` | a path on lab storage | Where to save the trained model (`.h5`); the folder is created if needed |
| `--seed` | 1111 | Random seed (split and training). Change it to check how stable a result is |

### Model and inputs

| Option | Default | Meaning |
|---|---|---|
| `--model_fn` | `class_model_quant` | Which model (section 5) |
| `--track_extra` / `--no-track_extra` | on (off for merged models) | Add the track features to each particle |
| `--calo_extra` | off | Add barrel calorimeter cells (ECal 45 + HCal 9) to each particle |
| `--hgcal_extra` | off | Add endcap HGCal shower features (21) to each particle |
| `--image` | `hcal` | Calorimeter picture for `cnn` / `deepset_cnn`: `hcal` (3×3) or `ecal` (3×15) |
| `--mask_empty` / `--no-mask_empty` | on | Deep sets: skip empty particle slots in the average |
| `--add_count` / `--no-add_count` | on | Deep sets: add the number of real particles to the summary |
| `--input_norm` | off | Merged models: normalize the raw inputs first (adds a layer, so it **no longer matches the firmware**) |

Features per particle (F) that these flags build:

| Block | Size | Flag |
|---|---|---|
| Base (pT, Δη, Δφ, particle type) | 8 | always |
| Track | 16 (old ntuples) / 17 (new) | `--track_extra` |
| Barrel ECal + HCal cells | 45 + 9 | `--calo_extra` |
| Endcap HGCal shower | 21 | `--hgcal_extra` |

For `cnn` / `deepset_cnn` the barrel calorimeter enters through `--image` instead, so `--calo_extra` is ignored.

### pT regression

| Option | Default | Meaning |
|---|---|---|
| `--pt_regress` | off | Add the pT-correction output |
| `--gamma` | 0.7 | Weight of the ID loss; the pT loss gets 1 − gamma |

### Cuts, split and weights

| Option | Default | Meaning |
|---|---|---|
| `--pt_cut`, `--pt_max` | 15, 9999 | L1 pT window (GeV) |
| `--eta_min`, `--eta_max` | 1.5, 2.5 | \|η\| window. Barrel: 0–1.5, endcap: 1.5–2.5, everything: 0–2.5 |
| `--train_frac`, `--val_frac`, `--test_frac` | 0.56, 0.14, 0.30 | Fractions of events for training / validation / test |
| `--pt_norm` | off | Divide pT-like inputs by the candidate's L1 pT (makes them fractions) |
| `--pt_weighting` | `balance` | `balance`: taus and fakes get equal total weight in every pT bin (so pT alone can't separate them). `flat`: same, and every pT bin counts equally (more weight at high pT). `none`: only overall tau/fake balance |

### Training

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 35 | Maximum number of passes over the data (training stops early if validation stops improving for 10 epochs) |
| `--batch_size` | 100 | Candidates per training step |
| `--load` | off | Continue training `--model_out`; the result is saved as `<name>_cont.h5` |

### Calorimeter flags

| \|η\| range | Allowed |  Warning | Error/Not Allowed |
|---|---|---|---|
| Barrel only (`--eta_max` ≤ 1.5) | `--calo_extra` | `+ --hgcal_extra` (adds all-zero columns) | `--hgcal_extra` alone |
| Endcap only (`--eta_min` ≥ 1.5) | `--hgcal_extra` | `+ --calo_extra` (adds all-zero columns) | `--calo_extra` alone |
| Crosses 1.5 (e.g. 0–2.5) | either, or both | only one of them (taus on the other side get no calo info) | – |
| `cnn` | barrel only | – | endcap or crossing 1.5 |
| `deepset_cnn` | barrel, or crossing 1.5 | crossing 1.5 without `--hgcal_extra` | endcap only |

### Training Output

| File | Contents |
|---|---|
| `<name>.h5` | The trained model |
| `<name>_config.json` | All settings of this training. **Keep it next to the model**: `plot.py`, `predict.py` and `export_hls4ml.py` need it |
| `<name>_scaler.pkl` | Input scaler (only `class_model`) for applying the same rescaling|

The log shows how many candidates passed the cuts (`Nsig, Nbkg`), the model layout, and loss / accuracy / AUC per epoch for training and validation (`val_...`).

---

## 7. Running trainings on SLURM

**1. Edit `jobs/settings.sh` once.** Set the data, model and plot folders, the environment name, and the SLURM resources (partition, time, memory).

**2. Write trainings in a list (one line per job/submission)**, e.g. `jobs/train_jobs.txt`:
```
# <name>              <training.py options>
firmware_all_ptreg    --model_fn quantized_merged_model --eta_min 0 --pt_regress
ds_endcap_hgcal       --eta_min 1.5 --hgcal_extra
ds_all_calo_hgcal     --eta_min 0 --calo_extra --hgcal_extra --pt_regress
dscnn_barrel_ecal     --model_fn deepset_cnn --eta_min 0 --eta_max 1.5 --image ecal
```
* `<name>` must be unique; the model is saved as `$MODEL_DIR/<name>.h5`.
* `--data_dir` and `--model_out` are added for you.

**3. Submit**
```bash
cd ~/L1NNTau
DRY_RUN=1 jobs/submit_train.sh                # prints the commands, submits nothing
jobs/submit_train.sh                          # submits one GPU job per line, jobs/train_jobs.txt is the default list to be submitted
jobs/submit_train.sh jobs/my_other_list.txt   # a different list
```
---

## 8. Making plots

`plot.py` runs the models on the **test set**

**Write `jobs/plot_list.txt` with one model per line, `<name> ; <legend label>`, for example:
```
firmware_all_ptreg   ; Firmware model
ds_all_calo_hgcal    ; Deep set (calo + HGCal)
```
then
```bash
jobs/run_plot.sh                # plots are stored in  $PLOT_DIR/plot_list/
jobs/run_plot.sh jobs/plot_list.txt --pt_cut 30   # extra options go to plot.py
```

**Or running it directly:**
```bash
cd merged_training
python plot.py --data_dir $DATA --plot_dir ../plots/ \
               --models model_a.h5 model_b.h5 --labels "Model A" "Model B"
```

| Option | Default | Meaning |
|---|---|---|
| `--models`, `--labels` | – | Model files and their legend names (same number of each) |
| `--plot_dir` | `../plots/` | Output folder |
| `--pt_cut`, `--pt_max`, `--eta_min`, `--eta_max` | from the first model | Cuts for the test set |
| `--train_frac`, `--val_frac`, `--test_frac`, `--seed` | from the first model | Must match training, otherwise test events may have been used in training (a warning is printed) |
| `--sig_file`, `--bkg_file` | from the first model | Input files |
| `--n_minbias_events` | counted from the file | Total number of MinBias events, if the file has no `ntuple/tree` |

**Outputs**

| File | Shows |
|---|---|
| `roc_all.png` | ROC curve: fraction of taus kept vs fraction of fakes kept, for every score cut. AUC in the legend (0.5 = random, 1 = perfect) |
| `rate_all.png` | Trigger rate (kHz) vs tau efficiency. Lower is better. An event fires if any of its candidates passes |
| `pt_response_all.png` | pT / true pT for taus: raw L1, current CMSSW correction, and each model with a pT head. Peak at 1 and narrow = good |
| `*_barrel.png`, `*_endcap.png` | The same per region (if the \|η\| range crosses 1.5) |
| `<label>_test.npz` | Each model's test-set outputs, for your own plots: `id_score`, `pt_factor`, `corrected_pt`, `reco_pt`, `ptcorr_cmssw`, `gen_pt`, `eta`, `label`, `event_id` |

Load the `.npz` in Python with `z = np.load("Model_A_test.npz"); z["id_score"]`.

To put a model's outputs back into the ntuples (as a ROOT friend tree), use `predict.py`; see the comment at the top of that file.

---

## 9. Exporting a model to firmware (hls4ml)

[hls4ml](https://fastmachinelearning.org/hls4ml/) turns a trained Keras / QKeras model into C++ that an FPGA tool (Vitis HLS) can synthesize.

Only `quantized_merged_model` trained with 8 base features, `--pt_regress`, and no `--pt_norm` / `--input_norm`is currently implemented on the trigger firmware. That network has exactly the same layout as the firmware and CMSSW emulator: 80 inputs, ID score + pT correction, and the same 9-bit types.

| Model | Can hls4ml convert it? | Drop-in for the current firmware? |
|---|---|---|
| `quantized_merged_model`, 8 base features, `--pt_regress` | Yes | **Yes** (new weights only) |
| `quantized_merged_model` with extra features / normalization | Yes | No: the firmware input code would need changes |
| `class_model_quant`, `deepset_cnn` | Only with `--no-mask_empty` (masked average is a custom layer) | No: new firmware block |
| `cnn` | Yes | No: new firmware block and calorimeter inputs |

**Export**
```bash
pip install hls4ml                     # once, inside tau_nn (tested with hls4ml 1.3.0)
cd merged_training
python export_hls4ml.py --model $MODEL_DIR/firmware_all_ptreg.h5 --out_dir ../hls/firmware_all_ptreg \
                        --data_dir $DATA          # optional: compares Keras vs HLS outputs on the test set
```
* This writes an hls4ml project; the generated C++ (weights, layer types) is in `../hls/firmware_all_ptreg/firmware/`.
* The comparison should show ID-score differences of about 0.005 or less (the sigmoid uses a lookup table in firmware).
* `--build` also runs HLS synthesis (resource and latency report). That needs Vitis HLS installed.

* **Firmware:** the correlator tau firmware includes the network as C++ headers (`weights/*.h`, `defines.h`, parameters), which get replaced and re-synthesized.
* **CMSSW:** recent releases (e.g. 17_0_0_pre5) load the network as a separate compiled package (see `cms-hls4ml/NNPuppiTauModel`), selected with the producer's `tauModelPath` setting.
