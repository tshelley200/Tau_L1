# Tau_L1
Codes and documentation for L1Tau.

Authors: Orion Foo, Duc Hoang, Phil Harris. (MIT)

Training pipeline in `merged_training/` and related verification updates added by Oz Amram.

## Repository structure

* `merged_training/` - main training pipeline: data loading, model definitions, training script, and plotting/feature-inspection scripts. See below.
* `verification/` - scripts to sanity-check ntuple inputs and compare against CMSSW-side quantities, plus instructions for setting up a CMSSW area (`verification/CMSSW/`) and a list of input samples (`verification/samples/`).
* `data/` - expects merged `sig.root` / `bkg.root` / `qcd.root` ntuples (produced upstream, not checked into the repo) consumed by `merged_training/`.
* `models/` - trained models (`.h5`) and their input scalers (`.pkl`), organized by input feature set (baseline, track/calo "extra" features, barrel/endcap, etc.).
* `hls4ml/`, `HLS_IP/` - firmware synthesis of trained models for the L1 trigger.
* `plots/`, `plots_barrel/` - output plots from the training/verification scripts.

## Conda environment

Create conda environment:

```
conda-env create -f environment.yml
```

Activate the environment:

```
conda activate tau_training
```

And then do whatever you want in this environment (edit files, open notebooks, etc.). To deactivate the environment:

```
conda deactivate
```

If you make any update for the environment, please edit the `environment.yml` file and run:

```
conda env update --file environment.yml  --prune
```

Reference on conda environment here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

## Running jupyter notebook on submit machine at MIT

You could run the notebook on the submit machine by ssh tunneling, first you need to ssh into the machine using: 

```
ssh -L 127.0.0.1:8722:127.0.0.1:8722 <username>@submit03.mit.edu
```

Then, `cd` to the directory of your choice:

```
conda activate tau_training
```

You could then open the notebook:

```
jupyter notebook --no-browser --port 8722 --ip 127.0.0.1
```

It would produce a link and then you can copy the link and paste it to your local browser to open the notebook!

## Training pipeline (`merged_training/`)

This is the main pipeline used to train the tau ID classifier.

* `utils.py` - shared code: `create_train_test_data()` loads the merged `sig.root`/`bkg.root`/`qcd.root` ntuples from `data/merged/`, applies pT/eta/deltaR selections, and returns per-candidate input arrays plus the "extra" track and calo (ECal/HCal or HGCal) shower-shape features; the model architectures (dense, quantized deepset, CNN, and combined deepset+CNN) and `train_weights()` (flattens the training weights in truth pT so all pT ranges contribute equally) also live here.
* `training.py` - main training script. Configuration (input data path, model output path, which extra features to include, pT cuts, eta range, which `model_fn` to use, etc.) is set via variables at the top of the file rather than command-line args, so edit them there before running: `python training.py`.
* `plot.py` - loads a set of already-trained models, evaluates them on held-out test data, and produces ROC curve and background-rate-vs-signal-efficiency comparison plots. Edit the `models`/`labels`/`model_str` lists at the top to choose which trained models to compare.
* `plot_features.py` - plots signal-vs-background distributions (and correlation with pT) for every input feature, useful for sanity-checking new input variables before adding them to training.

Trained models are saved under `models/`, and plots are written under `plots/`/`plots_barrel/`.

## Verification updates

`verification/check_inputs.py` compares input feature distributions (base candidate features and the extra track/calo features) between signal and background ntuples, and now reads directly from `data/merged/{sig,bkg}.root`. Note it uses `uproot` (not `uproot4` as in `environment.yml`) - update your environment accordingly if you hit an import error.

## Related Resources (latest addition on top)

* CDS publication: 
* Public Twiki (v2): https://twiki.cern.ch/twiki/bin/view/CMSPublic/L1Phase2NNPuppiTau
* Orion's talk in Tau-Jets-MET: https://indico.cern.ch/event/1321399/#17-phase-2-nn-puppi-taus
* DP Note Tau ID v2 talk: https://indico.cern.ch/event/1298326/#14-dp-note-nn-puppi-taus
  * Most recent version: https://docs.google.com/presentation/d/12offcwPbynGl94C5zrsw39owu82U9SG1xQRz2dTqPkw/edit#slide=id.g278ac055907_0_61   
* Duc's retraining Tau talk at L1 General Meeting: https://indico.cern.ch/event/1271855/#202-performance-report-algorit
* Official Document on Rates: https://indico.cern.ch/event/1257943/contributions/5343040/attachments/2633523/4567965/AlgorithmsAndPerformance_2023_4.pdf
* TauID @ L1 Talk (Phil): https://www.dropbox.com/s/84jbya301jtac8l/PCH_Tau_25_03.pdf?dl=0
