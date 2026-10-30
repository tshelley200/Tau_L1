#!/bin/bash
# Make comparison plots for the models in a plot list.
#
#   jobs/run_plot.sh                                  # uses jobs/plot_list.txt
#   jobs/run_plot.sh jobs/my_plots.txt                # another list
#   jobs/run_plot.sh jobs/plot_list.txt --pt_cut 30   # extra options are passed on to plot.py
#
# Plots go to $PLOT_DIR/<list name>/ (folders from jobs/settings.sh). Runs fine on a CPU.
# As a batch job:  sbatch -p test -c 2 --mem 16G -t 0:30:00 --wrap "jobs/run_plot.sh jobs/plot_list.txt"

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO_DIR/jobs/settings.sh"

PLOT_LIST=$(realpath "${1:-$REPO_DIR/jobs/plot_list.txt}")
shift || true
[ -f "$PLOT_LIST" ] || { echo "Plot list not found: $PLOT_LIST"; exit 1; }

MODELS=(); LABELS=()
while IFS=';' read -r name label; do
    name=$(echo "$name" | xargs); label=$(echo "$label" | xargs)
    [ -z "$name" ] && continue
    MODELS+=("$MODEL_DIR/$name.h5"); LABELS+=("${label:-$name}")
done < <(sed 's/#.*//' "$PLOT_LIST" | awk 'NF')

OUT="$PLOT_DIR/$(basename "$PLOT_LIST" .txt)/"
echo "Comparing ${#MODELS[@]} model(s) -> $OUT"

# activate the environment if it isn't already
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then module load python; source activate "$ENV_NAME"; fi
export TF_CPP_MIN_LOG_LEVEL=2

cd "$REPO_DIR/merged_training"
python plot.py --data_dir "$DATA_DIR" --plot_dir "$OUT" --models "${MODELS[@]}" --labels "${LABELS[@]}" "$@"
