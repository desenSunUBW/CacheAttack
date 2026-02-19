#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

export CUDA_VISIBLE_DEVICES=0
# collection_name="lexica"
collection_name="diffdb"

run_pipeline() {
    local run_id="$1"
    local is_test="$2"
    local min_words="$3"
    local max_words="$4"
    local model_name="$5"

    local run_dir="${exp_dir}/run_$(printf '%02d' "$run_id")"
    local filename

    if [[ "$model_name" == "flux" ]]; then
        filename="run_flux.py"
    elif [[ "$model_name" == "sd3" ]]; then
        filename="run_sd3.py"
    else
        echo "Error: Unsupported model name '$model_name'" >&2
        exit 1
    fi

    mkdir -p "${DATA_HOME}/diffusion-cache-security/${exp_dir}"
    mkdir -p "$run_dir"

    local start_time end_time elapsed
    start_time=$(date +%s)

    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate diff-sec
    cd diff-sec

    python -u input_constructor.py -o 0 -st 0.70 -s "$run_id" -cn "$collection_name" \
        2>"$1" -dir "$exp_dir" --min_words "$min_words" --max_words "$max_words" \
        | tee "$run_dir/result.txt"

    : >"$run_dir/result-recov.txt"
    : >"$run_dir/result-generated.txt"
    : >"$run_dir/classifier-result.txt"
    : >"$run_dir/image_result.txt"
    : >"$run_dir/result-mlp.txt"

    # Bash arrays are 0-based
    local thresholds=(0.60 0.55 0.50)
    local steps=(3000 5000 7000)

    local i threshold step
    for i in 1 2; do
        threshold="${thresholds[$i]}"
        step="${steps[$i]}"
        echo -e "\n>>> Running round with threshold: $threshold, step: ${step}, test: $is_test<<<\n"

        if [[ "$is_test" == "true" ]]; then
            conda activate diff-sec
            mkdir -p "${DATA_HOME}/diffusion-cache-security/${exp_dir}/experiments/round${run_id}/cache"
            mkdir -p "${DATA_HOME}/diffusion-cache-security/${exp_dir}/experiments/round${run_id}/images"

            python -u "$filename" -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/image_result.txt"
            echo -e "\n========================================================\n" >>"$run_dir/image_result.txt"

            conda activate diffusers-sec
            python -u output_classifier.py -m "$model_name" -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/classifier-result.txt"
            echo -e "\n========================================================\n" >>"$run_dir/classifier-result.txt"

            conda activate diff-sec
        fi

        python -u recover_emb.py -o 0 -s "$step" -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/result-recov.txt"
        echo -e "\n========================================================\n" >>"$run_dir/result-recov.txt"

        python -u input_constructor.py -o 1 -st "$threshold" -s "$run_id" -dir "$exp_dir" 2>&1 \
            -cn "$collection_name" --min_words "$min_words" --max_words "$max_words" \
            | tee -a "$run_dir/result-generated.txt"
        echo -e "\n========================================================\n" >>"$run_dir/result-generated.txt"

        conda activate ml
        python -u emb_to_text.py -o 0 -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/result-mlp.txt"
        echo -e "\n========================================================\n" >>"$run_dir/result-mlp.txt"

        conda activate diff-sec

        python -u input_constructor.py -o 2 --min_words "$min_words" --max_words "$max_words" \
            -st "$threshold" -s "$run_id" -dir "$exp_dir" 2>&1 -cn "$collection_name" \
            | tee -a "$run_dir/result.txt"
        echo -e "\n========================================================\n" >>"$run_dir/result.txt"
        echo -e "\n\n"
    done

    if [[ "$is_test" == "true" ]]; then
        conda activate diff-sec
        mkdir -p "${DATA_HOME}/diffusion-cache-security/${exp_dir}/experiments/round${run_id}/cache"
        mkdir -p "${DATA_HOME}/diffusion-cache-security/${exp_dir}/experiments/round${run_id}/images"

        python -u "$filename" -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/image_result.txt"
        echo -e "\n========================================================\n" >>"$run_dir/image_result.txt"

        conda activate diffusers-sec
        python -u output_classifier.py -m "$model_name" -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/classifier-result.txt"
        echo -e "\n========================================================\n" >>"$run_dir/classifier-result.txt"

        conda activate diff-sec
    fi

    echo -e "Rounds end\n"

    python -u recover_emb.py -o 1 -s 7000 -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/result-recov.txt"
    echo -e "\n========================================================\n" >>"$run_dir/result-recov.txt"

    conda activate ml
    python -u emb_to_text.py -o 1 -n "$run_id" -dir "$exp_dir" 2>&1 | tee -a "$run_dir/result-mlp.txt"
    echo -e "\n========================================================\n" >>"$run_dir/result-mlp.txt"

    conda activate diff-sec

    python -u check_cosine_sim.py -o 0 -n "$run_id" 2>&1 -cn "$collection_name" -dir "$exp_dir" \
        | tee "$run_dir/result-check.txt"

    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Time used: ${elapsed} seconds" | tee -a "$run_dir/result.txt"
    echo "Run completed at $(date)" | tee -a "$run_dir/result.txt"
}

# $1 = start
# $2 = end
# $3 = exp_dir
# $4 = min words
# $5 = max words
# $6 = model name (one of: sd3, flux)

exp_dir="$3"
mkdir -p "$exp_dir"

global_start_time=$(date +%s)

START="$1"
END="$2"

echo -e "Starting runs from $START to $END in directory $exp_dir\n"

min_words="$4"
max_words="$5"
model_name="$6"

run
for run in $(seq "$START" "$END"); do
    echo -e "===== Starting Run #$run =====\n"
    run_pipeline "$run" true "$min_words" "$max_words" "$model_name"
done

global_end_time=$(date +%s)
elapsed=$((global_end_time - global_start_time))
echo "Total time used: ${elapsed} seconds"
echo "Run completed at $(date)"
