# ALOHA RoboTwin 10,000 Demonstrations

This builder converts preprocessed RoboTwin ALOHA demonstrations into one merged
RLDS dataset. Each frame contains three RGB camera observations, a 14-dimensional
bimanual joint state, a 14-dimensional absolute joint-position action, one
episode-level language instruction, and one frame-level opcode annotation.

## Preprocess And Split

The input directory is expected to contain task directories and episode
directories:

```text
/PATH/TO/aloha-agilex/
    adjust_bottle/
        episode_0/
            episode_0.hdf5
            annotations.txt
            instructions.json
```

Run:

```bash
python preprocess_robotwin_aloha.py \
    --input_dir /PATH/TO/aloha-agilex \
    --output_dir /PATH/TO/aloha-agilex-preprocessed \
    --percent_val 0.05 \
    --seed 7 \
    --num_workers 16
```

The script supports the RoboTwin joint-action HDF5 structure and the aligned
ALOHA-style HDF5 structure emitted by the RoboTwin conversion script. It creates
next-state action targets, truncates annotations to the number of emitted
transitions, resizes and JPEG-encodes images at `256x256`, and splits episodes
per task. Use `--jpeg_quality` to override the default quality of `95`.

## Build One Merged RLDS Dataset

Set the preprocessed data root and run TFDS from this directory:

```bash
export ALOHA_ROBOTWIN_PREPROCESSED_DIR=/PATH/TO/aloha-agilex-preprocessed
tfds build --overwrite
```

The output dataset is named `aloha_robotwin_10000_demos`.

## Train With Annotations

Before training, add `<opcodes>` and `</opcodes>` as regular added tokens and
resize the model embeddings. Then select the merged dataset and the L1 action
head. Annotation prediction is disabled by default and is enabled explicitly
with `--use_annotation_prediction True`:

```bash
python ../../vla-scripts/finetune.py \
    --data_root_dir ~/tensorflow_datasets \
    --dataset_name aloha_robotwin_10000_demos \
    --use_l1_regression True \
    --use_annotation_prediction True \
    --annotation_action_l1_alpha 10 \
    --use_annotation_margin_loss True \
    --annotation_margin_lambda 1 \
    --annotation_margin_gamma 0.01 \
    --num_images_in_input 3 \
    --use_proprio True \
    --use_val_set True
```

Annotation-aware batches optimize annotation cross-entropy plus
`annotation_action_l1_alpha * action_l1_loss`. If
`--use_annotation_margin_loss True` and `--annotation_margin_lambda` is
positive, training also adds
`annotation_margin_lambda * relu(annotation_margin_gamma - control_action_mse + annotation_action_mse)`.
The control pass keeps the same token sequence but masks annotation tokens from
attention, so action slots cannot use the frame-level opcode text.
