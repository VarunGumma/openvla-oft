"""Builds one merged RLDS dataset from preprocessed RoboTwin ALOHA episodes."""

import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds

sys.path.append(str(Path(__file__).resolve().parents[1]))
from aloha1_put_X_into_pot_300_demos.conversion_utils import MultiThreadedDatasetBuilder  # noqa: E402


DATA_ROOT_ENV = "ALOHA_ROBOTWIN_PREPROCESSED_DIR"


def _choose_instruction(episode_path: str) -> str:
    instructions_path = Path(episode_path).parent / "instructions.json"
    with instructions_path.open("r", encoding="utf-8") as f:
        instructions = json.load(f)["instructions"]
    if not instructions:
        raise ValueError(f"No instructions found in {instructions_path}")

    seed = int(hashlib.sha256(episode_path.encode("utf-8")).hexdigest(), 16)
    return random.Random(seed).choice(instructions)


def _load_annotations(episode_path: str) -> list:
    annotations_path = Path(episode_path).parent / "annotations.txt"
    with annotations_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields preprocessed episodes from all RoboTwin tasks."""

    def _parse_example(episode_path):
        with h5py.File(episode_path, "r") as root:
            actions = root["/action"][()]
            states = root["/observations/qpos"][()]
            images = root["/observations/images/cam_high"][()]
            left_wrist_images = root["/observations/images/cam_left_wrist"][()]
            right_wrist_images = root["/observations/images/cam_right_wrist"][()]

        annotations = _load_annotations(episode_path)
        lengths = {
            "action": len(actions),
            "state": len(states),
            "cam_high": len(images),
            "cam_left_wrist": len(left_wrist_images),
            "cam_right_wrist": len(right_wrist_images),
            "annotation": len(annotations),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Preprocessed episode is not aligned: {episode_path}: {lengths}")

        instruction = _choose_instruction(episode_path)
        episode = []
        for i in range(len(actions)):
            episode.append(
                {
                    "observation": {
                        "image": images[i],
                        "left_wrist_image": left_wrist_images[i],
                        "right_wrist_image": right_wrist_images[i],
                        "state": np.asarray(states[i], np.float32),
                    },
                    "action": np.asarray(actions[i], dtype=np.float32),
                    "discount": 1.0,
                    "reward": float(i == (len(actions) - 1)),
                    "is_first": i == 0,
                    "is_last": i == (len(actions) - 1),
                    "is_terminal": i == (len(actions) - 1),
                    "language_instruction": instruction,
                    "annotation": annotations[i],
                }
            )

        sample = {
            "steps": episode,
            "episode_metadata": {
                "file_path": episode_path,
                "task_name": Path(episode_path).parents[2].name,
            },
        }
        return episode_path, sample

    for path in paths:
        yield _parse_example(path)


class aloha_robotwin_10000_demos(MultiThreadedDatasetBuilder):
    """Merged RoboTwin ALOHA demonstration dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release."}
    N_WORKERS = 40
    MAX_PATHS_IN_MEMORY = 80
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main head-camera RGB observation.",
                                    ),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Left wrist-camera RGB observation.",
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Right wrist-camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(14,),
                                        dtype=np.float32,
                                        doc="Bimanual joint state: left arm, left gripper, right arm, right gripper.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Absolute bimanual joint-position action.",
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(doc="Episode-level task instruction."),
                            "annotation": tfds.features.Text(doc="Frame-level opcode annotation."),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(doc="Path to the preprocessed episode HDF5 file."),
                            "task_name": tfds.features.Text(doc="RoboTwin task directory name."),
                        }
                    ),
                }
            )
        )

    def _split_paths(self):
        data_root = os.environ.get(DATA_ROOT_ENV)
        if not data_root:
            raise ValueError(f"Set {DATA_ROOT_ENV} to the preprocessed RoboTwin ALOHA directory.")

        return {
            "train": sorted(glob.glob(os.path.join(data_root, "*", "train", "episode_*", "*.hdf5"))),
            "val": sorted(glob.glob(os.path.join(data_root, "*", "val", "episode_*", "*.hdf5"))),
        }
