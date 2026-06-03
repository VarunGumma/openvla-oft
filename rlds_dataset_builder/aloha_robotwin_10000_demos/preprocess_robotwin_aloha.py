"""Preprocesses and splits RoboTwin ALOHA episodes for RLDS conversion."""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ROBOTWIN_CAMERA_NAMES = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _decode_and_resize_image(value: np.ndarray, image_size: int) -> np.ndarray:
    """Decodes a JPEG-like HDF5 value or resizes an already decoded RGB image."""
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[-1] == 3:
        image = Image.fromarray(value.astype(np.uint8)).convert("RGB")
    else:
        encoded = value.tobytes().rstrip(b"\0")
        image = Image.open(BytesIO(encoded)).convert("RGB")

    resampling = getattr(Image, "Resampling", Image)
    image = image.resize((image_size, image_size), resample=resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _encode_resized_images(values: np.ndarray, image_size: int, jpeg_quality: int) -> np.ndarray:
    """Resizes images and stores compact JPEG bytes in the intermediate HDF5."""
    encoded_images = []
    for value in values:
        image = Image.fromarray(_decode_and_resize_image(value, image_size))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        encoded_images.append(buffer.getvalue())

    max_length = max(len(image) for image in encoded_images)
    return np.asarray(encoded_images, dtype=f"S{max_length}")


def _as_scalar(value: np.ndarray) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _load_annotations(episode_dir: Path) -> List[str]:
    annotations_path = episode_dir / "annotations.txt"
    with annotations_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def _load_instructions(episode_dir: Path) -> Dict[str, List[str]]:
    instructions_path = episode_dir / "instructions.json"
    with instructions_path.open("r", encoding="utf-8") as f:
        instructions = json.load(f)

    if not isinstance(instructions.get("instructions"), list) or not instructions["instructions"]:
        raise ValueError(f"Missing non-empty instructions list: {instructions_path}")
    return instructions


def _load_robotwin_episode(root: h5py.File) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Loads RoboTwin joint actions and aligns state[t] with next-state action[t]."""
    left_gripper = root["/joint_action/left_gripper"][()]
    left_arm = root["/joint_action/left_arm"][()]
    right_gripper = root["/joint_action/right_gripper"][()]
    right_arm = root["/joint_action/right_arm"][()]
    num_states = min(len(left_gripper), len(left_arm), len(right_gripper), len(right_arm))

    states = []
    for i in range(num_states):
        state = np.concatenate(
            [
                np.asarray(left_arm[i]).reshape(-1),
                np.asarray([_as_scalar(left_gripper[i])]),
                np.asarray(right_arm[i]).reshape(-1),
                np.asarray([_as_scalar(right_gripper[i])]),
            ]
        )
        states.append(state.astype(np.float32))
    states = np.asarray(states, dtype=np.float32)

    images = {
        output_name: root[f"/observation/{input_name}/rgb"][()]
        for output_name, input_name in ROBOTWIN_CAMERA_NAMES.items()
    }
    return states[:-1], states[1:], {name: values[:-1] for name, values in images.items()}


def _load_aligned_episode(root: h5py.File) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Loads an episode already aligned by the RoboTwin conversion script."""
    qpos = np.asarray(root["/observations/qpos"][()], dtype=np.float32)
    actions = np.asarray(root["/action"][()], dtype=np.float32)
    images = {name: root[f"/observations/images/{name}"][()] for name in CAMERA_NAMES}
    return qpos, actions, images


def _write_episode(
    source_episode_dir: Path,
    output_episode_dir: Path,
    image_size: int,
    jpeg_quality: int,
    overwrite: bool,
) -> str:
    output_hdf5_path = output_episode_dir / f"{source_episode_dir.name}.hdf5"
    if output_hdf5_path.exists() and not overwrite:
        return f"Skipped existing {output_hdf5_path}"

    source_hdf5_path = source_episode_dir / f"{source_episode_dir.name}.hdf5"
    annotations = _load_annotations(source_episode_dir)
    instructions = _load_instructions(source_episode_dir)

    with h5py.File(source_hdf5_path, "r") as root:
        if "/joint_action/left_arm" in root:
            qpos, actions, image_dict = _load_robotwin_episode(root)
        elif "/observations/qpos" in root and "/action" in root:
            qpos, actions, image_dict = _load_aligned_episode(root)
        else:
            raise ValueError(f"Unsupported HDF5 structure: {source_hdf5_path}")

    lengths = {
        "qpos": len(qpos),
        "action": len(actions),
        "annotations": len(annotations),
        **{name: len(images) for name, images in image_dict.items()},
    }
    num_transitions = min(lengths.values())
    if num_transitions <= 0:
        raise ValueError(f"Episode has no usable transitions: {source_episode_dir}")

    if len(set(lengths.values())) != 1:
        print(f"Truncating {source_episode_dir}: {lengths} -> {num_transitions} transitions")

    qpos = qpos[:num_transitions]
    actions = actions[:num_transitions]
    annotations = annotations[:num_transitions]
    if qpos.shape[1:] != (14,) or actions.shape[1:] != (14,):
        raise ValueError(f"Expected 14D qpos and actions in {source_episode_dir}: {qpos.shape=}, {actions.shape=}")

    encoded_images = {
        name: _encode_resized_images(images[:num_transitions], image_size, jpeg_quality)
        for name, images in image_dict.items()
    }

    output_episode_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_hdf5_path, "w") as root:
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos, dtype=np.float32)
        image_group = obs.create_group("images")
        for name, images in encoded_images.items():
            image_group.create_dataset(name, data=images, dtype=images.dtype)
        root.create_dataset("action", data=actions, dtype=np.float32)

    with (output_episode_dir / "annotations.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(annotations))
    with (output_episode_dir / "instructions.json").open("w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2)

    return f"Wrote {output_hdf5_path} ({num_transitions} transitions)"


def _task_episode_dirs(input_dir: Path) -> Dict[str, List[Path]]:
    tasks = {}
    for task_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        episode_dirs = sorted(
            path
            for path in task_dir.glob("episode_*")
            if path.is_dir() and (path / f"{path.name}.hdf5").is_file()
        )
        if episode_dirs:
            tasks[task_dir.name] = episode_dirs
    return tasks


def _split_task_episodes(episode_dirs: List[Path], percent_val: float, seed: int) -> Dict[str, List[Path]]:
    episode_dirs = list(episode_dirs)
    random.Random(seed).shuffle(episode_dirs)
    num_val = int(len(episode_dirs) * percent_val)
    return {
        "train": episode_dirs[num_val:],
        "val": episode_dirs[:num_val],
    }


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not 0 <= args.percent_val < 1:
        raise ValueError("--percent_val must be in [0, 1).")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100].")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive.")

    tasks = _task_episode_dirs(input_dir)
    if not tasks:
        raise ValueError(f"No task/episode_* directories found under {input_dir}")

    jobs = []
    for task_idx, (task_name, episode_dirs) in enumerate(tasks.items()):
        splits = _split_task_episodes(episode_dirs, args.percent_val, args.seed + task_idx)
        print(f"{task_name}: {len(splits['train'])} train, {len(splits['val'])} val episodes")
        for split_name, split_episode_dirs in splits.items():
            for source_episode_dir in split_episode_dirs:
                jobs.append(
                    (
                        source_episode_dir,
                        output_dir / task_name / split_name / source_episode_dir.name,
                        args.image_size,
                        args.jpeg_quality,
                        args.overwrite,
                    )
                )

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(_write_episode, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing episodes"):
            print(future.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing task/episode_* directories.")
    parser.add_argument("--output_dir", required=True, help="Directory for split and resized episodes.")
    parser.add_argument("--percent_val", type=float, default=0.05, help="Per-task validation fraction.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for per-task splitting.")
    parser.add_argument("--image_size", type=int, default=256, help="Square output image size.")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for compact intermediate images.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of preprocessing workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed HDF5 files.")
    main(parser.parse_args())
