import argparse

import torch
from torchvision.transforms import Compose, ToTensor

from src.dataset.image import ImageDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--real-dir", action="append", default=[])
    parser.add_argument("--fake-dir", action="append", default=[])
    parser.add_argument("--protocol-json", default=None, type=str)
    parser.add_argument("--protocol-json-folder", default=None, type=str)
    parser.add_argument("--protocol-dataset-name", default=None, type=str)
    parser.add_argument("--protocol-split", default=None, type=str)
    parser.add_argument("--protocol-compression", default=None, type=str)
    parser.add_argument("--protocol-root", default=None, type=str)
    parser.add_argument("--protocol-real-label", action="append", default=[])
    parser.add_argument("--protocol-label-map", action="append", default=[])
    parser.add_argument("--protocol-path-rewrite", action="append", default=[])
    parser.add_argument("--num-samples", default=1, type=int)
    args = parser.parse_args()

    label_map = {}
    for pair in args.protocol_label_map:
        if "=" not in pair:
            raise ValueError("--protocol-label-map must be in LABEL=INT format")
        label, value = pair.split("=", 1)
        label_map[label] = int(value)

    path_rewrites = []
    for pair in args.protocol_path_rewrite:
        if "->" not in pair:
            raise ValueError("--protocol-path-rewrite must be in SRC->DST format")
        src, dst = pair.split("->", 1)
        path_rewrites.append((src, dst))

    dataset = ImageDF(
        data_dir=args.data_dir,
        vid_ext="",
        num_frames=1,
        clip_duration=1,
        split=args.split,
        transform=Compose([ToTensor()]),
        pack=False,
        real_dirs=args.real_dir or None,
        fake_dirs=args.fake_dir or None,
        recursive=True,
        protocol_json=args.protocol_json,
        protocol_json_folder=args.protocol_json_folder,
        protocol_dataset_name=args.protocol_dataset_name,
        protocol_split=args.protocol_split,
        protocol_compression=args.protocol_compression,
        protocol_root=args.protocol_root,
        protocol_real_labels=args.protocol_real_label or None,
        protocol_label_map=label_map or None,
        protocol_path_rewrites=path_rewrites or None
    )

    for idx in range(min(args.num_samples, len(dataset))):
        entity = dataset.get_entity(idx, with_entity_info=True)
        clips = entity["clips"]
        print(
            f"#{idx} {entity['df_type']} {entity['img_path']} -> clips shape {tuple(clips.shape)} "
            f"dtype={clips.dtype}"
        )

    if len(dataset) == 0:
        print("No images found. Check your dataset path and extensions.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
