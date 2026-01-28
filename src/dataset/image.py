from .base import *

import json
from os import walk
from PIL import Image


class ImageDF(DeepFakeDataset):
    TYPE_DIRS = {
        "REAL": "REAL",
        "FAKE": "FAKE"
    }

    @classmethod
    def prepare_data(cls, *args, **kargs):
        # image datasets are scanned on-the-fly
        return

    def __init__(
        self,
        img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        real_dirs: Optional[List[str]] = None,
        fake_dirs: Optional[List[str]] = None,
        recursive: bool = True,
        protocol_json: Optional[str] = None,
        protocol_split: Optional[str] = None,
        protocol_root: Optional[str] = None,
        protocol_real_labels: Optional[List[str]] = None,
        protocol_json_folder: Optional[str] = None,
        protocol_dataset_name: Optional[str] = None,
        protocol_compression: Optional[str] = None,
        protocol_label_map: Optional[Dict[str, int]] = None,
        protocol_path_rewrites: Optional[List[Tuple[str, str]]] = None,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.img_exts = tuple([ext.lower() for ext in img_exts])
        self.real_dirs = real_dirs or []
        self.fake_dirs = fake_dirs or []
        self.recursive = recursive
        self.protocol_json = protocol_json
        self.protocol_split = protocol_split
        self.protocol_root = protocol_root
        self.protocol_real_labels = [label.lower() for label in (protocol_real_labels or [])]
        self.protocol_json_folder = protocol_json_folder
        self.protocol_dataset_name = protocol_dataset_name
        self.protocol_compression = protocol_compression
        self.protocol_label_map = protocol_label_map or {}
        self.protocol_path_rewrites = protocol_path_rewrites or []
        if self.num_frames != 1:
            logging.warning(
                f"Image dataset expects num_frames=1, but got {self.num_frames}; overriding to 1."
            )
            self.num_frames = 1
        self._build_image_list()

    def _base_split_dir(self):
        split_dir = path.join(self.data_dir, self.split)
        return split_dir if path.isdir(split_dir) else self.data_dir

    def _build_image_list(self):
        self.image_list = []
        base_dir = self._base_split_dir()

        if self.protocol_json or (self.protocol_json_folder and self.protocol_dataset_name):
            self._build_image_list_from_protocol(base_dir)
            return

        real_dirs = self._normalize_dirs(self.real_dirs)
        fake_dirs = self._normalize_dirs(self.fake_dirs)
        if not fake_dirs:
            fake_dirs = [base_dir]

        if real_dirs or fake_dirs != [base_dir]:
            for img_dir in real_dirs:
                self._collect_images(img_dir, "REAL")
            for img_dir in fake_dirs:
                self._collect_images(img_dir, "FAKE")
        else:
            for df_type, subdir in self.TYPE_DIRS.items():
                img_dir = path.join(base_dir, subdir)
                self._collect_images(img_dir, df_type)

        random.Random(1019).shuffle(self.image_list)
        if self.ratio < 1.0:
            self.image_list = self.image_list[:int(len(self.image_list) * self.ratio)]

        if len(self.image_list) == 0:
            logging.warning(f"No images found under {base_dir} with extensions {self.img_exts}.")

        self.stack_video_clips = list(range(1, len(self.image_list) + 1))

    def __len__(self):
        return len(self.image_list)

    def _build_image_list_from_protocol(self, base_dir: str):
        protocol_path = self._resolve_protocol_path()

        if not path.exists(protocol_path):
            raise FileNotFoundError(f"Protocol JSON not found: {protocol_path}")

        with open(protocol_path, "r") as f:
            payload = json.load(f)

        entries = []
        split = self.protocol_split
        payload = self._select_protocol_dataset(payload)
        if self.protocol_compression:
            payload = payload.get(self.protocol_compression, payload)
        self._collect_protocol_entries(payload, entries, split=split, in_split=(split is None))

        if len(entries) == 0:
            logging.warning(f"No entries found in protocol JSON for split '{split}'.")

        for label, frames in entries:
            df_type = self._protocol_label_to_df_type(label)
            for frame_path in frames:
                resolved = self._resolve_frame_path(frame_path, base_dir)
                ext = path.splitext(resolved)[1].lower()
                if ext in self.img_exts:
                    self.image_list.append((df_type, resolved))

    def _resolve_protocol_path(self) -> str:
        if self.protocol_json:
            protocol_path = self.protocol_json
            return protocol_path if path.isabs(protocol_path) else path.join(self.data_dir, protocol_path)
        protocol_folder = self.protocol_json_folder
        if not protocol_folder:
            raise ValueError("protocol_json_folder must be provided when protocol_json is not set.")
        protocol_folder = protocol_folder if path.isabs(protocol_folder) else path.join(self.data_dir, protocol_folder)
        dataset_name = self.protocol_dataset_name
        if not dataset_name:
            raise ValueError("protocol_dataset_name must be provided when protocol_json is not set.")
        return path.join(protocol_folder, f"{dataset_name}.json")

    def _select_protocol_dataset(self, payload: Dict) -> Dict:
        if self.protocol_dataset_name and self.protocol_dataset_name in payload:
            return payload[self.protocol_dataset_name]
        return payload

    def _collect_protocol_entries(self, obj, entries, split: Optional[str], in_split: bool):
        if isinstance(obj, dict):
            if split and split in obj:
                self._collect_protocol_entries(obj[split], entries, split=split, in_split=True)

            if in_split and "frames" in obj and "label" in obj:
                entries.append((obj["label"], obj["frames"]))

            for value in obj.values():
                self._collect_protocol_entries(value, entries, split=split, in_split=in_split)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_protocol_entries(item, entries, split=split, in_split=in_split)

    def _protocol_label_to_df_type(self, label) -> str:
        label_str = str(label)
        if label_str in self.protocol_label_map:
            return "REAL" if self.protocol_label_map[label_str] == 0 else "FAKE"
        if isinstance(label, (int, float)):
            return "REAL" if int(label) == 0 else "FAKE"
        label_lower = label_str.lower()
        if label_lower in self.protocol_real_labels:
            return "REAL"
        if "real" in label_lower:
            return "REAL"
        return "FAKE"

    def _resolve_frame_path(self, frame_path: str, base_dir: str) -> str:
        if path.isabs(frame_path):
            return self._rewrite_protocol_path(frame_path)
        root = self.protocol_root or base_dir
        return self._rewrite_protocol_path(path.join(root, frame_path))

    def _rewrite_protocol_path(self, frame_path: str) -> str:
        for src, dst in self.protocol_path_rewrites:
            if frame_path.startswith(src):
                return frame_path.replace(src, dst, 1)
        return frame_path

    def _normalize_dirs(self, dirs: List[str]) -> List[str]:
        resolved = []
        for d in dirs:
            resolved.append(d if path.isabs(d) else path.join(self.data_dir, d))
        return resolved

    def _collect_images(self, img_dir: str, df_type: str):
        if not path.isdir(img_dir):
            logging.warning(f"Image directory not found: {img_dir}")
            return

        if self.recursive:
            for root, _, files in walk(img_dir):
                for filename in files:
                    ext = path.splitext(filename)[1].lower()
                    if ext in self.img_exts:
                        self.image_list.append((df_type, path.join(root, filename)))
        else:
            for f in scandir(img_dir):
                if not f.is_file():
                    continue
                ext = path.splitext(f.name)[1].lower()
                if ext in self.img_exts:
                    self.image_list.append((df_type, f.path))

    def __getitem__(self, idx):
        item_entities = self.get_item(idx)
        return [[entity[k] for entity in item_entities] for k in item_entities[0].keys()]

    def get_item(self, idx, with_entity_info=False):
        item_entities = [
            self.get_entity(
                idx,
                with_entity_info=with_entity_info
            )
        ]
        return item_entities

    def get_entity(self, idx, with_entity_info=False):
        df_type, img_path = self.image_list[idx]

        image = Image.open(img_path).convert("RGB")
        frame = self.transform(image)
        if not isinstance(frame, torch.Tensor):
            frame = torch.as_tensor(frame)

        if frame.dim() == 3:
            frames = frame.unsqueeze(0)
        elif frame.dim() == 4 and frame.shape[0] == self.num_frames:
            frames = frame
        else:
            raise ValueError(
                f"Unexpected frame shape {tuple(frame.shape)}; expected (C,H,W) or (1,C,H,W)."
            )

        entity_clips = frames.unsqueeze(0)
        entity_masks = torch.ones((1, frames.shape[0]), dtype=torch.bool)

        entity_info = {
            "df_type": df_type,
            "img_path": img_path
        }
        entity_data = {
            "clips": entity_clips,
            "label": 0 if df_type == "REAL" else 1,
            "masks": entity_masks,
            "idx": idx
        }

        if with_entity_info:
            return {**entity_data, **entity_info}
        else:
            return entity_data

    def video_info(self, idx):
        df_type, img_path = self.image_list[idx]
        return idx, df_type, img_path, 1

    def video_meta(self, idx):
        df_type, img_path = self.image_list[idx]
        return dict(
            df_type=df_type,
            img_path=img_path
        )

    def video_repr(self, idx):
        return self.image_list[idx][1]


class ImageDataModule(DeepFakeDataModule):
    def __init__(
        self,
        img_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        real_dirs: Optional[List[str]] = None,
        fake_dirs: Optional[List[str]] = None,
        recursive: bool = True,
        protocol_json: Optional[str] = None,
        protocol_split: Optional[str] = None,
        protocol_root: Optional[str] = None,
        protocol_real_labels: Optional[List[str]] = None,
        protocol_json_folder: Optional[str] = None,
        protocol_dataset_name: Optional[str] = None,
        protocol_compression: Optional[str] = None,
        protocol_label_map: Optional[Dict[str, int]] = None,
        protocol_path_rewrites: Optional[List[Tuple[str, str]]] = None,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.img_exts = img_exts
        self.real_dirs = real_dirs
        self.fake_dirs = fake_dirs
        self.recursive = recursive
        self.protocol_json = protocol_json
        self.protocol_split = protocol_split
        self.protocol_root = protocol_root
        self.protocol_real_labels = protocol_real_labels
        self.protocol_json_folder = protocol_json_folder
        self.protocol_dataset_name = protocol_dataset_name
        self.protocol_compression = protocol_compression
        self.protocol_label_map = protocol_label_map
        self.protocol_path_rewrites = protocol_path_rewrites

    def prepare_data(self):
        ImageDF.prepare_data()

    def setup(self, stage: str):
        data_cls = partial(
            ImageDF,
            data_dir=self.data_dir,
            vid_ext=self.vid_ext,
            num_frames=self.num_frames,
            clip_duration=self.clip_duration,
            transform=self.transform,
            ratio=self.ratio,
            pack=self.pack,
            max_clips=self.max_clips,
            img_exts=self.img_exts,
            real_dirs=self.real_dirs,
            fake_dirs=self.fake_dirs,
            recursive=self.recursive,
            protocol_json=self.protocol_json,
            protocol_split=self.protocol_split,
            protocol_root=self.protocol_root,
            protocol_real_labels=self.protocol_real_labels,
            protocol_json_folder=self.protocol_json_folder,
            protocol_dataset_name=self.protocol_dataset_name,
            protocol_compression=self.protocol_compression,
            protocol_label_map=self.protocol_label_map,
            protocol_path_rewrites=self.protocol_path_rewrites
        )

        if stage == "fit" or stage == "validate":
            self._train_dataset = data_cls(split="train")
            self._val_dataset = data_cls(split="val")
        elif stage == "test":
            self._test_dataset = data_cls(split="test")
        elif stage == "predict":
            self._predict_dataset = data_cls(split="test")


if __name__ == "__main__":
    from src.utility.visualize import dataset_entity_visualize

    class Dummy():
        pass

    dtm = ImageDataModule(
        data_dir="datasets_image",
        vid_ext="",
        batch_size=1,
        num_workers=0,
        num_frames=1,
        clip_duration=1,
        ratio=1.0,
        pack=False
    )

    model = Dummy()
    model.transform = lambda x: x
    dtm.prepare_data()
    dtm.affine_model(model)
    dtm.setup("fit")
    dtm.setup("validate")
    dtm.setup("test")

    # iterate the whole dataset for visualization and sanity check
    iterable = dtm._test_dataset
    save_folder = f"./misc/extern/dump_dataset/image/test/"

    # # entity dump
    # for entity_idx in tqdm(range(len(iterable))):
    #     if (entity_idx > 100):
    #         break
    #     dataset_entity_visualize(iterable.get_entity(entity_idx, with_entity_info=True), base_dir=save_folder)

    # # single dump
    # dataset_entity_visualize(iterable.get_entity(0, with_entity_info=True), base_dir=save_folder)

    # iterate the all dataloaders for debugging.
    for fn in [dtm.train_dataloader, dtm.val_dataloader, dtm.test_dataloader]:
        iterable = fn()
        if iterable is None:
            continue
        for batch in tqdm(iterable):
            pass
