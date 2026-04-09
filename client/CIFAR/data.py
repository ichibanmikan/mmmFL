from pathlib import Path

import torch
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader, Dataset

try:
    from pickle_compat import load_pickle_file
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from pickle_compat import load_pickle_file


class CIFARDataset(Dataset):
    def __init__(self, data_path, split):
        super().__init__()
        self.split = split
        payload = self._load_payload(Path(data_path))
        self.images, self.labels = self._extract_split(payload, split)
        self.images = torch.as_tensor(self.images, dtype=torch.float32)
        self.labels = torch.as_tensor(self.labels, dtype=torch.long)

    def _load_payload(self, data_path):
        return load_pickle_file(data_path)

    def _extract_split(self, payload, split):
        image_key = f"x_{split}"
        label_key = f"y_{split}"
        if image_key in payload and label_key in payload:
            return payload[image_key], payload[label_key]
        raise KeyError(f"Split {split} not found in {payload.keys()}")

    def _augment_train_image(self, image):
        image = torch_f.pad(image.unsqueeze(0), (4, 4, 4, 4), value=0.0).squeeze(0)
        top = torch.randint(0, 9, (1,)).item()
        left = torch.randint(0, 9, (1,)).item()
        image = image[:, top:top + 32, left:left + 32]
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=(2,))
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.split == "train":
            image = self._augment_train_image(image)
        return image, self.labels[index]


class data_factory:
    def __init__(self, datasets_dir, node_id, config):
        self.datasets_dir = Path(datasets_dir)
        self.config = config
        self.node_id = node_id
        validation_path = self.datasets_dir / "validation.pickle"
        test_path = self.datasets_dir / "test.pickle"
        if not validation_path.exists():
            validation_path = self.datasets_dir / "data.pickle"
        if not test_path.exists():
            test_path = self.datasets_dir / "data.pickle"
        self.train_set = CIFARDataset(self.datasets_dir / f"node_{node_id}" / "train.pickle", "train")
        self.validation_set = CIFARDataset(validation_path, "validation")
        self.test_set = CIFARDataset(test_path, "test")
        self.sample_length = len(self.train_set)

    def _build_loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def get_train_loader(self):
        return self._build_loader(self.train_set, True)

    def get_validation_loader(self):
        return self._build_loader(self.validation_set, False)

    def get_test_loader(self):
        return self._build_loader(self.test_set, False)
