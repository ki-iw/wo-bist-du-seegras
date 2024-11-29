from typing import Optional

from torch.utils.data import Dataset

from zug_seegras.core.datumaru_processor import DatumaroProcessor


class SeegrasDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        processor=DatumaroProcessor,
        transform: Optional[any] = None,  # noqa: UP007
    ) -> None:
        self.image_dir = image_dir
        self.transform = transform

        self.frame_ids, self.labels = processor(label_dir).get_frame_labels()

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        image = None
        label = None

        if self.transform:
            image = self.transform(image)

        return image, label
