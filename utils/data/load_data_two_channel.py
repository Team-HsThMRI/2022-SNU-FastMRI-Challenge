import h5py
import random
from utils.data.transforms_two_channel import DataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SliceData(Dataset):
    def __init__(self, root, transform, input_key_1, input_key_2, target_key, forward=False):
        self.transform = transform
        self.input_key_1 = input_key_1
        self.input_key_2 = input_key_2
        self.target_key = target_key
        self.forward = forward
        self.examples = []

        files = list(Path(root).iterdir())
        for fname in sorted(files):
            num_slices = self._get_metadata(fname)

            self.examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            num_slices = hf[self.input_key_1].shape[0]
        return num_slices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, dataslice = self.examples[i]
        with h5py.File(fname, "r") as hf:
            input_1 = hf[self.input_key_1][dataslice]
            input_2 = hf[self.input_key_2][dataslice]
            if self.forward:
                target = -1
            else:
                target = hf[self.target_key][dataslice]
            attrs = dict(hf.attrs)
        return self.transform(input_1, input_2, target, attrs, fname.name, dataslice)


def create_data_loaders(data_path, args, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=DataTransform(isforward, max_key_),
        input_key_1=args.input_key_1,
        input_key_2=args.input_key_2,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size
    )
    return data_loader
