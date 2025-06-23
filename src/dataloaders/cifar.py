import os

import numpy as np
import torch
from sklearn.utils import shuffle
from torchvision import datasets, transforms


def get(seed: int = 0, pc_valid: float = 0.10):
    """
    Build a 10-task split of CIFAR-100.
      • Task 0 holds classes   0- 9
      • Task 1 holds classes  10-19
      • …
      • Task 9 holds classes  90-99

    Returns
    -------
    data      : dict  – task-indexed tensors for train / valid / test
    taskcla   : list  – list[(task_id, ncla)]
    size      : list  – input image size, [3, 32, 32]
    """
    num_tasks = 10
    classes_per_task = 10
    root_bin = os.path.expanduser("../dat/cifar100_split10")  # <-- new folder
    size = [3, 32, 32]
    data, taskcla = {}, []

    # ------------------------------------------------------------------
    # 1.  Create binary files once (run only the first time)
    # ------------------------------------------------------------------
    if not os.path.isdir(root_bin):
        os.makedirs(root_bin, exist_ok=True)

        # normalise with torchvision's convention
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        full = {
            "train": datasets.CIFAR100(
                "../dat/", train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            ),
            "test": datasets.CIFAR100(
                "../dat/", train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            ),
        }

        # initialise per-task containers
        data_split = {
            tid: {
                "name": f"cifar100-{tid}",
                "ncla": classes_per_task,
                "train": {"x": [], "y": []},
                "test": {"x": [], "y": []},
            }
            for tid in range(num_tasks)
        }

        # fill the splits
        for split in ["train", "test"]:
            loader = torch.utils.data.DataLoader(full[split], batch_size=1, shuffle=False)
            for img, target in loader:
                cls = target.item()
                tid = cls // classes_per_task
                data_split[tid][split]["x"].append(img)
                data_split[tid][split]["y"].append(cls % classes_per_task)

        # stack & save
        for tid, obj in data_split.items():
            for split in ["train", "test"]:
                xs = torch.cat(obj[split]["x"], 0)
                ys = torch.LongTensor(obj[split]["y"])
                torch.save(xs, os.path.join(root_bin, f"task{tid}_{split}x.pt"))
                torch.save(ys, os.path.join(root_bin, f"task{tid}_{split}y.pt"))

    # ------------------------------------------------------------------
    # 2.  Load binaries (fast on every run)
    # ------------------------------------------------------------------
    idx_order = np.arange(num_tasks)
    print("Task order =", idx_order)

    for pos, tid in enumerate(idx_order):
        task_data = {"train": {}, "test": {}}
        for split in ["train", "test"]:
            task_data[split]["x"] = torch.load(
                os.path.join(root_bin, f"task{tid}_{split}x.pt")
            )
            task_data[split]["y"] = torch.load(
                os.path.join(root_bin, f"task{tid}_{split}y.pt")
            )
        task_data["name"] = f"cifar100-{tid}"
        task_data["ncla"] = classes_per_task
        data[pos] = task_data

    # ------------------------------------------------------------------
    # 3.  Build validation splits
    # ------------------------------------------------------------------
    for pos in data:
        r = shuffle(np.arange(len(data[pos]["train"]["x"])), random_state=seed)
        n_val = int(pc_valid * len(r))
        idx_val = torch.LongTensor(r[:n_val])
        idx_trn = torch.LongTensor(r[n_val:])

        data[pos]["valid"] = {
            "x": data[pos]["train"]["x"][idx_val].clone(),
            "y": data[pos]["train"]["y"][idx_val].clone(),
        }
        data[pos]["train"]["x"] = data[pos]["train"]["x"][idx_trn].clone()
        data[pos]["train"]["y"] = data[pos]["train"]["y"][idx_trn].clone()

    # ------------------------------------------------------------------
    # 4.  Meta-info
    # ------------------------------------------------------------------
    total_classes = 0
    for pos in data:
        taskcla.append((pos, data[pos]["ncla"]))
        total_classes += data[pos]["ncla"]
    data["ncla"] = total_classes

    return data, taskcla, size
