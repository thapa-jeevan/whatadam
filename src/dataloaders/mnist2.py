import torch, random, numpy as np
from torchvision import datasets, transforms


def get(seed: int = 0, fixed_order: bool = False, pc_valid: float = 0.):
    """
    Returns
    -------
    data      : dict  – same structure you already use
    taskcla   : list  – [(task-id, ncla), ...]
    size      : list  – [input-channels, H, W]
    """

    # --------------------------------------------------------------------- #
    # 1.  Prep
    # --------------------------------------------------------------------- #
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    size = [1, 28, 28]                 # (C, H, W)
    mean, std = (0.1307,), (0.3081,)    # MNIST stats

    mnist_train = datasets.MNIST(
        '../dat', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean, std)]))
    mnist_test = datasets.MNIST(
        '../dat', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean, std)]))

    # --------------------------------------------------------------------- #
    # 2.  Decide task order (five tasks, two consecutive digits each)
    # --------------------------------------------------------------------- #
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    # --------------------------------------------------------------------- #
    # 3.  Split data
    # --------------------------------------------------------------------- #
    data, taskcla = {}, []
    for t, (lo, hi) in enumerate(pairs):
        data[t] = {'name': f'mnist-{lo}-{hi}', 'ncla': 2,
                   'train': {'x': [], 'y': []},
                   'test':  {'x': [], 'y': []}}

    # helper to push samples into the right task bucket
    def _add_sample(split_dict, img, lbl):
        for t, (lo, hi) in enumerate(pairs):
            if lo <= lbl <= hi:
                split_dict[t]['x'].append(img)
                split_dict[t]['y'].append(lbl - lo)   # relabel to {0,1}
                return

    for split_name, dataset in [('train', mnist_train), ('test', mnist_test)]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for img, lbl in loader:
            _add_sample({k: data[k][split_name] for k in data.keys()},
                        img.squeeze(0), lbl.item())

    # --------------------------------------------------------------------- #
    # 4.  Tensor-ify and (optional) validation split
    # --------------------------------------------------------------------- #
    for t in data.keys():
        for split in ['train', 'test']:
            xs = torch.stack(data[t][split]['x']).view(-1, *size)
            ys = torch.LongTensor(np.array(data[t][split]['y'], dtype=int))
            data[t][split]['x'], data[t][split]['y'] = xs, ys

        # create a validation set (clone or true split)
        if pc_valid == 0:
            data[t]['valid'] = {
                'x': data[t]['train']['x'].clone(),
                'y': data[t]['train']['y'].clone()
            }
        else:
            n_train = data[t]['train']['y'].size(0)
            n_val = int(np.floor(pc_valid * n_train))
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed))
            val_idx, train_idx = perm[:n_val], perm[n_val:]

            data[t]['valid'] = {
                'x': data[t]['train']['x'][val_idx].clone(),
                'y': data[t]['train']['y'][val_idx].clone()
            }
            data[t]['train']['x'] = data[t]['train']['x'][train_idx].clone()
            data[t]['train']['y'] = data[t]['train']['y'][train_idx].clone()

        taskcla.append((t, data[t]['ncla']))

    # --------------------------------------------------------------------- #
    # 5.  Meta info
    # --------------------------------------------------------------------- #
    data['ncla'] = sum([ncla for _, ncla in taskcla])
    return data, taskcla, size
