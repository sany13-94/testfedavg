import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from wilds import get_dataset

# 1) wrapper to drop metadata (so __getitem__ returns (x, y))
class DropMetadata(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, _md = self.base_ds[idx]
        return x, y


def _infer_center_field(subset, candidates=("hospital", "center", "site")):
    fields = subset.dataset.metadata_fields
    for c in candidates:
        if c in fields:
            return c, fields.index(c)
    raise ValueError(f"Could not find center field. metadata_fields={fields}")


def _subset_center_ids(subset):
    name, j = _infer_center_field(subset)
    md = subset.dataset.metadata_array[subset.indices]  # aligned with subset.indices
    centers = md[:, j].astype(int)
    return name, centers


def _split_list(idxs, n_splits, rng):
    idxs = idxs.copy()
    rng.shuffle(idxs)
    chunks = np.array_split(np.array(idxs), n_splits)
    return [c.tolist() for c in chunks if len(c) > 0]


def make_camelyon17_clients_with_domains(
    num_clients: int,
    batch_size: int,
    root_dir: str = "./data",
    domain_shift: bool = True,
    heldout_center: int | None = None,
    split_each_center_into: int | None = None,
    seed: int = 42,
    num_workers: int = 4,
):
    """
    Returns:
      trainloaders: List[DataLoader] length = num_clients
      valloaders:   List[DataLoader] length = num_clients
      domain_assignment: List[int] length = num_clients, giving "center id" per client

    Behavior:
      - If domain_shift=True: last client (id = num_clients-1) becomes the held-out center (OOD).
        Other clients train on remaining centers.
      - If split_each_center_into is set: each center is split into that many clients (until you reach target).
        Otherwise: one client per center (or as close as possible).
    """
    rng = np.random.default_rng(seed)

    # Transforms (adjust normalization to your backbone if needed)
    transform = T.Compose([T.ToTensor()])

    dataset = get_dataset("camelyon17", root_dir=root_dir, download=True)
    train_subset = dataset.get_subset("train", transform=transform)
    val_subset   = dataset.get_subset("val",   transform=transform)
    test_subset  = dataset.get_subset("test",  transform=transform)

    # Extract center ids for each subset
    center_field, train_centers = _subset_center_ids(train_subset)
    _, val_centers  = _subset_center_ids(val_subset)
    _, test_centers = _subset_center_ids(test_subset)

    unique_centers = sorted(np.unique(train_centers).tolist())

    # Decide held-out center
    if domain_shift:
        if heldout_center is None:
            # common/simple default: last center id from train metadata universe
            heldout_center = unique_centers[-1]
        if heldout_center not in unique_centers:
            raise ValueError(f"heldout_center={heldout_center} not in train centers {unique_centers}")

    # Build index lists per center for train/val/test
    train_idx_by_center = {c: [] for c in unique_centers}
    for local_i, c in enumerate(train_centers):
        train_idx_by_center[int(c)].append(train_subset.indices[local_i])

    val_idx_by_center = {c: [] for c in sorted(np.unique(val_centers).tolist())}
    for local_i, c in enumerate(val_centers):
        val_idx_by_center.setdefault(int(c), []).append(val_subset.indices[local_i])

    test_idx_by_center = {c: [] for c in sorted(np.unique(test_centers).tolist())}
    for local_i, c in enumerate(test_centers):
        test_idx_by_center.setdefault(int(c), []).append(test_subset.indices[local_i])

    # ---------- Build client partitions (center-pure) ----------
    train_client_indices = []
    val_client_indices = []
    domain_assignment = []

    if domain_shift:
        train_client_target = num_clients - 1
        train_centers_used = [c for c in unique_centers if c != heldout_center]
    else:
        train_client_target = num_clients
        train_centers_used = unique_centers

    # Determine how many clients per center
    # Option 1: explicit split_each_center_into
    if split_each_center_into is not None:
        per_center = split_each_center_into
        buckets = []
        for c in train_centers_used:
            chunks = _split_list(train_idx_by_center[c], per_center, rng)
            for chunk in chunks:
                buckets.append((c, chunk))
        # If too many buckets, truncate; if too few, we'll further split largest ones
        buckets.sort(key=lambda x: len(x[1]), reverse=True)

        while len(buckets) < train_client_target:
            # split largest bucket
            c, idxs = buckets.pop(0)
            if len(idxs) < 2:
                buckets.append((c, idxs))
                break
            mid = len(idxs) // 2
            buckets.append((c, idxs[:mid]))
            buckets.append((c, idxs[mid:]))
            buckets.sort(key=lambda x: len(x[1]), reverse=True)

        buckets = buckets[:train_client_target]

    # Option 2: auto: try to spread centers across clients (splitting large centers if needed)
    else:
        # start one bucket per center
        buckets = [(c, train_idx_by_center[c]) for c in train_centers_used]
        for i in range(len(buckets)):
            c, idxs = buckets[i]
            idxs = idxs.copy()
            rng.shuffle(idxs)
            buckets[i] = (c, idxs)

        if train_client_target < len(buckets):
            raise ValueError(
                f"num_clients too small to keep clients center-pure: "
                f"need >= {len(buckets) + (1 if domain_shift else 0)} total clients."
            )

        # split largest buckets until reach target
        buckets.sort(key=lambda x: len(x[1]), reverse=True)
        while len(buckets) < train_client_target:
            c, idxs = buckets.pop(0)
            if len(idxs) < 2:
                buckets.append((c, idxs))
                break
            mid = len(idxs) // 2
            buckets.append((c, idxs[:mid]))
            buckets.append((c, idxs[mid:]))
            buckets.sort(key=lambda x: len(x[1]), reverse=True)

        buckets = buckets[:train_client_target]

    # Fill train clients
    for (c, idxs) in buckets:
        train_client_indices.append(idxs)
        # validation indices for that same center (if present), else empty
        val_client_indices.append(val_idx_by_center.get(c, []))
        domain_assignment.append(int(c))

    # Held-out / OOD client is the last one (if domain_shift)
    if domain_shift:
        ood_train = []  # usually no training data for held-out domain
        # choose whether OOD eval comes from val or test
        ood_val = val_idx_by_center.get(heldout_center, [])
        if len(ood_val) == 0:
            ood_val = test_idx_by_center.get(heldout_center, [])
        train_client_indices.append(ood_train)
        val_client_indices.append(ood_val)
        domain_assignment.append(int(heldout_center))

    # ---------- Build DataLoaders ----------
    base_train = train_subset.dataset
    base_val   = val_subset.dataset

    trainloaders = []
    valloaders = []

    for idxs_train, idxs_val in zip(train_client_indices, val_client_indices):
        train_ds = DropMetadata(Subset(base_train, idxs_train)) if len(idxs_train) > 0 else DropMetadata([])
        val_ds   = DropMetadata(Subset(base_val, idxs_val))     if len(idxs_val) > 0 else DropMetadata([])

        trainloaders.append(
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True)
        )
        valloaders.append(
            DataLoader(val_ds, batch_size=max(batch_size, 2*batch_size), shuffle=False,
                       num_workers=num_workers, pin_memory=True)
        )

    print(f"[Camelyon17/WILDS] center_field='{center_field}' centers={unique_centers}")
    print(f"[Camelyon17/WILDS] num_clients={num_clients} domain_shift={domain_shift} heldout_center={heldout_center}")
    print(f"[Camelyon17/WILDS] domain_assignment={domain_assignment}")

    return trainloaders, valloaders, domain_assignment