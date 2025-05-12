import torch
import numpy as np
from functools import partial


def get_size(data):
    """Return the size of the dataset."""
    sizes = {k: len(v) for k, v in data.items()}
    return max(sizes.values())


def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if isinstance(crop_from, np.ndarray):
        crop_from = torch.from_numpy(crop_from)
    
    padded_img = torch.nn.functional.pad(img, (0, 0, padding, padding, padding, padding), mode='replicate')
    
    # Crop from the padded image
    h, w, c = img.shape
    x, y, _ = crop_from
    cropped = padded_img[y:y+h, x:x+w, :]
    
    return cropped


def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)
    if isinstance(crop_froms, np.ndarray):
        crop_froms = torch.from_numpy(crop_froms)
    
    results = []
    for img, crop_from in zip(imgs, crop_froms):
        results.append(random_crop(img, crop_from, padding))
    
    return torch.stack(results)


class Dataset:
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays (ignored in PyTorch version).
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        return cls(data)

    def __init__(self, data_dict):
        self._dict = data_dict
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self._dict['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def __getitem__(self, key):
        return self._dict[key]

    def items(self):
        return self._dict.items()

    def copy(self, add_or_replace=None):
        """Create a copy of the dataset with optional additions or replacements."""
        new_dict = {k: v.copy() for k, v in self._dict.items()}
        if add_or_replace:
            new_dict.update(add_or_replace)
        return Dataset(new_dict)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append({k: v[cur_idxs] for k, v in self._dict.items() if k == 'observations'})
                if i != self.frame_stack - 1:
                    next_obs.append({k: v[cur_idxs] for k, v in self._dict.items() if k == 'observations'})
            next_obs.append({k: v[idxs] for k, v in self._dict.items() if k == 'next_observations'})

            # Concatenate along the last dimension
            if obs:
                obs_stack = np.concatenate([o['observations'] for o in obs], axis=-1)
                batch['observations'] = obs_stack
            
            if next_obs:
                next_items = []
                for o in next_obs:
                    key = 'observations' if 'observations' in o else 'next_observations'
                    next_items.append(o[key])
                if next_items:
                    next_obs_stack = np.concatenate(next_items, axis=-1)
                    batch['next_observations'] = next_obs_stack
            
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
                
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).float()
        
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = {k: v[idxs] for k, v in self._dict.items()}
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            if len(batch[key].shape) == 4:  # Only apply to images (B, H, W, C)
                batch[key] = batched_random_crop(batch[key], crop_froms, padding)


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape[1:]), dtype=example.dtype)

        buffer_dict = {k: create_buffer(v) for k, v in transition.items()}
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """
        buffer_dict = {}
        for k, v in init_dataset.items():
            buffer = np.zeros((size, *v.shape[1:]), dtype=v.dtype)
            buffer[:len(v)] = v
            buffer_dict[k] = buffer
        
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""
        for k, v in transition.items():
            self._dict[k][self.pointer] = v
            
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
