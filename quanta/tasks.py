import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

from quanta.utils import bipolar_parity_labels_for_indices, bipolar_random_bits



class Task(Dataset, ABC):
    """
    Abstract Base Class for Task Datasets.
    Designed to work with both synthetic matrices and natural language/vision datasets.
    """
    def __init__(self, task_id: str, device='cpu', dtype=torch.float32):
        self.id = task_id
        self.device = device
        self.dtype = dtype
        self.x = None
        self.y = None

    def __len__(self):
        if self.x is None:
            return 0
        return len(self.x)

    def __getitem__(self, idx):
        if self.x is None or self.y is None:
            raise ValueError("Data has not been populated yet.")
        return self.x[idx], self.y[idx]

    @abstractmethod
    def generate_data(self, **kwargs):
        """
        Populates self.x and self.y based on task-specific parameters.
        """
        pass

    def get_dataloader(self, batch_size, shuffle=True, **kwargs):
        """
        Returns a generic PyTorch DataLoader for streaming inputs.
        """
        if self.x is None or self.y is None:
            raise ValueError("Dataset is empty. Call generate_data() first.")
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)


class CXOR(Task):
    """
    Graph Multitask Sparse Parity (CXOR) Task with downward-closed constraints.
    """
    def __init__(self, task_id: str, n_tasks: int, n: int, device='cpu', dtype=torch.float32):
        super().__init__(task_id=task_id, device=device, dtype=dtype)
        self.n_tasks = n_tasks
        self.n = n

    @staticmethod
    def _get_ancestral_closure(node, deps):
        """Returns the set of all ancestors (inclusive) of `node` in the dependency graph."""
        closure = {node}
        if node in deps:
            for parent in deps[node]:
                closure.update(CXOR._get_ancestral_closure(parent, deps))
        return closure

    @staticmethod
    def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float32,
                  graph_dependencies=None):
        """
        Fast static batch generator for CXOR.

        Args:
            n_tasks:            Total number of task/control bits.
            n:                  Total number of parity/feature bits.
            Ss:                 List mapping task index -> list of parity bit positions.
            codes:              List of target node indices (one per subtask).
            sizes:              Number of samples per subtask (parallel to codes).
            device:             Torch device.
            dtype:              Dtype for x tensor.
            graph_dependencies: Dict mapping node -> list of parent nodes.
                                If None, each node is treated as independent (no ancestry).
        """
        if graph_dependencies is None:
            graph_dependencies = {}

        total = sum(sizes)
        x = torch.zeros((total, n_tasks + n), dtype=dtype, device=device)
        bits = bipolar_random_bits((total, n), dtype=dtype, device=device)
        x[:, n_tasks:] = bits

        y = torch.empty((total,), dtype=torch.int64, device=device)

        idx = 0
        for target, size in zip(codes, sizes):
            if size <= 0:
                continue
            active_quanta = CXOR._get_ancestral_closure(target, graph_dependencies)
            combined_indices = sorted(set(
                bit for q in active_quanta for bit in Ss[q]
            ))
            x[idx:idx+size, list(active_quanta)] = 1
            y[idx:idx+size] = bipolar_parity_labels_for_indices(bits[idx:idx+size], combined_indices)
            idx += size

        return x, y

    def generate_data(self, Ss, graph_dependencies, target_nodes, sizes):
        def get_ancestral_closure(node, deps):
            closure = {node}
            if node in deps:
                for parent in deps[node]:
                    closure.update(get_ancestral_closure(parent, deps))
            return closure

        total_samples = sum(sizes)
        batch_x = torch.zeros((total_samples, self.n_tasks + self.n), dtype=self.dtype, device=self.device)
        batch_y = torch.zeros((total_samples,), dtype=torch.int64, device=self.device)

        start_i = 0
        for target, size in zip(target_nodes, sizes):
            if size > 0:
                active_quanta = get_ancestral_closure(target, graph_dependencies)
                x_bits = bipolar_random_bits((size, self.n), dtype=self.dtype, device=self.device)

                combined_indices = []
                for q in active_quanta:
                    combined_indices.extend(Ss[q])

                y = bipolar_parity_labels_for_indices(x_bits, combined_indices)

                x_task_code = torch.zeros((size, self.n_tasks), dtype=self.dtype, device=self.device)
                x_task_code[:, list(active_quanta)] = 1

                x = torch.cat([x_task_code, x_bits], dim=1)
                batch_x[start_i:start_i+size, :] = x
                batch_y[start_i:start_i+size] = y
                start_i += size

        self.x = batch_x
        self.y = batch_y
