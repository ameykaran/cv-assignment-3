import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


PLY_TO_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def load_ply(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    end_marker = b"end_header\n"
    header_end = data.find(end_marker)
    marker_len = len(end_marker)

    if header_end == -1:
        end_marker = b"end_header\r\n"
        header_end = data.find(end_marker)
        marker_len = len(end_marker)

    if header_end == -1:
        raise ValueError(f"Invalid PLY file (missing end_header): {file_path}")

    header_bytes = data[:header_end]
    payload = data[header_end + marker_len :]
    header_lines = header_bytes.decode("ascii", errors="replace").splitlines()

    fmt = None
    vertex_count = None
    vertex_properties = []
    in_vertex_element = False

    for line in header_lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex_element = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex_element:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex_element:
            # Support scalar properties in the vertex element (x, y, z, ...)
            if len(parts) == 3:
                vertex_properties.append((parts[1], parts[2]))

    if fmt is None or vertex_count is None or len(vertex_properties) < 3:
        raise ValueError(f"Unsupported or malformed PLY header: {file_path}")

    x_idx = next((i for i, (_, name) in enumerate(vertex_properties) if name == "x"), None)
    y_idx = next((i for i, (_, name) in enumerate(vertex_properties) if name == "y"), None)
    z_idx = next((i for i, (_, name) in enumerate(vertex_properties) if name == "z"), None)
    if x_idx is None or y_idx is None or z_idx is None:
        raise ValueError(f"PLY vertex properties must include x, y, z: {file_path}")

    if fmt == "ascii":
        text = payload.decode("utf-8", errors="replace").splitlines()
        points = []
        for line in text[:vertex_count]:
            vals = line.strip().split()
            if len(vals) <= max(x_idx, y_idx, z_idx):
                continue
            points.append([float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])])
        points = np.asarray(points, dtype=np.float32)
    elif fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype_fields = []
        for ply_type, name in vertex_properties:
            if ply_type not in PLY_TO_NUMPY:
                raise ValueError(f"Unsupported PLY property type '{ply_type}' in {file_path}")
            dtype_fields.append((name, endian + PLY_TO_NUMPY[ply_type]))

        vertex_dtype = np.dtype(dtype_fields)
        expected_bytes = vertex_count * vertex_dtype.itemsize
        if len(payload) < expected_bytes:
            raise ValueError(
                f"PLY payload is too short for {vertex_count} vertices in {file_path}"
            )

        verts = np.frombuffer(payload[:expected_bytes], dtype=vertex_dtype, count=vertex_count)
        points = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PLY format '{fmt}' in {file_path}")

    # print(f"Loaded {points.shape[0]} points from {file_path}")
    return points


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, n_points=2048):
        self.root_dir = root_dir
        self.files = []
        self.labels = []
        self.n_points = n_points

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith(".ply"):
                    self.files.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.files)


    def normalize(self, points):
        # Center
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # Scale
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points

    def sample_points(self, points):
        N = points.shape[0]

        if N >= self.n_points:
            idx = np.random.choice(N, self.n_points, replace=False)
        else:
            idx = np.random.choice(N, self.n_points, replace=True)

        return points[idx]

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        points = load_ply(file_path)

        points = self.sample_points(points)
        points = self.normalize(points)

        return torch.tensor(points, dtype=torch.float32), label
    
    def get_each_class_sample(self, n_samples_per_class=1):
        class_samples = {}
        for file_path, label in zip(self.files, self.labels):
            if label not in class_samples:
                class_samples[label] = []
            if len(class_samples[label]) < n_samples_per_class:
                class_samples[label].append(file_path)

        sample_points = []
        sample_labels = []
        for label, files in class_samples.items():
            for file in files:
                points = load_ply(file)
                points = self.sample_points(points)
                points = self.normalize(points)
                sample_points.append(points)
                sample_labels.append(label)

        return torch.tensor(sample_points, dtype=torch.float32), torch.tensor(sample_labels, dtype=torch.long)

def visualize_interactive(points, critical_points=None):
    """
    points: (N, 3) torch tensor or numpy array
    critical_points: optional (M, 3)
    """

    if hasattr(points, "numpy"):
        points = points.numpy()

    # Full point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color: light gray
    colors = np.tile([0.7, 0.7, 0.7], (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # If critical points exist → highlight
    if critical_points is not None:
        if hasattr(critical_points, "numpy"):
            critical_points = critical_points.numpy()

        critical_pcd = o3d.geometry.PointCloud()
        critical_pcd.points = o3d.utility.Vector3dVector(critical_points)

        # Red color
        red = np.tile([1, 0, 0], (critical_points.shape[0], 1))
        critical_pcd.colors = o3d.utility.Vector3dVector(red)

        geometries.append(critical_pcd)

    # Launch interactive viewer
    o3d.visualization.draw_geometries(geometries)



def get_dataloaders(num_points=2048):
    full_dataset = PointCloudDataset(root_dir="/scratch/ameyk/cv/q3/data/train", n_points=num_points)
    test_dataset = PointCloudDataset(root_dir="/scratch/ameyk/cv/q3/data/test", n_points=num_points)

    val_size = int(0.25 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    from torch.utils.data import random_split, DataLoader
    
    generator = torch.Generator().manual_seed(42)  # reproducible split
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    dataset = PointCloudDataset("/scratch/ameyk/cv/q3/data/train")
    sample_points, sample_label = dataset[800]

    print(sample_points.shape)
    print(sample_label)

    visualize_interactive(sample_points)