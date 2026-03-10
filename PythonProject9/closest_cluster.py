import numpy as np
from collections import deque

class GridClusterCOM:
    def __init__(self, com_radius, reduction_threshold=None, frame_threshold=3):
        """
        com_radius: maximum distance from centroid to any point in the cluster.
        reduction_threshold: if set, each cluster stores at most this many
                             raw points (oldest are discarded). Centroid is
                             computed from the stored points.
        frame_threshold: number of frames (including current) to look back
                         when searching for existing clusters.
        """
        self.com_radius = com_radius
        self.frame_threshold = frame_threshold
        self.cell_size = com_radius / np.sqrt(2)   # so diagonal = com_radius
        self.reduction_threshold = reduction_threshold
        self.clusters = []          # list of cluster dicts
        self.cell_to_cluster = {}   # (x, y, frame) -> list of cluster indices

    def _cell_key(self, point, frame):
        """Return the grid cell key (x, y, frame) for a point."""
        cell_xy = tuple(np.floor(point / self.cell_size).astype(int))
        return (cell_xy[0], cell_xy[1], frame)

    def _temporal_neighbor_keys(self, key):
        """
        Generate all cell keys that should be considered for cluster matching.
        Includes the 3x3 spatial neighborhood for the current frame and
        for up to (frame_threshold-1) previous frames.
        """
        x, y, frame = key
        min_frame = max(0, frame - (self.frame_threshold - 1))  # avoid negative frames
        keys = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for t in range(min_frame, frame + 1):
                    candidate_key = (x + dx, y + dy, t)
                    if candidate_key not in self.cell_to_cluster:
                        continue
                    keys.append(candidate_key)
        return keys

    def nearest_neighbor_id(self, point, key):
        """
        Find the closest existing cluster within com_radius,
        considering clusters from the current and previous frames
        (up to frame_threshold frames back).
        Returns the cluster index (in self.clusters) or None.
        """
        best_idx = None
        best_dist = float('inf')

        # Check all candidate cells (spatial + temporal)
        for candidate_key in self._temporal_neighbor_keys(key):
            # if candidate_key not in self.cell_to_cluster:
            #     continue
            for cluster_idx in self.cell_to_cluster[candidate_key]:
                leader = np.array(self.clusters[cluster_idx]['leader'])
                dist = np.linalg.norm(point - leader)
                if dist <= self.com_radius and dist < best_dist:
                    best_dist = dist
                    best_idx = cluster_idx
        return best_idx

    def add_point(self, point, frame):
        point = np.asarray(point)
        point_tuple = tuple(float(x) for x in point)
        key = self._cell_key(point, frame)

        # Try to assign to an existing cluster
        cluster_idx = self.nearest_neighbor_id(point, key)
        if cluster_idx is None:
            # Create a new cluster
            new_idx = len(self.clusters)
            new_id = new_idx + 1

            if self.reduction_threshold is not None:
                points_buffer = deque(maxlen=self.reduction_threshold)
            else:
                points_buffer = []          # unbounded list
            points_buffer.append(point_tuple)

            # For incremental update in unbounded case, store sum
            cluster = {
                'id': new_id,
                'leader': point_tuple,
                'points': points_buffer,
                'count': 1,
                'last_frame': frame,
                'sum': np.array(point, dtype=float) if self.reduction_threshold is None else None
            }
            self.clusters.append(cluster)

            # Map this cell to the new cluster
            self.cell_to_cluster.setdefault(key, []).append(new_idx)
            return cluster['leader'], cluster['id']

        # Add point to existing cluster
        cluster = self.clusters[cluster_idx]

        # Update buffer
        cluster['points'].append(point_tuple)
        cluster['last_frame'] = frame

        # Update centroid
        if self.reduction_threshold is None:
            # Unbounded buffer → incremental update
            old_sum = cluster['sum']
            new_sum = old_sum + point
            new_count = cluster['count'] + 1
            new_leader = new_sum / new_count
            cluster['leader'] = tuple(float(x) for x in new_leader)
            cluster['sum'] = new_sum
            cluster['count'] = new_count
        else:
            # Bounded buffer → recompute from stored points (O(maxlen))
            total = np.zeros_like(point, dtype=float)   # float accumulator to avoid casting issues
            for pt in cluster['points']:
                total += np.asarray(pt)
            new_leader = total / len(cluster['points'])
            cluster['leader'] = tuple(float(x) for x in new_leader)
            cluster['count'] = len(cluster['points'])

        # Ensure this cell knows about the cluster (for future points)
        if cluster_idx not in self.cell_to_cluster.setdefault(key, []):
            self.cell_to_cluster[key].append(cluster_idx)

        return cluster['leader'], cluster['id']

    def get_all_centroids(self):
        return [cls['leader'] for cls in self.clusters]

    def get_cluster_points(self, cluster_id):
        for cls in self.clusters:
            if cls['id'] == cluster_id:
                return cls['points']
        return None

    def clear_old_clusters(self, current_frame):
        """
        Remove clusters that have not received any point for more than
        frame_threshold frames.
        """
        stale_indices = []
        for i, cls in enumerate(self.clusters):
            if current_frame - cls['last_frame'] > self.frame_threshold:
                stale_indices.append(i)

        # Remove from clusters (reverse order to keep indices valid)
        for i in reversed(stale_indices):
            # Remove all references from cell_to_cluster
            for cell, indices in list(self.cell_to_cluster.items()):
                if i in indices:
                    indices.remove(i)
                    if not indices:          # empty list → delete cell key
                        del self.cell_to_cluster[cell]
            # Delete the cluster itself
            del self.clusters[i]



# clusterer = GridClusterCOM(com_radius=5, reduction_threshold=3, frame_threshold=3)
#
# points = [(1,1), (1,2), (1,3), (10,10), (10,11), (1,4), (1,5),(1,6), (1,7),(1,8), (1,9),(1,10),(1,11)]
# for i, pt in enumerate(points):
#     frame = i // 3   # change frame every 3 points
#     # frame = 1
#     centroid, cid = clusterer.add_point(pt, frame)
#     print(f"Frame {frame}, point {pt} -> cluster {cid}, centroid {centroid}")
#
# print("\nCluster details:")
# for cls in clusterer.clusters:
#     print(f"ID {cls['id']}: leader={cls['leader']}, points={list(cls['points'])}")