"""Mesh refinement & quality statistics.

All numpy/scipy only. Refinement is a lightweight offline pass: edge-aware
weighted Laplacian smoothing (no smoothing across open boundary edges), removal
of degenerate / flipped faces, non-manifold statistics, and mesh quality
metrics (hole ratio, non-manifold edges, self-intersections, components).
"""

import numpy as np
from scipy.sparse import coo_matrix, csgraph


def face_normals(verts, faces):
    v = verts[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return n / norm


def face_areas(verts, faces):
    v = verts[faces]
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def _edge_incidence(faces):
    """Return ``(counts, rows, cols)`` for all edges of ``faces``.

    ``counts[i]`` is how many faces reference unique edge ``i`` (1 -> boundary,
    >2 -> non-manifold).  ``(rows, cols)`` are directed face-adjacency pairs:
    each shared edge contributes a path through its incident faces, so the
    undirected graph over those pairs has the same connected components as the
    face-adjacency graph.  Edges are packed into ``(min<<32)|max`` int64 keys
    and sorted once — no per-edge Python dict — so peak memory is O(n) instead
    of the old O(n)-entry dict (which OOM'd on 50M+-face meshes).
    """
    if faces.size == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64))
    n = faces.shape[0]
    keys = np.empty(3 * n, dtype=np.int64)
    for k in range(3):
        a = faces[:, k].astype(np.int64)
        b = faces[:, (k + 1) % 3].astype(np.int64)
        keys[k::3] = (np.minimum(a, b) << 32) | np.maximum(a, b)
        del a, b
    # keys are laid out interleaved: position p holds edge (p % 3) of face (p // 3).
    fid = np.repeat(np.arange(n, dtype=np.int64), 3)
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    fs = fid[order]
    del keys, fid, order
    uniq, start = np.unique(ks, return_index=True)
    counts = np.diff(np.append(start, ks.size))
    same = ks[1:] == ks[:-1]
    rows = fs[:-1][same]
    cols = fs[1:][same]
    del ks, fs, same, uniq, start
    return counts, rows, cols


def boundary_edges(faces):
    """Return the (E,2) edges referenced by exactly one face (open boundary).

    Memory-efficient: edges are packed into a single ``(3n,)`` int64 buffer of
    ``(min<<32)|max`` keys (one column at a time) instead of stacking three full
    ``(n,2)`` copies — the old ``np.vstack`` materialised ~3 copies + a stacked
    array (~7x the face buffer) and OOM'd the host for the anisotropic kernel's
    146M-face pre-clean mesh.
    """
    faces = np.asarray(faces)
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    n = faces.shape[0]
    keys = np.empty(3 * n, dtype=np.int64)
    for k in range(3):
        a = faces[:, k].astype(np.int64)
        b = faces[:, (k + 1) % 3].astype(np.int64)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keys[k::3] = (lo << 32) | hi
        del a, b, lo, hi
    uniq, counts = np.unique(keys, return_counts=True)
    e = uniq[counts == 1]
    lo = (e >> 32).astype(np.int32)
    hi = (e & 0xFFFFFFFF).astype(np.int32)
    return np.stack([lo, hi], axis=1)


def mesh_stats(mesh):
    verts = np.asarray(mesh["verts"])
    faces = np.asarray(mesh["faces"])
    n_v, n_f = verts.shape[0], faces.shape[0]
    stats = {"n_vertices": int(n_v), "n_faces": int(n_f)}

    if n_f == 0:
        stats.update({"n_boundary_edges": 0, "hole_ratio": 0.0,
                      "n_nonmanifold_edges": 0, "n_degenerate_faces": 0,
                      "n_components": 0, "n_self_intersections": 0})
        return stats

    counts, rows, cols = _edge_incidence(faces)
    stats["n_boundary_edges"] = int(np.sum(counts == 1))
    stats["hole_ratio"] = float(np.sum(counts == 1) / counts.size)
    stats["n_nonmanifold_edges"] = int(np.sum(counts > 2))
    areas = face_areas(verts, faces)
    stats["n_degenerate_faces"] = int(np.sum(areas < 1e-9))

    # Connected components (face adjacency via shared edges).
    if rows.size:
        adj = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_f, n_f)).tocsr()
        n_comp = csgraph.connected_components(adj, directed=False)[0]
    else:
        n_comp = 0
    stats["n_components"] = int(n_comp)

    stats["n_self_intersections"] = int(self_intersection_count(verts, faces))
    return stats


def cleanup(verts, faces, normals, orient_thresh=0.0):
    """Remove degenerate + flipped faces.

    Orientation is judged against *face-derived* vertex normals (robust to the
    noisy alpha-weighted per-vertex normals), so a consistently-wound
    single-view depth mesh is not over-pruned.

    ``orient_thresh`` is the cosine-agreement cutoff: a face is dropped when
    ``face_normal . mean(vertex_normals) < orient_thresh``. The default 0.0
    drops any face wound against its local average, which is fine for a
    coherent surface but *over-prunes* self-overlapping meshes (many sheets ->
    vertex normals cancel -> legitimate faces look flipped). For such meshes a
    lenient threshold (e.g. -0.5) removes only clearly-flipped faces.
    """
    faces = np.asarray(faces)
    if faces.size == 0:
        return verts, normals, faces
    areas = face_areas(verts, faces)
    faces = faces[areas >= 1e-9]
    if faces.size == 0:
        return verts, normals, faces
    fn = face_normals(verts, faces)
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    vn /= n
    vn_face = vn[faces].mean(axis=1)
    vn_face /= np.linalg.norm(vn_face, axis=1, keepdims=True).clip(1e-9)
    keep_orient = (fn * vn_face).sum(axis=1) > orient_thresh
    return verts, normals, faces[keep_orient]


def laplacian_smooth(verts, faces, boundary_edges=None, lam=0.3, iters=3):
    """Edge-aware uniform Laplacian smoothing (no smoothing across boundary edges)."""
    verts = np.asarray(verts, dtype=np.float64).copy()
    faces = np.asarray(faces)
    if faces.size == 0 or iters <= 0:
        return verts.astype(np.float32)
    n = verts.shape[0]
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    edges = np.sort(np.vstack([e0, e1, e2]), axis=1)
    bset = set()
    if boundary_edges is not None and boundary_edges.size:
        for a, b in np.asarray(boundary_edges):
            bset.add((int(min(a, b)), int(max(a, b))))
    # Build symmetric adjacency, dropping boundary edges.
    valid = np.array([(int(a), int(b)) not in bset for a, b in edges])
    edges = edges[valid]
    if edges.size == 0:
        return verts.astype(np.float32)
    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones_like(i, dtype=np.float64)
    W = coo_matrix((data, (i, j)), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg < 1e-12] = 1.0
    # Uniform Laplacian: L x = (W x / deg) - x
    for _ in range(iters):
        lap = (W @ verts) / deg[:, None] - verts
        verts = verts + lam * lap
    return verts.astype(np.float32)


def edge_weighted_laplacian(verts, faces, boundary_edges=None, edge=None,
                            gamma=5.0, lam=0.3, iters=3, vert_confidence=None):
    """Edge-aware weighted Laplacian smoothing.

    Like ``laplacian_smooth`` but the adjacency is weighted by the half-Gaussian
    boundary response so smoothing is suppressed *across* a high-E_v edge (a
    trustworthy geometric fold line / open boundary) while flat regions — and
    the fold line itself — still relax. Open boundary edges are never smoothed
    across.

    The weight is edge-crossing aware: ``w_ij = exp(-gamma * |E_i - E_j|)``.
    Using ``max(E_i, E_j)`` instead would suppress *every* edge incident on a
    high-E_v vertex, which cancels in the normalised Laplacian and leaves that
    vertex moving like the uniform case; ``|E_i - E_j|`` suppresses only the
    edges that cross the discontinuity and keeps along-fold smoothing intact.

    Args:
        verts: (N,3) float.
        faces: (F,3) int.
        boundary_edges: (E,2) int open-boundary edges to preserve (optional).
        edge: (N,) float per-vertex E_v response (uniform weights if None).
        gamma: edge falloff strength.
        lam / iters: step size and iteration count.
        vert_confidence: (N,) float anchor strength in [0,1]; if given, verts
            are pulled back toward their original position with this weight
            (high confidence -> anchored -> moves less).

    Returns:
        (N,3) float32 smoothed positions.
    """
    verts = np.asarray(verts, dtype=np.float64).copy()
    orig = verts.copy()
    faces = np.asarray(faces)
    if faces.size == 0 or iters <= 0:
        return verts.astype(np.float32)
    n = verts.shape[0]
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]
    edges = np.sort(np.vstack([e0, e1, e2]), axis=1)
    bset = set()
    if boundary_edges is not None and boundary_edges.size:
        for a, b in np.asarray(boundary_edges):
            bset.add((int(min(a, b)), int(max(a, b))))
    valid = np.array([(int(a), int(b)) not in bset for a, b in edges])
    edges = edges[valid]
    if edges.size == 0:
        return verts.astype(np.float32)

    if edge is not None:
        e = np.asarray(edge, dtype=np.float64)
        w = np.exp(-gamma * np.abs(e[edges[:, 0]] - e[edges[:, 1]]))
    else:
        w = np.ones(edges.shape[0], dtype=np.float64)

    i = np.concatenate([edges[:, 0], edges[:, 1]])
    j = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([w, w])
    W = coo_matrix((data, (i, j)), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg < 1e-12] = 1.0

    anchor = None
    if vert_confidence is not None:
        anchor = np.asarray(vert_confidence, dtype=np.float64).reshape(-1, 1)
        anchor = np.clip(anchor, 0.0, 1.0)

    for _ in range(iters):
        lap = (W @ verts) / deg[:, None] - verts
        verts = verts + lam * lap
        if anchor is not None:
            verts = verts * (1.0 - anchor) + orig * anchor
    return verts.astype(np.float32)


def remove_small_components(mesh, min_faces=50, silhouette=None, silhouette_thresh=0.2):
    """Drop connected face components with fewer than ``min_faces`` faces.

    Such components are typically spurious slivers.
    A component is retained despite being small if its mean per-vertex
    ``silhouette`` support is >= ``silhouette_thresh`` (a real, thin surface
    patch whose boundary aligns with geometric edges).

    Returns a new mesh dict with remapped faces/verts and a recomputed
    ``boundary_edges``; other per-vertex arrays are remapped in place.
    """
    verts = np.asarray(mesh["verts"])
    faces = np.asarray(mesh["faces"])
    n_v = verts.shape[0]
    if faces.size == 0:
        return mesh

    _, rows, cols = _edge_incidence(faces)
    n_f = faces.shape[0]
    if rows.size == 0:
        return mesh
    adj = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_f, n_f)).tocsr()
    n_comp, labels = csgraph.connected_components(adj, directed=False)

    sil = None
    if silhouette is not None:
        sil = np.asarray(silhouette, dtype=np.float64)

    keep_faces = np.ones(n_f, dtype=bool)
    for c in range(n_comp):
        mask = labels == c
        if int(mask.sum()) >= min_faces:
            continue
        keep = False
        if sil is not None:
            fids = np.where(mask)[0]
            vids = np.unique(faces[fids].ravel())
            if vids.size and float(sil[vids].mean()) >= silhouette_thresh:
                keep = True
        if not keep:
            keep_faces[mask] = False

    kept_idx = np.where(keep_faces)[0]
    if kept_idx.size == n_f:
        return mesh

    kept_faces = faces[kept_idx]
    kept_verts = np.unique(kept_faces.ravel())
    remap = np.full(n_v, -1, dtype=np.int64)
    remap[kept_verts] = np.arange(kept_verts.size)

    new = {k: v for k, v in mesh.items()}
    new["faces"] = remap[kept_faces].astype(np.int32)
    for k in list(new.keys()):
        if k in ("faces", "boundary_edges"):
            continue
        arr = np.asarray(new[k])
        if arr.ndim >= 1 and arr.shape[0] == n_v:
            new[k] = arr[kept_verts]
    new["boundary_edges"] = boundary_edges(new["faces"])
    return new


def self_intersection_count(verts, faces, bin_size=None, max_pairs=500000):
    """Approximate self-intersection count via spatial hashing + Möller–Trumbore.

    Triangles are binned by their bounding-box centroid; candidate pairs within
    a bin are tested for a proper edge-triangle intersection (not sharing a
    vertex). When the candidate pair count exceeds ``max_pairs``, a
    deterministic head-of-stream subsample is tested and scaled back up — the
    metric only flags gross overlap, so an estimate suffices for large meshes.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces)
    if faces.size == 0:
        return 0
    v = verts[faces]
    tri_min = v.min(axis=1)
    tri_max = v.max(axis=1)
    if bin_size is None:
        span = (tri_max - tri_min).max(axis=0)
        bin_size = float(max(span.max() / 32.0, 1e-3))
    # bin by bbox centroid
    center = 0.5 * (tri_min + tri_max)
    cell = np.floor(center / bin_size).astype(np.int64)
    key = cell[:, 0] * 73856093 ^ cell[:, 1] * 19349663 ^ cell[:, 2] * 83492791
    order = np.argsort(key)
    key_sorted = key[order]
    _, start = np.unique(key_sorted, return_index=True)
    ends = np.append(start[1:], key_sorted.size)
    total = 0
    tested = 0
    count = 0
    for s, e in zip(start, ends):
        ids = order[s:e]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                total += 1
                if tested < max_pairs:
                    tested += 1
                    if _triangles_overlap(verts, faces[ids[x]], faces[ids[y]]):
                        count += 1
    if total == 0:
        return 0
    if total > max_pairs:
        count = int(round(count * total / max_pairs))
    return count


def _triangles_overlap(verts, ta, tb):
    if len(set(ta.tolist()) & set(tb.tolist())) > 0:
        return False  # share a vertex -> not a spurious intersection
    A = verts[ta]
    B = verts[tb]
    # Any edge of A intersecting the interior of B, or vice versa.
    for i in range(3):
        p, q = A[i], A[(i + 1) % 3]
        if _seg_tri(p, q, B):
            return True
        p, q = B[i], B[(i + 1) % 3]
        if _seg_tri(p, q, A):
            return True
    return False


def _seg_tri(p, q, T):
    """Möller–Trumbore segment-triangle intersection (segments fully contained
    are treated as non-intersecting; this is a spurious-overlap detector)."""
    e1 = T[1] - T[0]
    e2 = T[2] - T[0]
    d = q - p
    h = np.cross(d, e2)
    a = e1 @ h
    if abs(a) < 1e-12:
        return False
    f = 1.0 / a
    s = p - T[0]
    u = f * (s @ h)
    if u < 0.0 or u > 1.0:
        return False
    qq = np.cross(s, e1)
    v = f * (d @ qq)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * (e2 @ qq)
    return t > 1e-9 and t < 1.0 - 1e-9
