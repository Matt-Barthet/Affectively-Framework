import numpy as np


def get_best_cell(archive):
    best_cell = list(archive.values())[0]
    for cell in archive.values():
        if cell.reward > best_cell.reward:
            best_cell = cell
    return best_cell


def process_surrogate_vectors(best_cell, model):
    """
    Extract valid surrogate vectors from an archive best cell.
    Deduplicates consecutive identical vectors so each generate_arousal window
    contributes one entry regardless of how many steps it covered. This also
    removes stale entries recorded before the first generate_arousal fires in
    each exploration segment (they are exact duplicates of the previous window).
    Returns (mean_vector, valid_vectors, count). mean_vector is empty if
    no valid vectors exist.
    """
    all_vectors = best_cell.trajectory_dict['arousal_vectors']
    valid_vectors = [v for v in all_vectors if len(v) in (model.surrogate_length, model.surrogate_length + 1)]

    if not valid_vectors:
        return np.array([]), [], 0

    # Keep only the first occurrence of each run of identical vectors.
    # Compare only the surrogate portion; the appended episode_length changes
    # every step and would otherwise prevent any deduplication.
    n_surr = model.surrogate_length
    unique_vectors = [valid_vectors[0]]
    for v in valid_vectors[1:]:
        if not np.array_equal(v[:n_surr], unique_vectors[-1][:n_surr]):
            unique_vectors.append(v)

    return np.mean(unique_vectors, axis=0), unique_vectors, len(unique_vectors)
