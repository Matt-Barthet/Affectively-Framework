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
    Returns (mean_vector, valid_vectors, count). mean_vector is empty if
    no valid vectors exist.
    """

    valid_vectors = [np.array(v[-model.surrogate_length:]) for v in best_cell.trajectory_dict['raw_state']]
    try:
        print(valid_vectors[0][0])
    except:
        valid_vectors = [v for v in best_cell.trajectory_dict['arousal_vectors']]

    print()   
    print() 
    return np.mean(valid_vectors, axis=0), valid_vectors, len(valid_vectors)
