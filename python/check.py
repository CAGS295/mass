#!/bin/env/python

import numpy as np
import pandas as pd
import mass_ts as mts

import tarfile

def load_file(archive, fp):
    """
    Utility function that reads a tar file directly into a numpy array.
    
    Parameters
    ----------
    archive : str
        The archive file to read.
    fp : str
        The file path of the file to read relative to the archive.
    
    
    Returns
    -------
    None if data reading failed or the numpy array of values.
    """
    data = None
    with tarfile.open(archive) as a:
        f = a.extractfile(dict(zip(a.getnames(),a.getmembers()))[fp])
        data = pd.read_csv(f, header=None, names=['reading', ])['reading'].values
    
    return data

# total number of matches we want returned
top_matches = 5
# length of the subsequence to search in batch processing.
# note that this has an impact on memory usage
batch_size = 10000
# run same task with 4 cpu threads
n_jobs = 4

ecg = load_file('ecg.tar.gz', 'ecg.txt')
ecg_query = load_file('ecg_query.tar.gz', 'ecg_query.txt')


best_indices, best_dists = mts.mass2_batch(ecg, ecg_query, batch_size=batch_size, top_matches=top_matches, n_jobs=n_jobs)

print(best_indices)