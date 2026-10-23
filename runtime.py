"""Minimal reproducible NCC thermal inversion."""
import os

# Single-threaded BLAS keeps sums, and hence results, bitwise reproducible.
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
