"""Weighted prediction-error and correlation statistics."""
import numpy as np
from scipy.stats import spearmanr


def metrics(observed, predicted, weights=None):
    w = np.ones(len(observed)) if weights is None else np.asarray(weights)
    w = w/w.sum()
    om, pm = w@observed, w@predicted
    oc, pc = observed-om, predicted-pm
    vo, vp, cov = w@oc**2, w@pc**2, w@(oc*pc)
    error = predicted-observed
    return dict(n=len(w), r=float(cov/np.sqrt(vo*vp)),
        spearman_r=float(spearmanr(observed, predicted).statistic),
        R2=float(1-(w@error**2)/vo), RMSE=float(np.sqrt(w@error**2)),
        MAE=float(w@abs(error)), bias=float(w@error),
        centered_RMSE=float(np.sqrt(w@(pc-oc)**2)),
        observed_mean=float(om), predicted_mean=float(pm),
        observed_SD=float(np.sqrt(vo)), predicted_SD=float(np.sqrt(vp)),
        SD_ratio=float(np.sqrt(vp/vo)), slope_predicted_on_observed=float(cov/vo),
        observed_mean_baseline_RMSE=float(np.sqrt(vo)))
