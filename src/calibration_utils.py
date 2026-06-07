from sklearn.calibration import CalibratedClassifierCV


def calibrate_prefit_classifier(estimator, X_val, y_val, method: str = "sigmoid"):
    """Calibrate an already-fitted classifier across sklearn versions."""
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated
