from functools import wraps

from sklearo.utils import infer_target_type


def check_X_y(func):
    @wraps(func)
    def wrapper(self, X, y, *args, **kwargs):
        if not X.shape[0] == y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, but got X.shape={X.shape} and "
                f"y.shape={y.shape}"
            )

        if y.null_count() > 0:
            raise ValueError(
                "y contains missing values. Please remove or impute missing values in y."
            )
        return func(self, X, y, *args, **kwargs)

    return wrapper


def check_if_fitted(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not any(
            attr.endswith("_") and not attr.startswith("__") for attr in vars(self)
        ):
            raise ValueError(
                f"{self.__class__.__name__} not fitted. Please fit {self.__class__.__name__} by "
                "calling the fit method first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def check_type_of_target(*allowed_types_of_target):
    def decorator(func):
        @wraps(func)
        def wrapper(self, X, y, *args, **kwargs):
            inferred_type_of_target = infer_target_type(y)
            if inferred_type_of_target not in allowed_types_of_target:
                raise ValueError(
                    f"{self.__class__.__name__} supports the following types of target: "
                    f"{allowed_types_of_target}, but the inferred type of target was "
                    f"type_of_target={inferred_type_of_target}. To know more on how target is "
                    "inferred please refer to the documentation of sklearo.utils.type_of_target."
                )
            return func(self, X, y, *args, **kwargs)

        return wrapper

    return decorator
