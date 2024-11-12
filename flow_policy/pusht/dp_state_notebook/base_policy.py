from torch import Tensor


class Policy:
    def __call__(self, obs: Tensor) -> Tensor:
        pass
