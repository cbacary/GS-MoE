import torch
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
from torch import Tensor

from typing import Callable


class LatentLinear(torch.nn.Module):

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # shared_weight: Parameter,
        shared_weight_getter: Callable[[], torch.Tensor],
        bias: bool = True,
        device=None,
        dtype=None,
        flipped_down: bool = True,
        is_down: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flipped_down = flipped_down

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # self._shared = {"weight": shared_weight}
        self.get_shared_weight = shared_weight_getter
        self.is_down = is_down
        # self.shared_weight = shared_weight
        # self.register_buffer("shared_weight", shared_weight.data, persistent=False)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, use_shared: bool = True) -> Tensor:
        """
        Runs the forward pass.
        """

        # up_proj [d, d_int] --> stored transposed: [d_int, d]
        # shared_up [d, r]

        shared_weight = self.get_shared_weight()

        if shared_weight is not None:
            if not self.flipped_down:
                # this is when original factorization
                latent = input @ shared_weight
                return latent @ self.weight.T

            # print(
            #     f"input.shape = {input.shape}, self.weight.shape = {self.weight.shape}, shared_weight.shape = {shared_weight.shape}"
            # )
            if self.is_down:
                latent = input @ self.weight.T
                return latent @ shared_weight.T if use_shared else latent
            else:
                if use_shared:
                    input = input @ shared_weight

                # input has [T, r] since shared weight already transformed input
                return input @ self.weight.T

        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    # def state_dict(self, *args, **kwargs):
    #     # Get the normal state dict, then remove shared_weight
    #     # It's already saved once via the parent block's ParameterDict
    #     sd = super().state_dict(*args, **kwargs)
    #     keys_to_remove = [k for k in sd if "shared_weight" in k]
    #     for k in keys_to_remove:
    #         del sd[k]
    #     return sd

    # def _load_from_state_dict(
    #     self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    # ):
    #     # shared_weight won't be in the state dict (we excluded it during save),
    #     # so temporarily set strict=False for this module to avoid "missing key" errors
    #     super()._load_from_state_dict(
    #         state_dict,
    #         prefix,
    #         local_metadata,
    #         strict=False,  # don't complain about missing shared_weight
    #         missing_keys=missing_keys,
    #         unexpected_keys=unexpected_keys,
    #         error_msgs=error_msgs,
    #     )
