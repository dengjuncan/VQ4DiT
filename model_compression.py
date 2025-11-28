from typing import Callable, Dict, List, NewType, Optional, Set, Union

import torch

from CompressedLinear import CompressedLinear
from coding import get_kmeans_fn

RecursiveReplaceFn = NewType("RecursieReplaceFn", Callable[[torch.nn.Module, torch.nn.Module, int, str, str], bool])


def _replace_child(model: torch.nn.Module, child_name: str, compressed_child_model: torch.nn.Module, idx: int) -> None:
    """Replaces a given module into `model` with another module `compressed_child_model`

    Parameters:
        model: Model where we are replacing elements
        child_name: The key of `compressed_child_model` in the parent `model`. Used if `model` is a torch.nn.ModuleDict
        compressed_child_model: Child module to replace into `model`
        idx: The index of `compressed_child_model` in the parent `model` Used if `model` is a torch.nn.Sequential
    """
    if isinstance(model, torch.nn.Sequential):
        # Add back in the correct position
        model[idx] = compressed_child_model
    elif isinstance(model, torch.nn.ModuleDict):
        model[child_name] = compressed_child_model
    else:
        model.add_module(child_name, compressed_child_model)


def prefix_name_lambda(prefix: str) -> Callable[[str], str]:
    """Returns a function that preprends `prefix.` to its arguments.

    Parameters:
        prefix: The prefix that the return function will prepend to its inputs
    Returns:
        A function that takes as input a string and prepends `prefix.` to it
    """
    return lambda name: (prefix + "." + name) if prefix else name


@torch.no_grad()
def apply_recursively_to_model(fn, model: torch.nn.Module, prefix: str = "") -> None:
    """Recursively apply fn on all modules in models

    Parameters:
        fn: The callback function, it is given the parents, the children, the index of the children,
            the name of the children, and the prefixed name of the children
            It must return a boolean to determine whether we should stop recursing the branch
        model: The model we want to recursively apply fn to
        prefix: String to build the full name of the model's children (eg `layer1` in `layer1.conv1`)
    """
    get_prefixed_name = prefix_name_lambda(prefix)

    for idx, named_child in enumerate(model.named_children()):

        child_name, child = named_child
        child_prefixed_name = get_prefixed_name(child_name)

        if fn(model, child, idx, child_name, child_prefixed_name):
            continue
        else:
            apply_recursively_to_model(fn, child, child_prefixed_name)


def compress_model(
    model: torch.nn.Module,
    ignored_modules: Union[List[str], Set[str]],
    k: int,
    d: int,
) -> torch.nn.Module:
    """
    Given a neural network, modify it to its compressed representation with hard codes
      - Linear is replaced with compressed_layers.CompressedLinear
    Parameters:
        model: Network to compress. This will be modified in-place
        ignored_modules: List or set of submodules that should not be compressed
        k: Number of centroids to use for each compressed codebook
        d: Subvector size to use for linear layers
    Returns:
        The passed model, which is now compressed
    """

    def _compress_and_replace_layer(
        parent: torch.nn.Module, child: torch.nn.Module, idx: int, name: str, prefixed_child_name: str
    ) -> bool:
        """Compresses the `child` layer and replaces the uncompressed version into `parent`"""

        assert isinstance(parent, torch.nn.Module)
        assert isinstance(child, torch.nn.Module)

        if prefixed_child_name in ignored_modules:
            print(f"Ignoring {prefixed_child_name}")
            return True
        

        if isinstance(child, torch.nn.Linear):
            
            compressed_child = CompressedLinear.from_uncompressed(
                child, k, d, name=prefixed_child_name
            )
            _replace_child(parent, name, compressed_child, idx)
            return True

        else:
            return False

    apply_recursively_to_model(_compress_and_replace_layer, model)
    return model
