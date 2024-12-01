from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
from ignite.engine.engine import Engine
from ignite.metrics import Metric
from torch import nn
from tensordict import TensorDict
from ignite.engine import _check_arg
import contextlib


def convert_tensordict(
    x: TensorDict,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> TensorDict:
    """Move tensors to relevant device.

    Args:
        x: input tensor or mapping, or sequence of tensors.
        device: device type to move ``x``.
        non_blocking: convert a CPU Tensor with pinned memory to a CUDA Tensor
            asynchronously with respect to the host if possible
    """

    return x.to(device=device, non_blocking=non_blocking)


def _prepare_batch(
    batch: TensorDict,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[TensorDict, TensorDict]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    return (
        convert_tensordict(batch, device=device, non_blocking=non_blocking),  # data
        convert_tensordict(
            batch["labels"], device=device, non_blocking=non_blocking
        ),  # y
    )


def trainer_output_transform(x, y, y_pred, loss):
    return (y_pred, y, loss)


def eval_output_transform(x, y, y_pred):
    return (y_pred, y)


def model_transform(output):
    return output["predicts"]
    
def supervised_evaluation_step(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    no_grad: bool = True
) -> Callable:
    """
    Factory function for supervised evaluation.

    Args:
        model: the model to train.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform: function that receives the output from the model and convert it into the predictions:
            ``y_pred = model_transform(model(x))``.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.
        model_fn: the model function that receives `model` and `x`, and returns `y_pred`.

    Returns:
        Inference function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

    .. versionadded:: 0.4.5
    .. versionchanged:: 0.4.12
        Added ``model_transform`` to transform model's output
    .. versionchanged:: 0.4.13
        Added `model_fn` to customize model's application on the sample
    """

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        prepare_context = torch.no_grad if no_grad else contextlib.nullcontext
        with prepare_context():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            output = model_fn(model, x)
            y_pred = model_transform(output)
            return output_transform(x, y, y_pred)

    return evaluate_step


def supervised_evaluation_step_amp(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    no_grad: bool = True
) -> Callable:
    """
    Factory function for supervised evaluation using ``torch.cuda.amp``.

    Args:
        model: the model to train.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform: function that receives the output from the model and convert it into the predictions:
            ``y_pred = model_transform(model(x))``.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.
        model_fn: the model function that receives `model` and `x`, and returns `y_pred`.

    Returns:
        Inference function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

    .. versionadded:: 0.4.5
    .. versionchanged:: 0.4.12
        Added ``model_transform`` to transform model's output
    .. versionchanged:: 0.4.13
        Added `model_fn` to customize model's application on the sample
    """
    try:
        from torch.cuda.amp import autocast
    except ImportError:
        raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

    def evaluate_step(engine: Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        prepare_context = torch.no_grad if no_grad else contextlib.nullcontext
        with prepare_context():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            with autocast(enabled=True):
                output = model_fn(model, x)
                y_pred = model_transform(output)
            return output_transform(x, y, y_pred)

    return evaluate_step


def create_supervised_evaluator(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, Metric]] = None,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
    prepare_batch: Callable = _prepare_batch,
    model_transform: Callable[[Any], Any] = lambda output: output,
    output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, y),
    amp_mode: Optional[str] = None,
    model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    no_grad: bool = True
) -> Engine:
    """
    Factory function for creating an evaluator for supervised models.

    Args:
        model: the model to train.
        metrics: a map of metric names to Metrics.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform: function that receives the output from the model and convert it into the predictions:
            ``y_pred = model_transform(model(x))``.
        output_transform: function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics.
        amp_mode: can be ``amp``, model will be casted to float16 using
            `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_
        model_fn: the model function that receives `model` and `x`, and returns `y_pred`.

    Returns:
        an evaluator engine with supervised inference function.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is
        a tuple of `(batch_pred, batch_y)` by default.

    .. warning::

        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.

        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_

        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    .. versionchanged:: 0.4.5
        Added ``amp_mode`` argument for automatic mixed precision.
    .. versionchanged:: 0.4.12
        Added ``model_transform`` to transform model's output
    .. versionchanged:: 0.4.13
        Added `model_fn` to customize model's application on the sample
    .. versionchanged:: 0.5.0
        Added support for ``mps`` device
    """
    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _ = _check_arg(on_tpu, on_mps, amp_mode, None)

    metrics = metrics or {}
    if mode == "amp":
        evaluate_step = supervised_evaluation_step_amp(
            model,
            device,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            model_transform=model_transform,
            output_transform=output_transform,
            model_fn=model_fn,
            no_grad=no_grad
        )
    else:
        evaluate_step = supervised_evaluation_step(
            model,
            device,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            model_transform=model_transform,
            output_transform=output_transform,
            model_fn=model_fn,
            no_grad=no_grad
        )

    evaluator = Engine(evaluate_step)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
