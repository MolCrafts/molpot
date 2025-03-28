from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from ignite.engine import _check_arg
from tensordict import TensorDict

def model_transform(td: TensorDict) -> Tuple[TensorDict, TensorDict]:
    return td["predicts"], td["labels"]

def supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
) -> Callable:
    """Factory function for supervised training.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function that receives `y_pred` and `y`, and returns the loss as a tensor.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform: function that receives the output from the model and convert it into the form as required
            by the loss function
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        gradient_accumulation_steps: Number of steps the gradients should be accumulated across.
            (default: 1 (means no gradient accumulation))
        model_fn: the model function that receives `model` and `x`, and returns `y_pred`.

    Returns:
        Callable: update function.

    Examples:
        .. code-block:: python

            from ignite.engine import Engine, supervised_training_step

            model = ...
            optimizer = ...
            loss_fn = ...

            update_fn = supervised_training_step(model, optimizer, loss_fn, 'cuda')
            trainer = Engine(update_fn)

    .. versionadded:: 0.4.5
    .. versionchanged:: 0.4.7
        Added Gradient Accumulation.
    .. versionchanged:: 0.4.11
        Added `model_transform` to transform model's output
    .. versionchanged:: 0.4.13
        Added `model_fn` to customize model's application on the sample
    .. versionchanged:: 0.5.0
        Added support for ``mps`` device
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    def update(
        engine: Engine, td: TensorDict
    ) -> Union[Any, Tuple[torch.Tensor]]:
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        model.train()
        td = td.to(model.device)
        td = model(td)  # https://pytorch.org/tensordict/stable/tutorials/tensordict_module.html#do-s-and-don-t-with-tensordictmodule
        loss = loss_fn(
            *model_transform(td)
        )
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        loss.backward()
        td["loss"] = loss
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
        return td

    return update


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    deterministic: bool = False,
    amp_mode: Optional[str] = None,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
) -> Engine:
    """Factory function for creating a trainer for supervised models.

    Args:
        model: the model to train.
        optimizer: the optimizer to use.
        loss_fn: the loss function that receives `y_pred` and `y`, and returns the loss as a tensor.
        device: device type specification (default: None).
            Applies to batches after starting the engine. Model *will not* be moved.
            Device can be CPU, GPU or TPU.
        non_blocking: if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch: function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        model_transform: function that receives the output from the model and convert it into the form as required
            by the loss function
        output_transform: function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.
        deterministic: if True, returns deterministic engine of type
            :class:`~ignite.engine.deterministic.DeterministicEngine`, otherwise :class:`~ignite.engine.engine.Engine`
            (default: False).
        amp_mode: can be ``amp`` or ``apex``, model and optimizer will be casted to float16 using
            `torch.cuda.amp <https://pytorch.org/docs/stable/amp.html>`_ for ``amp`` and
            using `apex <https://nvidia.github.io/apex>`_ for ``apex``. (default: None)
        scaler: GradScaler instance for gradient scaling if `torch>=1.6.0`
            and ``amp_mode`` is ``amp``. If ``amp_mode`` is ``apex``, this argument will be ignored.
            If True, will create default GradScaler. If GradScaler instance is passed, it will be used instead.
            (default: False)
        gradient_accumulation_steps: Number of steps the gradients should be accumulated across.
            (default: 1 (means no gradient accumulation))
        model_fn: the model function that receives `model` and `x`, and returns `y_pred`.

    Returns:
        a trainer engine with supervised update function.

    Examples:

        Create a trainer

        .. code-block:: python

            from ignite.engine import create_supervised_trainer
            from ignite.utils import convert_tensor
            from ignite.handlers.tqdm_logger import ProgressBar

            model = ...
            loss = ...
            optimizer = ...
            dataloader = ...

            def prepare_batch_fn(batch, device, non_blocking):
                x = ...  # get x from batch
                y = ...  # get y from batch

                # return a tuple of (x, y) that can be directly runned as
                # `loss_fn(model(x), y)`
                return (
                    convert_tensor(x, device, non_blocking),
                    convert_tensor(y, device, non_blocking)
                )

            def output_transform_fn(x, y, y_pred, loss):
                # return only the loss is actually the default behavior for
                # trainer engine, but you can return anything you want
                return loss.item()

            trainer = create_supervised_trainer(
                model,
                optimizer,
                loss,
                prepare_batch=prepare_batch_fn,
                output_transform=output_transform_fn
            )

            pbar = ProgressBar()
            pbar.attach(trainer, output_transform=lambda x: {"loss": x})

            trainer.run(dataloader, max_epochs=5)

    Note:
        If ``scaler`` is True, GradScaler instance will be created internally and trainer state has attribute named
        ``scaler`` for that instance and can be used for saving and loading.

    Note:
        `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.

    .. warning::
        The internal use of `device` has changed.
        `device` will now *only* be used to move the input data to the correct device.
        The `model` should be moved by the user before creating an optimizer.
        For more information see:

        - `PyTorch Documentation <https://pytorch.org/docs/stable/optim.html#constructing-it>`_
        - `PyTorch's Explanation <https://github.com/pytorch/pytorch/issues/7844#issuecomment-503713840>`_

    .. warning::
        If ``amp_mode='apex'`` , the model(s) and optimizer(s) must be initialized beforehand
        since ``amp.initialize`` should be called after you have finished constructing your model(s)
        and optimizer(s), but before you send your model through any DistributedDataParallel wrapper.

        See more: https://nvidia.github.io/apex/amp.html#module-apex.amp

    .. versionchanged:: 0.4.5

        - Added ``amp_mode`` argument for automatic mixed precision.
        - Added ``scaler`` argument for gradient scaling.

    .. versionchanged:: 0.4.7
        Added Gradient Accumulation argument for all supervised training methods.
    .. versionchanged:: 0.4.11
        Added ``model_transform`` to transform model's output
    .. versionchanged:: 0.4.13
        Added `model_fn` to customize model's application on the sample
    .. versionchanged:: 0.5.0
        Added support for ``mps`` device
    """

    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _scaler = _check_arg(on_tpu, on_mps, amp_mode, scaler)

    if mode == "amp":
        raise NotImplementedError(
            "Automatic Mixed Precision (AMP) is not supported in this version of molpot. "
            "Please use torch.cuda.amp or apex for AMP."
        )
        # _update = supervised_training_step_amp(
        #     model,
        #     optimizer,
        #     loss_fn,
        #     device,
        #     non_blocking,
        #     prepare_batch,
        #     model_transform,
        #     output_transform,
        #     _scaler,
        #     gradient_accumulation_steps,
        #     model_fn,
        # )
    # elif mode == "apex":
    #     _update = supervised_training_step_apex(
    #         model,
    #         optimizer,
    #         loss_fn,
    #         device,
    #         non_blocking,
    #         prepare_batch,
    #         model_transform,
    #         output_transform,
    #         gradient_accumulation_steps,
    #         model_fn,
    #     )
    # elif mode == "tpu":
    #     _update = supervised_training_step_tpu(
    #         model,
    #         optimizer,
    #         loss_fn,
    #         device,
    #         non_blocking,
    #         prepare_batch,
    #         model_transform,
    #         output_transform,
    #         gradient_accumulation_steps,
    #         model_fn,
    #     )
    else:
        _update = supervised_training_step(
            model,
            optimizer,
            loss_fn,
            gradient_accumulation_steps,
            clip_grad_norm
        )

    trainer = Engine(_update) if not deterministic else DeterministicEngine(_update)
    if _scaler and scaler and isinstance(scaler, bool):
        trainer.state.scaler = _scaler  # type: ignore[attr-defined]

    return trainer


def supervised_evaluation_step(
    model: torch.nn.Module,
    grad_context: Callable = torch.no_grad,
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

    def evaluate_step(
        engine: Engine, td: TensorDict
    ) -> Union[Any, Tuple[torch.Tensor]]:
        model.eval()
        td = td.to(model.device)
        with grad_context():
            td = model(td)
            return td

    return evaluate_step


def create_supervised_evaluator(
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    amp_mode: Optional[str] = None,
    no_grad: bool = True,
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
    grad_context = torch.no_grad if no_grad else torch.enable_grad

    if mode == "amp":
        # evaluate_step = supervised_evaluation_step_amp(
        #     model,
        #     device,
        #     non_blocking=non_blocking,
        #     prepare_batch=prepare_batch,
        #     model_transform=model_transform,
        #     output_transform=output_transform,
        #     model_fn=model_fn,
        # )
        raise NotImplementedError(
            "Automatic Mixed Precision (AMP) is not supported in this version of molpot. "
            "Please use torch.cuda.amp or apex for AMP."
        )
    else:
        evaluate_step = supervised_evaluation_step(
            model,
            grad_context=grad_context,
        )

    evaluator = Engine(evaluate_step)

    return evaluator
