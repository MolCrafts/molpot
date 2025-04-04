from typing import Any, Callable, Optional, Tuple, Union

import torch
from ignite.engine import _check_arg
from ignite.engine.deterministic import DeterministicEngine
from ignite.engine.engine import Engine
from tensordict import TensorDict

def model_transform(td: TensorDict) -> Tuple[TensorDict, TensorDict]:
    return td["predicts"], td["labels"]

def supervised_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: Union[Callable[[TensorDict, TensorDict], torch.Tensor], torch.nn.Module],
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
) -> Callable:
    """Creates and returns a function that performs a single training step in a supervised learning setting.
    The returned function is designed to be used as an update function with an ignite Engine.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer | None
        The optimizer to use for training. If None, the returned update function
        will only perform a forward pass without parameter updates.
    loss_fn : Union[Callable[[TensorDict, TensorDict], torch.Tensor], torch.nn.Module]
        The loss function to compute model loss. Takes the outputs of model_transform(td)
        as arguments and returns a tensor.
    gradient_accumulation_steps : int, default=1
        Number of steps to accumulate gradients before performing an optimizer step.
        When set to 1 (default), no gradient accumulation is performed.
    clip_grad_norm : Optional[float], default=None
        Maximum norm for gradient clipping. None means no gradient clipping.

    Returns
    -------
    Callable
        A function that takes an engine and a tensordict as inputs and returns a tensordict
        with updated values. This function handles forward pass, loss calculation, and when
        optimizer is not None, gradient computation and parameter updates.

    Raises
    ------
    ValueError
        If gradient_accumulation_steps is not a positive integer.

    Notes
    -----
    The returned update function expects the input TensorDict to be compatible with the model.
    It will move the TensorDict to the model's device before processing.

    When optimizer is provided, the returned function:
    1. Zeroes gradients at appropriate intervals based on gradient_accumulation_steps
    2. Performs forward pass via model(td)
    3. Calculates loss using the provided loss_fn
    4. Scales loss if using gradient accumulation
    5. Applies gradient clipping if specified
    6. Performs backward pass
    7. Updates parameters at appropriate intervals based on gradient_accumulation_steps
    8. Stores the loss in td["loss"]
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    if optimizer is None:
        def update(engine: Engine, td: TensorDict) -> Union[Any, Tuple[torch.Tensor]]:
            model.train()
            td = td.to(model.device)
            td = model(td)
            return td
        return update
    else:
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
            # TODO: engine.state.loss = loss?
            if engine.state.iteration % gradient_accumulation_steps == 0:
                optimizer.step()
            return td

    return update


def create_supervised_trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Union[Callable[[TensorDict, TensorDict], torch.Tensor], torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    deterministic: bool = False,
    amp_mode: Optional[str] = None,
    scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
    gradient_accumulation_steps: int = 1,
    clip_grad_norm: Optional[float] = None,
) -> Engine:
    """
    Creates a supervised training engine for a given model, optimizer, and loss function.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        loss_fn (Union[Callable[[TensorDict, TensorDict], torch.Tensor], torch.nn.Module]): 
            The loss function to compute the loss between predictions and targets.
        device (Optional[Union[str, torch.device]]): The device to run the training on. 
            Can be a string (e.g., "cpu", "cuda") or a `torch.device` object. Defaults to None.
        deterministic (bool): If True, uses a deterministic engine for training. Defaults to False.
        amp_mode (Optional[str]): The automatic mixed precision (AMP) mode to use. 
            Can be "amp" or "apex". Defaults to None.
        scaler (Union[bool, "torch.cuda.amp.GradScaler"]): If True, enables AMP with a GradScaler. 
            Can also pass a custom GradScaler instance. Defaults to False.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating 
            the model parameters. Defaults to 1.
        clip_grad_norm (Optional[float]): Maximum norm of gradients for gradient clipping. 
            If None, gradient clipping is not applied. Defaults to None.

    Returns:
        Engine: An Ignite Engine object that runs the supervised training process.
    """
    device_type = device.type if isinstance(device, torch.device) else device
    on_tpu = "xla" in device_type if device_type is not None else False
    on_mps = "mps" in device_type if device_type is not None else False
    mode, _scaler = _check_arg(on_tpu, on_mps, amp_mode, scaler)


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
    ) -> Union[TensorDict, Tuple[torch.Tensor]]:
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
