# Engine: Core of molpot

Although forcefield training task is complex and diverse, the core of training always involves the same steps. With the great help of `pytorch-ignite`, we can easily implement and modify the training process. 
Let's start with a naive training process:

``` py
optimizer = torch.optim.Adam(potential.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.994)
loss_fn = mpot.engine.loss.Constraint()
loss_fn.add_loss(torch.nn.MSELoss(), "dipole", "mu", 1.0, "dipole", log=True)

trainer = mpot.PotentialTrainer(  # (1)! potential etc already defined in previous secion
    model=potential, optimizer=optimizer, loss_fn=loss_fn, device="cpu"
)
trainer.compile()
trainer.add_lr_scheduler(scheduler)
trainer.add_checkpoint("ckpt")

train_metric_usage = MetricUsage(
    started=Events.ITERATION_STARTED(every=100),
    completed=Events.ITERATION_COMPLETED(every=100),
    iteration_completed=Events.ITERATION_COMPLETED,
)
eval_metric_usage = MetricUsage(
    started=Events.EPOCH_STARTED,
    completed=Events.EPOCH_COMPLETED,
    iteration_completed=Events.ITERATION_COMPLETED,
)

trainer.set_metric_usage(
    trainer=train_metric_usage, evaluator=eval_metric_usage
)

trainer.add_metric(
    "d_mae",
    lambda: MeanAbsoluteError(
        output_transform=lambda x: (x["predicts", "dipole"], x["labels", "mu"]),
        device="cpu",
    ),
)
trainer.add_metric(
    "q_mae",
    lambda: MeanAbsoluteError(
        output_transform=lambda x: (
            x["predicts", "dipole"],
            torch.zeros_like(x["predicts", "dipole"]),
        ),
        device="cpu",
    ),
)

trainer.attach_tensorboard(log_dir="tblog")
trainer.run(train_data=train_dl, max_steps=max_steps, eval_data=eval_dl)
```

`PotentialTrainer` manages muiltiple `ignite` `Engine` instances for such as `training` and `evaluation`. So you can configure training settings uniformly.

## More than training

### TODO: MD engine
