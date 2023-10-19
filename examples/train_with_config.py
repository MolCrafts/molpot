import molpot as mpot

config = mpot.app.config_example

trainer = mpot.app.create_trainer(config)

trainer.train()