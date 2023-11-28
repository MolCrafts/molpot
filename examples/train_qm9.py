import molpot as mpot

def load_qm9()->tuple[mpot.DataLoader, mpot.DataLoader]:

    qm9_dataset = mpot.QM9(data_dir="data/qm9", num_workers=0)
    dp = qm9_dataset.prepare()
    train, valid = dp.shuffle().set_length(1000).random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    train_dataloader = mpot.create_dataloader(train, num_workers=0)
    valid_dataloader = mpot.create_dataloader(valid, num_workers=0)
    return train_dataloader, valid_dataloader

def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader])->str:

    train_dataloader, valid_dataloader = load_qm9

    trainer = mpot.Trainer(mpot.PiNet(), mpot.MSELoss(), mpot.Adam(), mpot.Metric())
    trainer.train()
    return "done"
