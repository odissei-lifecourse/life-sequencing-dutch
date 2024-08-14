import torch
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from pytorch_lightning.loggers import CSVLogger


# Step 0: Create some dummy data
class DummyIterableDataset(IterableDataset):
    def __init__(self, validation):
      self.validation = validation

    def __iter__(self):
        if self.validation:
          for i in range(1000):  # 10000 dummy data points
              x = torch.randn(10)  # 10 features
              x[0] = i
              y = torch.randn(1)   # 1 target value
              yield x, y
        else:
          for i in range(1000000):  # 10000 dummy data points
              x = torch.randn(10)  # 10 features
              x[0] = i
              y = torch.randn(1)   # 1 target value
              yield x, y
          

# Step 1: Use pytorch_lightning to create a dummy model to train on the dummy data
class DummyModel(pl.LightningModule):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.layer = torch.nn.Linear(10, 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss)
        #print(f"val_loss {val_loss}")
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Step 2: Write a custom_iterable_dataset
class CustomIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for x, y in self.data:
            yield x, y

# Step 3: Use a train_dataloader and a validation_dataloader, both objects of custom_iterable_dataset
train_data = DummyIterableDataset(validation=False)
val_data = DummyIterableDataset(validation=True)

train_dataloader = DataLoader(CustomIterableDataset(train_data), batch_size=64)
val_dataloader = DataLoader(CustomIterableDataset(val_data), batch_size=64)

# Step 4: Use val_check_interval=1000 for trainer
# Step 5: Use callback to save model after every 1000 steps
# Step 6: Save only the top 2 models
# Step 7: Save the validation_loss metric along with the models

# Define a ModelCheckpoint callback to save top 2 models based on validation loss
ckpoint_dir = 'ckpoints/'
if os.path.exists(ckpoint_dir) is False:
  os.mkdir(ckpoint_dir)
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpoint_dir,
    monitor='val_loss',
    save_top_k=2,
    mode='min',
    filename='dummy_models/{epoch:02d}-{step}-{val_loss:.2f}',
    save_last=False,
    save_weights_only=False,
    every_n_train_steps=1001,
    verbose=True,
)

# Initialize the PyTorch Lightning Trainer
logger = CSVLogger(ckpoint_dir)
trainer = pl.Trainer(
    default_root_dir=ckpoint_dir,
    max_epochs=5,
    val_check_interval=1000,
    callbacks=[checkpoint_callback],
    logger=logger,
)

# Instantiate the model
model = DummyModel()

# Train the model
print("skipped running the model!")
# trainer.fit(model, train_dataloader, val_dataloader)
