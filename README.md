# ERA V2 S5 Assignment

This assignment is continuation of Assignment 4 with refactoring

### File Structure

There are 3 files

- `S5.ipynb` - the runner file
- `model.py` - it contains the defined model
- `utils.py` - contains the utility functions

---

### How to get sense of the code

- The python notebook `S5.ipynb` contains user defined code such as datasets and dataloaders and to run main functions defined in other files

- `model.py` contains the pytorch model of the type `nn.Module`

- `utils.py` has `visualize_data` function and a Model helper class which takes the following arguments: `ModelHelper(model, device, train_loader, test_loader)` and it has many functions:
  - `train(optimizer, scheduler, criterion, num_epochs)` - train the model
  - `plot()` - plot the model after training
  - `get_model_summary(input_size)` - get summary of the model

## Example usage:

### Initialize model

```python
model = Net() # any PyTorch model

mh = ModelHelper(model=model, device='cuda', train_loader=train_loader, test_loader=test_loader)
```

### Train model

```python
optimizer = optim.SGD(mh.model.to(mh.device).parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
criterion = F.nll_loss
num_epochs = 20

mh.train(optimizer, scheduler, criterion, num_epochs)
```

### Plot Results after training

```python
mh.plot()
```

### Get Model summary
```python
mh.get_model_summary()
```