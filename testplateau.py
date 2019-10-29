import torch
import numpy as np
from lib.lr_scheduler import CustomReduceLROnPlateau

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                    optimizer, patience=4, factor=0.1, verbose=True,
#                                threshold_mode="abs", threshold=0.1)
scheduler = CustomReduceLROnPlateau(optimizer,
        patience=4, factor=0.1, improvement_thr=0.1,
        limit=3)

losses = np.round(np.linspace(1, 0.6, 31), 2)
#losses = [1.,  0.98, 0.96, 0.94, 0.92, 0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.68, 0.66, 0.64, 0.62, 0.6]
print(losses)

for i in range(len(losses)):
    print('Loss: '+str(i))
    scheduler.step(losses[i])
    print(scheduler.limit_cnt)
    #print(optimizer)
