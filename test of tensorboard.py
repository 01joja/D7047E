
# Information https://pytorch.org/docs/stable/tensorboard.html
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer.add_scalar("henrik/anton", np.random.random(), n_iter)
    writer.add_scalar("henrik/bertil", np.random.random(), n_iter)
    writer.add_scalar("henrik/kalle", np.random.random(), n_iter)