import torch
import sys
import os

sys.path.append("../2_train_models")

assert len(sys.argv) == 3, len(sys.argv)
in_path = sys.argv[1]
out_path = sys.argv[2]
assert os.path.exists(in_path), in_path

print("Converting ", in_path, " to ", out_path)

model = torch.load(in_path)
torch.save(model.state_dict(), out_path)

print("Model converted.")