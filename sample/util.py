import torch
import meshplot as mp

def plot_mesh(verts):
    faces = torch.arange(0, verts.shape[0]).view(-1,3)
    mp.plot(verts.cpu().numpy(), faces.numpy())
