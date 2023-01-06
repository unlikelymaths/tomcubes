import torch
from tomcubes import MarchingCubes
import matplotlib.pyplot as plt

def plot_mesh(vertices, triangles, extend=(-1,1), show=True, figname=None):
    """Plot the mesh to a new figure using matplotlib"""
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu()
    if triangles is None:
        triangles = torch.arange(0, vertices.shape[0])
    if isinstance(triangles, torch.Tensor):
        triangles = triangles.detach().cpu()
    if len(vertices) == 0 or len(triangles) == 0:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if triangles is None:
        ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2])
    else:
        ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=triangles)
    ax.set_xlim(extend[0],extend[1])
    ax.set_ylim(extend[0],extend[1])
    ax.set_zlim(extend[0],extend[1])
    ax.set_box_aspect([1,1,1])
    if figname:
        plt.title(figname)
    if show:
        plt.show()

n = 16
x = torch.linspace(-1, 1, n, device='cuda')
volume = 1-torch.sqrt(torch.sum(torch.stack(torch.meshgrid(x, x, x))**2, dim=0))
volume = volume[None]
print(volume.shape)
# volume = torch.zeros((1, 3, 3, 3), device='cuda')
# volume[0, 1, 1, :2] = 1
# print(volume)

verts_list = MarchingCubes.apply(volume, 0.2, [0,0,0], [1,1,1])

plot_mesh(verts_list[0], None, extend=(0, 2), show=False)
plt.savefig("test.png")

print(verts_list[0].shape)
print(verts_list[0])

with open('mesh.obj', 'w') as f:
    f.writelines([
        f'v {verts_list[0][i,0]} {verts_list[0][i,1]} {verts_list[0][i,2]}\n'
        for i in range(verts_list[0].shape[0])
        ])
    f.writelines([
        f'f {1+3*i} {1+3*i+1} {1+3*i+2}\n'
        for i in range(verts_list[0].shape[0]//3)
        ])
