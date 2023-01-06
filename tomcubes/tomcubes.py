"""tomcubes"""
import torch
import typing

import tomcubes_cuda

class MarchingCubesFunction(torch.autograd.Function):
    """Run marching cubes on volume to generate vertices

    All entries int the channel dimension after the first will be interpreted
    as attributes and linearly interpolated.

    Args:
        volume (torch.Tensor): Volume with optional attributes [B, 1+C, Z, Y, X]
        iso_value (float): Iso value for marching cubes
        volume_start (typing.Tuple[float, float, float]): (x,y,z) start coordinates of the volume
        volume_end (typing.Tuple[float, float, float]): (x,y,z) end coordinates of the volume

    Returns:
        typing.Tuple[torch.Tensor]: Tuple of length B containing verticex coordinates and
                                    interpolated attributes for each mesh ([N_1,3+C],...,[N_B,3+C])
    """

    @staticmethod
    def forward(ctx, volume: torch.Tensor, iso_value: float,
                volume_start: typing.Tuple[float, float, float],
                volume_end: typing.Tuple[float, float, float]
                ) -> typing.Tuple[torch.Tensor]:
        ctx.set_materialize_grads(False)
        # Volume
        if volume.ndim != 5:
            raise ValueError(
                f'Volume must have shape [B, 1+C, Z, Y, X]. '
                f'Got shape {volume.shape}.')
        ctx.save_for_backward(volume)
        # Parameters
        ctx.iso_value = iso_value
        ctx.volume_start = volume_start
        ctx.volume_end = volume_end
        # Marching Cubes
        verts_list = tomcubes_cuda.forward(
            volume, iso_value, tuple(volume_start), tuple(volume_end))
        # Return as tuple
        return tuple(verts_list)

    @staticmethod
    def backward(ctx, *verts_grad_list: typing.Tuple[torch.Tensor]):
        grad = [None, None, None, None]
        if verts_grad_list is None or all([v is None for v in verts_grad_list]):
            return tuple(grad)
        if ctx.needs_input_grad[0]: # volume
            volume, = ctx.saved_tensors
            volume_grad = torch.zeros_like(volume)
            for b, verts_grad in enumerate(verts_grad_list):
                if verts_grad is not None:
                    tomcubes_cuda.grad_volume(
                        volume[b], verts_grad, volume_grad[b], ctx.iso_value,
                        tuple(ctx.volume_start), tuple(ctx.volume_end))
            grad[0] = volume_grad
        if ctx.needs_input_grad[1]: # iso_value
            raise NotImplementedError()
        if ctx.needs_input_grad[2]: # volume_start
            raise NotImplementedError()
        if ctx.needs_input_grad[3]: # volume_end
            raise NotImplementedError()
        return tuple(grad)
