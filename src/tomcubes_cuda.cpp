/* Copyright (c) 2022, Markus Plack
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  Marching cubes

  This sample extracts a geometric isosurface from a volume dataset using
  the marching cubes algorithm. It uses the scan (prefix sum) function from
  the Thrust library to perform stream compaction.  Similar techniques can
  be used for other problems that require a variable-sized output per
  thread.

  For more information on marching cubes see:
  http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
  http://en.wikipedia.org/wiki/Marching_cubes

  Volume data courtesy:
  http://www9.informatik.uni-erlangen.de/External/vollib/

  For more information on the Thrust library
  http://code.google.com/p/thrust/

  The algorithm consists of several stages:

  1. Execute "classifyVoxel" kernel
  This evaluates the volume at the corners of each voxel and computes the
  number of vertices each voxel will generate.
  It is executed using one thread per voxel.
  It writes number of vertices for each voxel voxelVertices to global memory.

  2. Scan voxelVertices array
  This gives the start address for the vertex data for each voxel.
  We read back the total number of vertices generated from GPU to CPU.

  3. Execute "generateTriangles" kernel
  It looks up the field values again and generates the triangle data,
  using the results of the scan to write the output to the correct addresses.
  The marching cubes look-up tables are stored in 1D textures.
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tomcubes_kernel.h"

py::list forward(
    const at::Tensor volume, // [B, 1+C, Z, Y, X]
    float iso_value,
    py::tuple volume_start, // X, Y, Z
    py::tuple volume_end // X, y, Z
    ){ // [[N_1, 3+C], ..., [N_B, 3+C]]
  const int batch_size = volume.size(0);
  allocateTextures();
  py::list verts_list(batch_size);
  uint num_voxels = (volume.size(2) - 1) * (volume.size(3) - 1) * (volume.size(4) - 1);
  uint num_attributes = volume.size(1) - 1;
  uint *d_voxel_verts = 0; // Buffer for vertex start address for each voxel
  float3 voxel_start = make_float3(
    volume_start[0].cast<float>(),
    volume_start[1].cast<float>(),
    volume_start[2].cast<float>());
  float3 voxel_size = make_float3(
    (volume_end[0].cast<float>() - volume_start[0].cast<float>()) / volume.size(4),
    (volume_end[1].cast<float>() - volume_start[1].cast<float>()) / volume.size(3),
    (volume_end[2].cast<float>() - volume_start[2].cast<float>()) / volume.size(2));
  cudaMalloc((void **)&d_voxel_verts, sizeof(uint) * (num_voxels + 1));
  dim3 grid(((num_voxels + 1) + NTHREADS - 1) / NTHREADS, 1, 1);
  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535) {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }
  for (int b = 0; b<batch_size; ++b) {
    // calculate number of vertices need per voxel
    launchClassifyVoxel(grid, NTHREADS, volume, d_voxel_verts, num_voxels,
                        iso_value, b);
    // scan voxel vertex count array
    thrustExclusiveScan(d_voxel_verts, d_voxel_verts, num_voxels + 1);
    // readback total number of vertices
    uint totalVerts;
    cudaMemcpy((void *)&totalVerts,
               (void *)(d_voxel_verts + num_voxels),
               sizeof(uint), cudaMemcpyDeviceToHost);
    at::Tensor verts = at::empty(
      {totalVerts, 3+num_attributes},
      c10::ScalarType::Float,
      c10::Layout::Strided,
      volume.device(),
      false,
      c10::MemoryFormat::Contiguous);
    // generate triangles
    dim3 grid2((num_voxels + NTHREADS - 1) / NTHREADS, 1, 1);
    while (grid2.x > 65535) {
      grid2.x /= 2;
      grid2.y *= 2;
    }
    launchGenerateTriangles(grid2, NTHREADS, volume, verts, d_voxel_verts,
                            num_voxels, iso_value, voxel_start, voxel_size, b);
    verts_list[b] = verts;
  }
  destroyAllTextureObjects();
  cudaFree(d_voxel_verts);
  return verts_list;
}

void grad_volume(
    const at::Tensor volume, // [C+1, Z, Y, X]
    const at::Tensor verts_grad, // [N, 3+C]
    at::Tensor volume_grad, // [C+1, Z, Y, X]
    float iso_value,
    py::tuple volume_start, // X, y, Z
    py::tuple volume_end // X, y, Z
    ) {
  allocateTextures();
  uint num_voxels = (volume.size(1) - 1) * (volume.size(2) - 1) * (volume.size(3) - 1);
  uint *d_voxel_verts = 0; // Buffer for vertex start address for each voxel
  float3 voxel_start = make_float3(
    volume_start[0].cast<float>(),
    volume_start[1].cast<float>(),
    volume_start[2].cast<float>());
  float3 voxel_size = make_float3(
    (volume_end[0].cast<float>() - volume_start[0].cast<float>()) / volume.size(1),
    (volume_end[1].cast<float>() - volume_start[1].cast<float>()) / volume.size(2),
    (volume_end[2].cast<float>() - volume_start[2].cast<float>()) / volume.size(3));
  cudaMalloc((void **)&d_voxel_verts, sizeof(uint) * num_voxels);
  dim3 grid((num_voxels + NTHREADS - 1) / NTHREADS, 1, 1);
  // get around maximum grid size of 65535 in each dimension
  if (grid.x > 65535) {
    grid.y = grid.x / 32768;
    grid.x = 32768;
  }
  // calculate number of vertices need per voxel
  launchClassifyVoxel(grid, NTHREADS, volume, d_voxel_verts, num_voxels,
                        iso_value);
  // scan voxel vertex count array
  thrustExclusiveScan(d_voxel_verts, d_voxel_verts, num_voxels);
  // compute grad
  dim3 grid2((num_voxels + NTHREADS - 1) / NTHREADS, 1, 1);
  while (grid2.x > 65535) {
    grid2.x /= 2;
    grid2.y *= 2;
  }
  launchGradVolume(
    grid2, NTHREADS, volume, verts_grad, volume_grad, d_voxel_verts, num_voxels,
    iso_value, voxel_start, voxel_size);
  // Cleanup
  destroyAllTextureObjects();
  cudaFree(d_voxel_verts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Torch Marching Cubes Forward (CUDA)");
  m.def("grad_volume", &grad_volume, "Torch Marching Cubes Gradient w.r.t. volume (CUDA)");
}
