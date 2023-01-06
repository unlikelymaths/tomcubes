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

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "defines.h"

void launchClassifyVoxel(
  dim3 grid, dim3 threads,
  const at::Tensor& volume, uint *voxel_verts,
  uint num_voxels, float iso_value,
  int b);

void launchClassifyVoxel(
  dim3 grid, dim3 threads,
  const at::Tensor& volume, uint *voxel_verts,
  uint num_voxels, float iso_value);

void launchGenerateTriangles(
  dim3 grid, dim3 threads,
  const at::Tensor& volume,
  at::Tensor& verts,
  uint *voxel_verts_scanned,
  uint num_voxels, float iso_value,
  float3 voxel_start, float3 voxel_size,
  int b);

void launchGradVolume(
  dim3 grid, dim3 threads,
  const at::Tensor& volume,
  const at::Tensor& verts_grad,
  at::Tensor& volume_grad,
  uint *voxel_verts_scanned,
  uint num_voxels, float iso_value,
  float3 voxel_start, float3 voxel_size);

void allocateTextures();

void destroyAllTextureObjects();

void thrustExclusiveScan(
  uint *output, uint *input, uint numElements);
