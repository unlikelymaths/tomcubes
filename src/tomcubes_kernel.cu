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

#include <stdio.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "tomcubes_kernel.h"

#include "tables.h"

// textures containing look-up tables
uint *d_triTable = 0;
uint *d_numVertsTable = 0;
cudaTextureObject_t triTex;
cudaTextureObject_t numVertsTex;

void allocateTextures() {
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
  // d_triTable
  cudaMalloc((void **)&d_triTable, 256 * 16 * sizeof(uint));
  cudaMemcpy((void *)d_triTable, (void *)triTable,
             256 * 16 * sizeof(uint), cudaMemcpyHostToDevice);

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_triTable;
  texRes.res.linear.sizeInBytes = 256 * 16 * sizeof(uint);
  texRes.res.linear.desc = channelDesc;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&triTex, &texRes, &texDescr, NULL);

  // d_numVertsTable
  cudaMalloc((void **)&d_numVertsTable, 256 * sizeof(uint));
  cudaMemcpy((void *)d_numVertsTable, (void *)numVertsTable,
             256 * sizeof(uint), cudaMemcpyHostToDevice);

  memset(&texRes, 0, sizeof(cudaResourceDesc));

  texRes.resType = cudaResourceTypeLinear;
  texRes.res.linear.devPtr = d_numVertsTable;
  texRes.res.linear.sizeInBytes = 256 * sizeof(uint);
  texRes.res.linear.desc = channelDesc;

  memset(&texDescr, 0, sizeof(cudaTextureDesc));

  texDescr.normalizedCoords = false;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  cudaCreateTextureObject(&numVertsTex, &texRes, &texDescr, NULL);
}

void destroyAllTextureObjects() {
  cudaDestroyTextureObject(triTex);
  cudaDestroyTextureObject(numVertsTex);
  cudaFree(d_triTable);
  cudaFree(d_numVertsTable);
}

// compute position in 3d grid from 1d index
__device__ inline uint3 calcGridPos(
    uint i,
    const at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits>& volume) {
  return make_uint3(
    i % (volume.size(3)-1),
    (i / (volume.size(3)-1)) % (volume.size(2)-1),
    (i / (volume.size(3)-1)) / (volume.size(2)-1)
  );
}

__device__ inline uint3 calcGridPos(
    uint i,
    const at::PackedTensorAccessor32<float, 5, at::DefaultPtrTraits>& volume) {
  return make_uint3(
    i % (volume.size(4)-1),
    (i / (volume.size(4)-1)) % (volume.size(3)-1),
    (i / (volume.size(4)-1)) / (volume.size(3)-1)
  );
}

__device__ inline void sampleField(
    float (&field)[8],
    const int b,
    const int c,
    const uint3& grid_pos,
    const at::PackedTensorAccessor32<float, 5, at::DefaultPtrTraits>& volume) {
  field[0] = volume[b][c][grid_pos.z  ][grid_pos.y  ][grid_pos.x  ];
  field[1] = volume[b][c][grid_pos.z  ][grid_pos.y  ][grid_pos.x+1];
  field[2] = volume[b][c][grid_pos.z  ][grid_pos.y+1][grid_pos.x+1];
  field[3] = volume[b][c][grid_pos.z  ][grid_pos.y+1][grid_pos.x  ];
  field[4] = volume[b][c][grid_pos.z+1][grid_pos.y  ][grid_pos.x  ];
  field[5] = volume[b][c][grid_pos.z+1][grid_pos.y  ][grid_pos.x+1];
  field[6] = volume[b][c][grid_pos.z+1][grid_pos.y+1][grid_pos.x+1];
  field[7] = volume[b][c][grid_pos.z+1][grid_pos.y+1][grid_pos.x  ];
}

__device__ inline void sampleField(
    float (&field)[8],
    const int c,
    const uint3& grid_pos,
    const at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits>& volume) {
  field[0] = volume[c][grid_pos.z  ][grid_pos.y  ][grid_pos.x  ];
  field[1] = volume[c][grid_pos.z  ][grid_pos.y  ][grid_pos.x+1];
  field[2] = volume[c][grid_pos.z  ][grid_pos.y+1][grid_pos.x+1];
  field[3] = volume[c][grid_pos.z  ][grid_pos.y+1][grid_pos.x  ];
  field[4] = volume[c][grid_pos.z+1][grid_pos.y  ][grid_pos.x  ];
  field[5] = volume[c][grid_pos.z+1][grid_pos.y  ][grid_pos.x+1];
  field[6] = volume[c][grid_pos.z+1][grid_pos.y+1][grid_pos.x+1];
  field[7] = volume[c][grid_pos.z+1][grid_pos.y+1][grid_pos.x  ];
}

__device__ inline float sampleAttribute(
    const int v,
    const int b,
    const int c,
    const uint3& grid_pos,
    const at::PackedTensorAccessor32<float, 5, at::DefaultPtrTraits>& volume) {
  const int z = grid_pos.z + int(v > 3);
  const int y = grid_pos.y + int(v % 4 > 1);
  const int x = grid_pos.x + int((v + 1) % 4 > 1);
  return volume[b][c+1][z][y][x];
}

__device__ inline float sampleAttribute(
    const int v,
    const int c,
    const uint3& grid_pos,
    const at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits>& volume) {
  const int z = grid_pos.z + int(v > 3);
  const int y = grid_pos.y + int(v % 4 > 1);
  const int x = grid_pos.x + int((v + 1) % 4 > 1);
  return volume[c+1][z][y][x];
}

__device__ inline void writeGrad(
    const float (&grad)[8],
    const int c,
    const uint3& grid_pos,
    at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits>& volume_grad) {
  if (grad[0] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z  ][grid_pos.y  ][grid_pos.x  ], grad[0]);
  if (grad[1] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z  ][grid_pos.y  ][grid_pos.x+1], grad[1]);
  if (grad[2] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z  ][grid_pos.y+1][grid_pos.x+1], grad[2]);
  if (grad[3] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z  ][grid_pos.y+1][grid_pos.x  ], grad[3]);
  if (grad[4] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z+1][grid_pos.y  ][grid_pos.x  ], grad[4]);
  if (grad[5] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z+1][grid_pos.y  ][grid_pos.x+1], grad[5]);
  if (grad[6] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z+1][grid_pos.y+1][grid_pos.x+1], grad[6]);
  if (grad[7] != 0.f)
    atomicAdd(&volume_grad[c][grid_pos.z+1][grid_pos.y+1][grid_pos.x  ], grad[7]);
}

__device__ inline uint computeCubeIndex(
    const float (&field)[8],
    const float iso_value) { 
  uint cubeindex;
  cubeindex  = uint(field[0] < iso_value);
  cubeindex += uint(field[1] < iso_value) * 2;
  cubeindex += uint(field[2] < iso_value) * 4;
  cubeindex += uint(field[3] < iso_value) * 8;
  cubeindex += uint(field[4] < iso_value) * 16;
  cubeindex += uint(field[5] < iso_value) * 32;
  cubeindex += uint(field[6] < iso_value) * 64;
  cubeindex += uint(field[7] < iso_value) * 128;
  return cubeindex;
}

__device__ inline int2 getEdgeVertexIndices(const uint edge) {
  int2 vidx;
  switch (edge) {
    case  0: vidx.x = 0; vidx.y = 1; break;
    case  1: vidx.x = 1; vidx.y = 2; break;
    case  2: vidx.x = 2; vidx.y = 3; break;
    case  3: vidx.x = 3; vidx.y = 0; break;
    case  4: vidx.x = 4; vidx.y = 5; break;
    case  5: vidx.x = 5; vidx.y = 6; break;
    case  6: vidx.x = 6; vidx.y = 7; break;
    case  7: vidx.x = 4; vidx.y = 7; break;
    case  8: vidx.x = 0; vidx.y = 4; break;
    case  9: vidx.x = 1; vidx.y = 5; break;
    case 10: vidx.x = 2; vidx.y = 6; break;
    case 11: vidx.x = 3; vidx.y = 7; break;
  }
  return vidx;
}

// tuple functions
__device__ inline float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ inline float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ inline float3 operator*(const float& a, const float3& b) {
  return make_float3(a * b.x, a * b.y, a * b.z);
}

// lerp - linear interpolation between a and b, based on value t in [0, 1] range
__device__ inline float lerp(const float a, const float b, const float t)
{
    return a + t*(b-a);
}
__device__ inline float3 lerp(const float3& a, const float3& b, const float t)
{
    return a + t*(b-a);
}
// compute interpolated vertex along an edge
__device__ inline float3 vertexInterp(
    const float isolevel, const float3& p0, const float3& p1,
    const float f0, const float f1) {
  float t = (isolevel - f0) / (f1 - f0);
  return lerp(p0, p1, t);
}

__device__ inline void vertexInterpGrad(
    float& f0_grad, float& f1_grad,
    const float verts_grad, const float iso_value,
    const float p0, const float p1,
    const float f0, const float f1 ) {
  const float df = fabsf(f1 - f0);
  if (df > 1.e-6) {
    const float common = verts_grad * (p1 - p0) / powf(df, 2.f);
    f0_grad += common * (iso_value - f1);
    f1_grad += common * (f0 - iso_value);
  }
}

// compute interpolated attribute along an edge
__device__ inline float attribInterp(float isolevel, float a0, float a1, float f0,
                               float f1) {
  float t = (isolevel - f0) / (f1 - f0);
  return lerp(a0, a1, t);
}

__device__ inline void attribInterpGradAttrib(
    float& a0_grad, float& a1_grad,
    const float attrib_grad, float isolevel,
    const float f0, const float f1) {
  const float common = (isolevel - f0) / (f1-f0);
  a0_grad += attrib_grad * (1.f - common);
  a1_grad += attrib_grad * common;
}

// classify voxel based on number of vertices it will generate
// one thread per voxel
__global__ void classifyVoxel(
    const at::PackedTensorAccessor32<float, 5, at::DefaultPtrTraits> volume,
    uint *voxel_verts, uint num_voxels, float iso_value,
    cudaTextureObject_t numVertsTex, int b) {
  uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
  uint numVerts = 0;
  if (i > num_voxels) {
    return;
  }
  if (i < num_voxels) {
    const uint3 grid_pos = calcGridPos(i, volume);
    float field[8];
    sampleField(field, b, 0, grid_pos, volume);
    const uint cubeindex = computeCubeIndex(field, iso_value);
    numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);
  }
  voxel_verts[i] = numVerts;
}

void launchClassifyVoxel(
    dim3 grid, dim3 threads,
    const at::Tensor& volume, uint *voxel_verts,
    uint num_voxels, float iso_value,
    int b) {
  classifyVoxel<<<grid, threads>>>(
    volume.packed_accessor32<float, 5, torch::DefaultPtrTraits>(),
    voxel_verts, num_voxels, iso_value, numVertsTex, b);
}

__global__ void classifyVoxel(
    const at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits> volume,
    uint *voxel_verts, uint num_voxels, float iso_value,
    cudaTextureObject_t numVertsTex) {
  uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
  if (i < num_voxels) {
    const uint3 grid_pos = calcGridPos(i, volume);
    float field[8];
    sampleField(field, 0, grid_pos, volume);
    const uint cubeindex = computeCubeIndex(field, iso_value);
    voxel_verts[i] = tex1Dfetch<uint>(numVertsTex, cubeindex);
  }
}

void launchClassifyVoxel(
    dim3 grid, dim3 threads,
    const at::Tensor& volume, uint *voxel_verts,
    uint num_voxels, float iso_value) {
  classifyVoxel<<<grid, threads>>>(
    volume.packed_accessor32<float, 4, torch::DefaultPtrTraits>(),
    voxel_verts, num_voxels, iso_value, numVertsTex);
}

__global__ void generateTriangles(
    const at::PackedTensorAccessor32<float, 5, at::DefaultPtrTraits> volume,
    at::PackedTensorAccessor32<float, 2, at::DefaultPtrTraits> verts,
    uint *voxel_verts_scanned,
    uint num_voxels, float iso_value,
    float3 voxel_start, float3 voxel_size,
    cudaTextureObject_t numVertsTex, cudaTextureObject_t triTex,
    int b) {
  const uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  const uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
  const uint num_attributes = volume.size(1) - 1;
  uint numVerts;
  uint cubeindex;
  uint3 grid_pos;
  float field[8];
  __shared__ float3 vertlist[12 * NTHREADS];
  if (i < num_voxels) {
    // compute position in 3d grid
    grid_pos = calcGridPos(i, volume);

    float3 p = make_float3(
      voxel_start.x + (grid_pos.x * voxel_size.x),
      voxel_start.y + (grid_pos.y * voxel_size.y),
      voxel_start.z + (grid_pos.z * voxel_size.z));

    // calculate cell vertex positions
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(voxel_size.x, 0, 0);
    v[2] = p + make_float3(voxel_size.x, voxel_size.y, 0);
    v[3] = p + make_float3(0, voxel_size.y, 0);
    v[4] = p + make_float3(0, 0, voxel_size.z);
    v[5] = p + make_float3(voxel_size.x, 0, voxel_size.z);
    v[6] = p + make_float3(voxel_size.x, voxel_size.y, voxel_size.z);
    v[7] = p + make_float3(0, voxel_size.y, voxel_size.z);

    sampleField(field, b, 0, grid_pos, volume);
    cubeindex = computeCubeIndex(field, iso_value);
    numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

    // find the vertices where the surface intersects the cube
    // use shared memory to avoid using local
    vertlist[                  threadIdx.x] = vertexInterp(iso_value, v[0], v[1], field[0], field[1]);
    vertlist[(NTHREADS     ) + threadIdx.x] = vertexInterp(iso_value, v[1], v[2], field[1], field[2]);
    vertlist[(NTHREADS *  2) + threadIdx.x] = vertexInterp(iso_value, v[2], v[3], field[2], field[3]);
    vertlist[(NTHREADS *  3) + threadIdx.x] = vertexInterp(iso_value, v[3], v[0], field[3], field[0]);
    vertlist[(NTHREADS *  4) + threadIdx.x] = vertexInterp(iso_value, v[4], v[5], field[4], field[5]);
    vertlist[(NTHREADS *  5) + threadIdx.x] = vertexInterp(iso_value, v[5], v[6], field[5], field[6]);
    vertlist[(NTHREADS *  6) + threadIdx.x] = vertexInterp(iso_value, v[6], v[7], field[6], field[7]);
    vertlist[(NTHREADS *  7) + threadIdx.x] = vertexInterp(iso_value, v[4], v[7], field[4], field[7]);
    vertlist[(NTHREADS *  8) + threadIdx.x] = vertexInterp(iso_value, v[0], v[4], field[0], field[4]);
    vertlist[(NTHREADS *  9) + threadIdx.x] = vertexInterp(iso_value, v[1], v[5], field[1], field[5]);
    vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(iso_value, v[2], v[6], field[2], field[6]);
    vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(iso_value, v[3], v[7], field[3], field[7]);
  }
  __syncthreads();
  if (i < num_voxels) {
    // output triangle vertices
    for (int j = 0; j < numVerts; ++j) {
      const uint vert_index = voxel_verts_scanned[i] + j;
      const uint edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + j);
      const float3 v = vertlist[(edge * NTHREADS) + threadIdx.x];
      verts[vert_index][0] = v.x;
      verts[vert_index][1] = v.y;
      verts[vert_index][2] = v.z;
      for (int c=0; c<num_attributes; ++c) {
        const int2 vidx = getEdgeVertexIndices(edge);
        const float a0 = sampleAttribute(vidx.x, b, c, grid_pos, volume);
        const float a1 = sampleAttribute(vidx.y, b, c, grid_pos, volume);
        const float a = attribInterp(iso_value, a0, a1, field[vidx.x], field[vidx.y]);
        verts[vert_index][3+c] = a;
      }
    }
  }
}

void launchGenerateTriangles(
    dim3 grid, dim3 threads,
    const at::Tensor& volume,
    at::Tensor& verts,
    uint *voxel_verts_scanned,
    uint num_voxels,float iso_value,
    float3 voxel_start, float3 voxel_size,
    int b) {
  generateTriangles<<<grid, NTHREADS>>>(
    volume.packed_accessor32<float, 5, torch::DefaultPtrTraits>(),
    verts.packed_accessor32<float, 2, torch::DefaultPtrTraits>(),
    voxel_verts_scanned, num_voxels, iso_value, voxel_start, voxel_size,
    numVertsTex, triTex, b);
}

__global__ void gradVolume(
    const at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits> volume,
    const at::PackedTensorAccessor32<float, 2, at::DefaultPtrTraits> verts_grad,
    at::PackedTensorAccessor32<float, 4, at::DefaultPtrTraits> volume_grad,
    uint *voxel_verts_scanned,
    uint num_voxels, float iso_value,
    float3 voxel_start, float3 voxel_size,
    cudaTextureObject_t numVertsTex, cudaTextureObject_t triTex) {
  const uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
  const uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
  const uint num_attributes = volume.size(0) - 1;
  if (i < num_voxels) {
    // compute position in 3d grid
    uint3 grid_pos = calcGridPos(i, volume);

    float3 p = make_float3(
      voxel_start.x + (grid_pos.x * voxel_size.x),
      voxel_start.y + (grid_pos.y * voxel_size.y),
      voxel_start.z + (grid_pos.z * voxel_size.z));

    // calculate cell vertex positions
    float3 v[8];
    v[0] = p;
    v[1] = p + make_float3(voxel_size.x,            0,            0);
    v[2] = p + make_float3(voxel_size.x, voxel_size.y,            0);
    v[3] = p + make_float3(           0, voxel_size.y,            0);
    v[4] = p + make_float3(           0,            0, voxel_size.z);
    v[5] = p + make_float3(voxel_size.x,            0, voxel_size.z);
    v[6] = p + make_float3(voxel_size.x, voxel_size.y, voxel_size.z);
    v[7] = p + make_float3(           0, voxel_size.y, voxel_size.z);

    float field[8];
    sampleField(field, 0, grid_pos, volume);
    const uint cubeindex = computeCubeIndex(field, iso_value);
    const uint numVerts = tex1Dfetch<uint>(numVertsTex, cubeindex);

    float grad[8];
    grad[0] = 0.f; grad[1] = 0.f; grad[2] = 0.f; grad[3] = 0.f;
    grad[4] = 0.f; grad[5] = 0.f; grad[6] = 0.f; grad[7] = 0.f;
    // output triangle vertices
    for (int j = 0; j < numVerts; ++j) {
      const uint vert_index = voxel_verts_scanned[i] + j;
      const uint edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + j);
      // Iterpolated dimension
      int didx = 2;
      if (edge < 8) { 
        didx = edge % 2;
      }
      // Read verts grad
      const float vertd_grad = verts_grad[vert_index][didx];
      // Get indices
      const int2 vidx = getEdgeVertexIndices(edge);
      // We just need p1-p0, so we don't need full v (use voxel_size[didx])
      vertexInterpGrad(
        grad[vidx.x], grad[vidx.y],
        vertd_grad, iso_value,
        reinterpret_cast<float*>(&(v[vidx.x]))[didx],
        reinterpret_cast<float*>(&(v[vidx.y]))[didx],
        field[vidx.x], field[vidx.y]);
      for (int c=0; c<num_attributes; ++c) {
        const float verts_attr_grad = verts_grad[vert_index][3+c];
        const float a0 = sampleAttribute(vidx.x, c, grid_pos, volume);
        const float a1 = sampleAttribute(vidx.y, c, grid_pos, volume);
        vertexInterpGrad(
          grad[vidx.x], grad[vidx.y],
          verts_attr_grad, iso_value,
          a0, a1,
          field[vidx.x], field[vidx.y]);
      }
    }
    writeGrad(grad, 0, grid_pos, volume_grad);
    for (int c=0; c<num_attributes; ++c) {
      grad[0] = 0.f; grad[1] = 0.f; grad[2] = 0.f; grad[3] = 0.f;
      grad[4] = 0.f; grad[5] = 0.f; grad[6] = 0.f; grad[7] = 0.f;
      // output triangle vertices
      for (int j = 0; j < numVerts; ++j) {
        const uint vert_index = voxel_verts_scanned[i] + j;
        // Read attr grad
        const float verts_attr_grad = verts_grad[vert_index][3+c];
        const uint edge = tex1Dfetch<uint>(triTex, (cubeindex * 16) + j);
        // Get indices
        const int2 vidx = getEdgeVertexIndices(edge);
        attribInterpGradAttrib(
          grad[vidx.x], grad[vidx.y],
          verts_attr_grad, iso_value,
          field[vidx.x], field[vidx.y]);
      }
      writeGrad(grad, c+1, grid_pos, volume_grad);
    }
  }
}

void launchGradVolume(
    dim3 grid, dim3 threads,
    const at::Tensor& volume,
    const at::Tensor& verts_grad,
    at::Tensor& volume_grad,
    uint *voxel_verts_scanned,
    uint num_voxels, float iso_value,
    float3 voxel_start, float3 voxel_size) {
  gradVolume<<<grid, NTHREADS>>>(
    volume.packed_accessor32<float, 4, torch::DefaultPtrTraits>(),
    verts_grad.packed_accessor32<float, 2, torch::DefaultPtrTraits>(),
    volume_grad.packed_accessor32<float, 4, torch::DefaultPtrTraits>(),
    voxel_verts_scanned, num_voxels, iso_value, voxel_start, voxel_size,
    numVertsTex, triTex);
}

void thrustExclusiveScan(uint *output, uint *input, uint numElements) {
  thrust::exclusive_scan(thrust::device_ptr<uint>(input),
                         thrust::device_ptr<uint>(input + numElements),
                         thrust::device_ptr<uint>(output));
}
