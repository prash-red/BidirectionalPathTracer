//
// Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <vector_types.h>

// Tiny Encryption Algorithm (TEA) to calculate a the seed per launch index and iteration.
// This results in a ton of integer instructions! Use the smallest N necessary.
template<unsigned int N>
__forceinline__ __device__ unsigned int tea(const unsigned int val0, const unsigned int val1)
{
  unsigned int v0 = val0;
  unsigned int v1 = val1;
  unsigned int s0 = 0;

  for (unsigned int n = 0; n < N; ++n)
  {
    s0 += 0x9e3779b9;
    v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
    v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
  }
  return v0;
}


// Just do one LCG step. The new random unsigned int number is in the referenced argument.
__forceinline__ __device__ void lcg(unsigned int& previous)
{
  previous = previous * 1664525u + 1013904223u;
}

// Return a random sample in the range [0, 1) with a simple Linear Congruential Generator.
// __forceinline__ __device__ float rng(unsigned int& previous)
// {
//   previous = previous * 1664525u + 1013904223u;
//
//   //return float(previous & 0x00FFFFFF) / float(0x01000000u); // Use the lower 24 bits.
//   return float(previous >> 8) / float(0x01000000u); // Use the upper 24 bits
// }


// Convenience function
__forceinline__ __device__ float2 rng2(unsigned int& previous)
{
  float2 s;

  previous = previous * 1664525u + 1013904223u;
  s.x = float(previous >> 8) / float(0x01000000u);

  previous = previous * 1664525u + 1013904223u;
  s.y = float(previous >> 8) / float(0x01000000u);

  return s;
}


// Convenience function
__forceinline__ __device__ float3 rng3(unsigned int& previous)
{
  float3 s;

  previous = previous * 1664525u + 1013904223u;
  s.x = float(previous >> 8) / float(0x01000000u);

  previous = previous * 1664525u + 1013904223u;
  s.y = float(previous >> 8) / float(0x01000000u);

  previous = previous * 1664525u + 1013904223u;
  s.z = float(previous >> 8) / float(0x01000000u);

  return s;
}


// Convenience function
__forceinline__ __device__ float4 rng4(unsigned int& previous)
{
  float4 s;

  previous = previous * 1664525u + 1013904223u;
  s.x = float(previous >> 8) / float(0x01000000u);

  previous = previous * 1664525u + 1013904223u;
  s.y = float(previous >> 8) / float(0x01000000u);

  previous = previous * 1664525u + 1013904223u;
  s.z = float(previous >> 8) / float(0x01000000u);

  previous = previous * 1664525u + 1013904223u;
  s.w = float(previous >> 8) / float(0x01000000u);

  return s;
}


// PCG random number generator (better statistical properties than LCG)
// http://www.pcg-random.org/
__forceinline__ __device__ unsigned int pcg(unsigned int& state)
{
  const unsigned int old = state;
  state = old * 747796405u + 2891336453u;
  const unsigned int word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
  return (word >> 22u) ^ word;
}

__forceinline__ __device__ float rng(unsigned int& state)
{
  return float(pcg(state) >> 8) * (1.0f / float(1u << 24));
}

// PCG float in range [0, 1)
__forceinline__ __device__ float pcg_float(unsigned int& state)
{
  return float(pcg(state) >> 8) * (1.0f / float(1u << 24));
}

// Box-Muller transform for Gaussian noise using PCG
__forceinline__ __device__ float pcg_gaussian(unsigned int& state)
{
  const float u1 = fmaxf(fminf(pcg_float(state), 1.0f - 1e-6f), 1e-6f);
  const float u2 = pcg_float(state);
  return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979f * u2);
}

// Convenience functions for PCG
__forceinline__ __device__ float2 pcg2(unsigned int& state)
{
  return make_float2(pcg_float(state), pcg_float(state));
}

__forceinline__ __device__ float3 pcg3(unsigned int& state)
{
  return make_float3(pcg_float(state), pcg_float(state), pcg_float(state));
}

__forceinline__ __device__ float4 pcg4(unsigned int& state)
{
  return make_float4(pcg_float(state), pcg_float(state), pcg_float(state), pcg_float(state));
}
