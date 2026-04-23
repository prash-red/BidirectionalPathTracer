 /* 
 * Copyright (c) 2013-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "config.h"

#include <optix.h>

#include "launch_parameters.h"
#include "light_definition.h"
#include "per_ray_data.h"
#include "random.h"
#include "shader_common.h"
#include "transform.h"

extern "C" __constant__ LaunchParameters theLaunchParameters;


// Note that all light sampling routines return lightSample.direction and lightSample.distance in world space!

extern "C" __device__ LightSample __direct_callable__light_env_constant(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  const float2 sample = rng2(prd->seed);

  unitSquareToSphere(sample.x, sample.y, lightSample.direction, lightSample.pdf);

  // The emission is constant in all directions.
  // There is no transformation of the object space direction into world space necessary.

  lightSample.distance = RT_DEFAULT_MAX; // Environment light.
  
  // Potential ambient occlusion is only applied for environment lights.
  lightSample.radiance_over_pdf = light.emission * (prd->occlusion / lightSample.pdf);

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_env_sphere(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  lightSample.pdf = 0.0f;

  // Importance-sample the spherical environment light direction in object space.
  // FIXME The binary searches are generating a lot of memory traffic. Replace this with an alias-map lookup.
  const float2 sample = rng2(prd->seed);

  // Note that the marginal CDF is one bigger than the texture height. As index this is the 1.0f at the end of the CDF.
  const float* cdfV = reinterpret_cast<const float*>(light.cdfV);
  const unsigned int idxV = binarySearchCDF(cdfV, light.height, sample.y);

  const float* cdfU = reinterpret_cast<const float*>(light.cdfU);
  cdfU += (light.width + 1) * idxV; // Horizontal CDF is one bigger than the texture width!
  const unsigned int idxU = binarySearchCDF(cdfU, light.width, sample.x);

  // Continuous sampling of the CDF.
  const float cdfLowerU = cdfU[idxU];
  const float cdfUpperU = cdfU[idxU + 1];
  const float du = (sample.x - cdfLowerU) / (cdfUpperU - cdfLowerU);
  const float u = (float(idxU) + du) / float(light.width);

  const float cdfLowerV = cdfV[idxV];
  const float cdfUpperV = cdfV[idxV + 1];
  const float dv = (sample.y - cdfLowerV) / (cdfUpperV - cdfLowerV);
  const float v = (float(idxV) + dv) / float(light.height);

  // Light sample direction vector in object space polar coordinates.
  const float phi   = u * M_PIf * 2.0f;
  const float theta = v * M_PIf; // theta == 0.0f is south pole, theta == M_PIf is north pole.

  const float sinTheta = sinf(theta);

  // All lights shine down the positive z-axis in this renderer.
  // Orient the 2D texture map so that the center (u, v) = (0.5, 0.5) lies exactly on the positive z-axis.
  // Means the seam from u == 1.0 -> 0.0 lies on the negative z-axis and the u range [0.0, 1.0]
  // goes clockwise on the xz-plane when looking from the positive y-axis.
  const float3 dir = make_float3( sinf(phi) * sinTheta,  // Starting on negative z-axis going around clockwise (to positive x-axis).
                                 -cosf(theta),           // From south pole to north pole.
                                 -cosf(phi) * sinTheta); // Starting on negative z-axis.
  
  // Now rotate that normalized object space direction into world space. 
  // This is only using the upper 3x3 matrix which contains a rotation for spherical environment lights.
  lightSample.direction = transformVector(light.matrix, dir);

  lightSample.distance = RT_DEFAULT_MAX; // Environment light.
  
  // Get the emission from the spherical environment texture.
  const float3 emission = make_float3(tex2D<float4>(light.textureEmission, u, v));
  
  // For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
  // and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
  // (Note that this does not contain the light.emission which just modulates the texture.)
  lightSample.pdf = intensity(emission) * light.invIntegral;

  if (DENOMINATOR_EPSILON < lightSample.pdf)
  {
    // Potential ambient occlusion is only applied for environment lights.
    lightSample.radiance_over_pdf = light.emission * emission * (prd->occlusion / lightSample.pdf);
  }

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_point(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  // Default return, invalid light sample (Too near to the surface or outside light's range.)
  lightSample.pdf = 0.0f; 

  // Get the world space position from the object to world matrix translation.
  const float3 position = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);

  lightSample.direction = position - prd->pos; // Sample direction from surface point to light sample position.
  
  const float distanceSquared = dot(lightSample.direction, lightSample.direction);

  if (DENOMINATOR_EPSILON < distanceSquared)
  {
    lightSample.distance = sqrtf(distanceSquared);

    // GLTF can limit the point light influence to a range.
    // Do not return a valid light sample when the range is exceeded-
    if (lightSample.distance < light.range)
    {
      lightSample.direction *= 1.0f / lightSample.distance; // Normalized direction to light.

      lightSample.radiance_over_pdf = light.emission * (1.0f / distanceSquared); // Quadratic attenuation. pdf == 1.0f for singular light.

      // Indicate valid light sample (pdf != 0.0f).
      // This value is otherwise unused in a singular light. 
      // The PDF is a Dirac with infinite value for this case.
      lightSample.pdf = 1.0f;
    }
  }

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__light_spot(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  // Default return, invalid light sample 
  // Too near to the surface or outside light's range or outer cone angle.
  lightSample.pdf = 0.0f;

  // Get the world space position from the world to object matrix translation.
  const float3 position = make_float3(light.matrix[0].w, light.matrix[1].w, light.matrix[2].w);
  lightSample.direction = position - prd->pos; // Sample direction from surface point to light sample position.
  
  const float distanceSquared = dot(lightSample.direction, lightSample.direction);

  if (DENOMINATOR_EPSILON < distanceSquared)
  {
    lightSample.distance = sqrtf(distanceSquared);
    if (lightSample.distance < light.range)
    {
      lightSample.direction *= 1.0f / lightSample.distance; // Normalized direction to light.

      // The GLTF lights shine down the negative z-axis by default. Use that as light normal.
      //const float3 normalLight = normalize(transformNormal(light.matrixInv, make_float3(0.0f, 0.0f, -1.0f)));
      const float3 normalLight = normalize(-make_float3(light.matrixInv[2])); // optimized

      // Negative because lightSample.direction is from surface to light.
      const float cosTheta = -dot(lightSample.direction, normalLight); 
      if (light.cosOuter < cosTheta) // lightSample.direction inside the outer cone angle?
      {
        // Calculate a linear falloff over the cosines from inner (falloff == 1.0f) to outer (falloff == 0.0f) cone angles.
        const float cosRange = fmaxf(DENOMINATOR_EPSILON, light.cosInner - light.cosOuter);
        const float falloff  = fminf((cosTheta - light.cosOuter) / cosRange, 1.0f); 

        lightSample.radiance_over_pdf = light.emission * (falloff / distanceSquared);
        lightSample.pdf = 1.0f;
      }
    }
  }
  return lightSample;
}

extern "C" __device__ LightSample __direct_callable__light_directional(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;

  // The GLTF lights shine down the negative z-axis by default. Use that as light normal.
  //const float3 normalLight = normalize(transformNormal(light.matrixInv, make_float3(0.0f, 0.0f, -1.0f)));
  // Negated to get light sample direction from surface point to light!
  lightSample.direction = normalize(make_float3(light.matrixInv[2])); // optimized
  lightSample.distance = RT_DEFAULT_MAX; // Not a positional light.
  // The directional light units is illuminance in lux, lumens per meter squared (lm/m^2)
  // light.area contains the scene's maximum radius disk area. Multiply by that to get lumens.
  lightSample.radiance_over_pdf = light.emission * light.area;
  lightSample.pdf = 1.0f;

  return lightSample;
}


extern "C" __device__ LightSample __direct_callable__sampleMeshLight(const LightDefinition& light, PerRayData* prd)
{
  LightSample lightSample;
  lightSample.pdf = 0.0f; // default: invalid sample

  const float3* positions = reinterpret_cast<const float3*>(light.cdfU); // world-space, 3N entries
  const float*  cdf       = reinterpret_cast<const float*> (light.cdfV); // N+1 entries
  const int     N         = static_cast<int>(light.width);                // triangle count

  if (N <= 0 || light.area <= 0.0f)
    return lightSample;

  // Pick a triangle by sampling the normalized area CDF.
  const float        u0     = rng(prd->seed);
  const unsigned int triIdx = binarySearchCDF(cdf, static_cast<unsigned int>(N), u0);

  // Uniform sample on the chosen triangle (Turk 1990).
  const float u1 = rng(prd->seed);
  const float u2 = rng(prd->seed);
  const float su = sqrtf(u1);
  const float b0 = 1.0f - su;
  const float b1 = su * (1.0f - u2);
  const float b2 = su * u2;

  const float3 v0 = positions[3 * triIdx + 0];
  const float3 v1 = positions[3 * triIdx + 1];
  const float3 v2 = positions[3 * triIdx + 2];

  const float3 p_world = b0 * v0 + b1 * v1 + b2 * v2;

  // Face normal from world-space edges (positions already in world space).
  const float3 e0 = v1 - v0;
  const float3 e1 = v2 - v0;
  const float3 cr = cross(e0, e1);

  if (dot(cr, cr) <= DENOMINATOR_EPSILON)
    return lightSample; // Degenerate triangle; skip.

  const float3 n_world = normalize(cr);

  const float3 toLight = p_world - prd->pos;
  const float  dist2   = dot(toLight, toLight);

  if (dist2 <= DENOMINATOR_EPSILON)
    return lightSample; // Shading point on light surface; avoid division by zero.

  const float  dist = sqrtf(dist2);
  const float3 dir  = toLight / dist;

  const float cosTheta = dot(-dir, n_world);
  if (cosTheta <= 0.0f)
    return lightSample; // Back-facing sample; caller checks pdf <= 0.

  // Solid-angle PDF: (1 / total_area) * dist^2 / cosTheta
  lightSample.direction         = dir;
  lightSample.distance          = dist;
  lightSample.pdf               = dist2 / (cosTheta * light.area);
  lightSample.radiance_over_pdf = light.emission / lightSample.pdf;
  return lightSample;
}

