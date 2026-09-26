/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "CameraPath.h"
#include "Camera.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace dev
{
  void CameraPath::clear()
  {
    m_poses.clear();
  }

  void CameraPath::append(float time, const Camera& camera)
  {
    // Poses must be strictly increasing in time for the bracketing search.
    if (!m_poses.empty() && time <= m_poses.back().time)
    {
      return;
    }
    m_poses.push_back({ time, camera.getPosition(), camera.getLookat(), camera.getUp(), camera.getFovY() });
  }

  bool CameraPath::save(const std::string& filename, int width, int height) const
  {
    std::ofstream file(filename);
    if (!file)
    {
      std::cerr << "ERROR: CameraPath::save() cannot open " << filename << '\n';
      return false;
    }
    file << "# GLTF_renderer camera path: time px py pz lx ly lz ux uy uz fovY\n";
    file << "resolution " << width << ' ' << height << '\n';
    file << std::setprecision(9);
    for (const CameraPose& p : m_poses)
    {
      file << p.time << ' '
           << p.position.x << ' ' << p.position.y << ' ' << p.position.z << ' '
           << p.lookat.x   << ' ' << p.lookat.y   << ' ' << p.lookat.z   << ' '
           << p.up.x       << ' ' << p.up.y       << ' ' << p.up.z       << ' '
           << p.fovY << '\n';
    }
    return static_cast<bool>(file);
  }

  bool CameraPath::load(const std::string& filename, int& width, int& height)
  {
    width  = 0;
    height = 0;

    std::ifstream file(filename);
    if (!file)
    {
      std::cerr << "ERROR: CameraPath::load() cannot open " << filename << '\n';
      return false;
    }

    std::vector<CameraPose> poses;
    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line))
    {
      ++lineNumber;
      if (line.empty() || line[0] == '#')
      {
        continue;
      }
      std::istringstream in(line);
      if (line.rfind("resolution", 0) == 0)
      {
        std::string keyword;
        in >> keyword >> width >> height;
        continue;
      }
      CameraPose p;
      in >> p.time
         >> p.position.x >> p.position.y >> p.position.z
         >> p.lookat.x   >> p.lookat.y   >> p.lookat.z
         >> p.up.x       >> p.up.y       >> p.up.z
         >> p.fovY;
      if (!in || (!poses.empty() && p.time <= poses.back().time))
      {
        std::cerr << "ERROR: CameraPath::load() " << filename << ':' << lineNumber << ": malformed or out-of-order pose\n";
        return false;
      }
      poses.push_back(p);
    }

    if (poses.empty())
    {
      std::cerr << "ERROR: CameraPath::load() " << filename << " contains no poses\n";
      return false;
    }
    m_poses = std::move(poses);
    return true;
  }

  void CameraPath::makeOrbit(const Camera& start, float seconds, float degrees, int samplesPerSecond)
  {
    m_poses.clear();

    const glm::vec3 lookat = start.getLookat();
    const glm::vec3 up     = start.getUp();
    const glm::vec3 axis   = glm::normalize(up);
    const glm::vec3 offset = start.getPosition() - lookat;
    const float     angle  = glm::radians(degrees);
    const int       count  = std::max(2, static_cast<int>(seconds * samplesPerSecond + 0.5f) + 1);

    for (int i = 0; i < count; ++i)
    {
      const float s     = static_cast<float>(i) / static_cast<float>(count - 1);
      const float theta = angle * s;
      const float c     = cosf(theta);
      const float n     = sinf(theta);
      // Rodrigues' rotation of the offset about the up axis.
      const glm::vec3 rotated = offset * c + glm::cross(axis, offset) * n + axis * (glm::dot(axis, offset) * (1.0f - c));

      m_poses.push_back({ seconds * s, lookat + rotated, lookat, up, start.getFovY() });
    }
  }

  void CameraPath::makeCircle(const Camera& start, float seconds, float radiusFraction, int samplesPerSecond)
  {
    m_poses.clear();

    const glm::vec3 position = start.getPosition();
    const glm::vec3 lookat   = start.getLookat();
    const glm::vec3 up       = start.getUp();
    // View-plane basis: right and the true (view-orthogonal) up.
    const glm::vec3 w      = glm::normalize(lookat - position);
    const glm::vec3 u      = glm::normalize(glm::cross(w, up));
    const glm::vec3 v      = glm::cross(u, w);
    const float     radius = radiusFraction * glm::length(lookat - position);
    const int       count  = std::max(2, static_cast<int>(seconds * samplesPerSecond + 0.5f) + 1);

    // The start pose is the centre. Jumping out to the rim would be a cut in the
    // video, so the radius eases in over the first RAMP of the duration and back
    // out over the last, while the angle turns uniformly throughout: the camera
    // leaves and returns to the centre at rest, and a loop repeats seamlessly.
    const float RAMP = 0.15f;
    auto ease = [](float x) { x = std::min(std::max(x, 0.0f), 1.0f); return x * x * (3.0f - 2.0f * x); };

    for (int i = 0; i < count; ++i)
    {
      const float s     = static_cast<float>(i) / static_cast<float>(count - 1);
      const float theta = 2.0f * glm::pi<float>() * s;
      const float r     = radius * std::min(ease(s / RAMP), ease((1.0f - s) / RAMP));

      m_poses.push_back({ seconds * s, position + (u * cosf(theta) + v * sinf(theta)) * r, lookat, up, start.getFovY() });
    }
  }

  bool CameraPath::empty() const
  {
    return m_poses.empty();
  }

  size_t CameraPath::size() const
  {
    return m_poses.size();
  }

  float CameraPath::duration() const
  {
    return m_poses.empty() ? 0.0f : m_poses.back().time;
  }

  void CameraPath::apply(float time, Camera& camera) const
  {
    if (m_poses.empty())
    {
      return;
    }

    CameraPose pose;
    if (time <= m_poses.front().time)
    {
      pose = m_poses.front();
    }
    else if (m_poses.back().time <= time)
    {
      pose = m_poses.back();
    }
    else
    {
      // First pose strictly later than time; its predecessor brackets from below.
      const auto next = std::upper_bound(m_poses.begin(), m_poses.end(), time,
                                         [](float t, const CameraPose& p) { return t < p.time; });
      const CameraPose& b = *next;
      const CameraPose& a = *(next - 1);
      const float s = (time - a.time) / (b.time - a.time);

      // Poses are recorded every frame, so segments are short enough that
      // linear interpolation is indistinguishable from a spline.
      pose.position = glm::mix(a.position, b.position, s);
      pose.lookat   = glm::mix(a.lookat,   b.lookat,   s);
      pose.up       = glm::mix(a.up,       b.up,       s);
      pose.fovY     = a.fovY + (b.fovY - a.fovY) * s;
    }

    // A held pose (the -play-delay segment, or a pause in the recording) must not
    // touch the camera: setting it marks it dirty, which restarts accumulation
    // every frame and would stop the hold from ever converging.
    if (pose.position == camera.getPosition() && pose.lookat == camera.getLookat() &&
        pose.up == camera.getUp() && pose.fovY == camera.getFovY())
    {
      return;
    }

    camera.setPosition(pose.position);
    camera.setLookat(pose.lookat);
    camera.setUp(pose.up);
    camera.setFovY(pose.fovY);
    camera.setIsDirty(true);
  }

} // namespace dev
