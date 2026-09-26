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

#pragma once

#ifndef CAMERA_PATH_H
#define CAMERA_PATH_H

#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace dev
{
  class Camera;

  // A recorded camera flight: the camera STATE sampled against wall-clock time,
  // not the mouse events that produced it. The trackball applies mouse deltas
  // once per frame, so replaying events in a mode with another frame rate would
  // fly a different path; replaying poses by time gives every render mode the
  // same camera at the same second, so screen recordings line up.
  struct CameraPose
  {
    float     time;     // Seconds since the recording started.
    glm::vec3 position;
    glm::vec3 lookat;
    glm::vec3 up;
    float     fovY;
  };

  class CameraPath
  {
  public:
    void clear();
    void append(float time, const Camera& camera);

    bool save(const std::string& filename, int width, int height) const;
    // Returns false on a missing or malformed file. width/height receive the
    // recorded resolution (0 when the header has none).
    bool load(const std::string& filename, int& width, int& height);

    bool  empty() const;
    size_t size() const;
    float duration() const; // Time of the last pose.

    // Procedural paths starting exactly at the camera's current pose, sampled at
    // samplesPerSecond so apply() interpolates short chords. Both keep the
    // camera's look-at point, up vector and fovY fixed.
    //
    // Orbit: position rotates by `degrees` about the up axis through the
    // look-at point, at constant distance.
    void makeOrbit(const Camera& start, float seconds, float degrees, int samplesPerSecond = 120);
    // Circle: position travels a circle of radius `radiusFraction` x the
    // distance to the look-at point, in the plane perpendicular to the view
    // direction, CENTRED on the start pose; the radius eases in and out over the
    // first and last 15% so the path starts and ends at the centre. The camera
    // stays aimed at the look-at point, so the object sways in frame rather
    // than revolving.
    void makeCircle(const Camera& start, float seconds, float radiusFraction, int samplesPerSecond = 120);

    // Linear interpolation between the two poses bracketing time; clamps to the
    // first/last pose outside the recorded interval. Leaves the camera untouched
    // (not dirty) when the pose equals its current state.
    void apply(float time, Camera& camera) const;

  private:
    std::vector<CameraPose> m_poses;
  };

} // namespace dev

#endif // CAMERA_PATH_H
