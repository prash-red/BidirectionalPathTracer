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


#include "Options.h"
#include <GLFW/glfw3.h> // dpi 

#include <algorithm>
#include <iostream>

static int toInt(float x)
{
  return static_cast<int>(x);
}

// public:

Options::Options()
  : m_widthClient(toInt(512 * getViewportScale()))
  , m_heightClient(toInt(512 * getViewportScale()))
  , m_widthResolution(toInt(512 * getViewportScale()))
  , m_heightResolution(toInt(512 * getViewportScale()))
  , m_launches(1)
  , m_interop(0)
  , m_miss(2)
  , m_punctual(true)
  , m_nrc(0)
  , m_saveAfterFrames(0)
  , m_playDelay(0.0f)
  , m_playLoop(false)
  , m_animate(false)
  , m_motion(0)
  , m_motionSeconds(10.0f)
  , m_orbitDegrees(360.0f)
  , m_circleRadius(0.1f)
  , m_sphereRadiusFraction(0.005f)
{
}

//Options::~Options()
//{
//}

bool Options::parseCommandLine(int argc, char *argv[])
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);

    if (arg == "?" || arg == "help" || arg == "--help")
    {
      printUsage(std::string(argv[0])); // Application name.
      return false;
    }
    else if (arg == "-f" || arg == "--file")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_filename = std::filesystem::path(argv[++i]);
    }
    else if (arg == "-w" || arg == "--width")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_widthClient = atoi(argv[++i]);
    }
    else if (arg == "-h" || arg == "--height")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_heightClient = atoi(argv[++i]);
    }
    else if (arg == "-x" || arg == "--xres")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_widthResolution = atoi(argv[++i]);
    }
    else if (arg == "-y" || arg == "--yres")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_heightResolution = atoi(argv[++i]);
    }
    else if (arg == "-l" || arg == "--launches")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_launches = atoi(argv[++i]);
    }
    else if (arg == "-i" || arg == "--interop")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_interop = atoi(argv[++i]);
    }
    else if (arg == "-p" || arg == "--punctual")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_punctual = (atoi(argv[++i]) != 0);
    }
    else if (arg == "-save-after")
    {
      if (i + 2 >= argc)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires <frames> <file.pfm>.\n";
        printUsage(argv[0]);
        return false;
      }
      m_saveAfterFrames = atoi(argv[++i]);
      m_saveAfterPath   = argv[++i];
      if (m_saveAfterFrames <= 0)
      {
        std::cerr << "ERROR: Option '" << arg << "' needs a positive frame count.\n";
        printUsage(argv[0]);
        return false;
      }
    }
    else if (arg == "-record-camera" || arg == "-play-camera" || arg == "-play-delay")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      if (arg == "-record-camera")
        m_recordCameraPath = argv[++i];
      else if (arg == "-play-camera")
        m_playCameraPath = argv[++i];
      else
        m_playDelay = std::max(0.0f, static_cast<float>(atof(argv[++i])));
    }
    else if (arg == "-play-orbit" || arg == "-play-circle")
    {
      if (i + 2 >= argc)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires <seconds> <" << (arg == "-play-orbit" ? "degrees" : "radius") << ">.\n";
        printUsage(argv[0]);
        return false;
      }
      m_motionSeconds = static_cast<float>(atof(argv[++i]));
      const float value = static_cast<float>(atof(argv[++i]));
      if (m_motionSeconds <= 0.0f || (arg == "-play-circle" && value <= 0.0f))
      {
        std::cerr << "ERROR: Option '" << arg << "' needs a positive duration" << (arg == "-play-circle" ? " and radius" : "") << ".\n";
        printUsage(argv[0]);
        return false;
      }
      if (arg == "-play-orbit")
      {
        m_motion       = 1;
        m_orbitDegrees = value;
      }
      else
      {
        m_motion       = 2;
        m_circleRadius = value;
      }
    }
    else if (arg == "-play-loop")
    {
      m_playLoop = true;
    }
    else if (arg == "-animate")
    {
      m_animate = true;
    }
    else if (arg == "-nrc")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_nrc = atoi(argv[++i]);
      if (m_nrc < 0 || m_nrc > 3) {
        std::cerr << "ERROR: Option '" << arg << "'\n";
        printUsage(argv[0]);
        return false;
      }
    }
    else if (arg == "-m" || arg == "--miss")
    {
      if (i == argc - 1)
      { 
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_miss = atoi(argv[++i]);
    }
    else if (arg == "-e" || arg == "--env")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_environment = std::string(argv[++i]);
    }
    else if (arg == "-r" || arg == "--radius")
    {
      if (i == argc - 1)
      {
        std::cerr << "ERROR: Option '" << arg << "' requires additional argument.\n";
        printUsage(argv[0]);
        return false;
      }
      m_sphereRadiusFraction = static_cast<float>(std::atof(argv[++i]));
    }
    else
    {
      std::cerr << "ERROR: Unknown option '" << arg << "'\n";
      printUsage(argv[0]);
      return false;
    }
  }

  // Validate all input values.
  bool isInvalidArgument = false;

  if (m_filename.empty())
  {
    std::cerr << "WARNING: No --file (-f) argument provided, starting with empty scene.\n(Use drag-and-drop of *.gltf or *.glb files to load asset.)\n";
  }
  
  if (m_widthClient <= 0)
  {
    std::cerr << "ERROR: Invalid --width (-w) argument (must be > 0).\n";
    isInvalidArgument = true;
  }
  
  if (m_heightClient <= 0)
  {
    std::cerr << "ERROR: Invalid --height (-h) argument (must be > 0).\n";
    isInvalidArgument = true;
  }

  if (m_widthResolution <= 0)
  {
    std::cerr << "ERROR: Invalid --xres (-x) argument (must be > 0).\n";
    isInvalidArgument = true;
  }
  
  if (m_heightResolution <= 0)
  {
    std::cerr << "ERROR: Invalid --yres (-y) argument (must be > 0).\n";
    isInvalidArgument = true;
  }

  if (m_launches < 1)
  {
    std::cerr << "WARNING: --launches (-l) needs to be greater than 0.\n";
    m_launches = 1; 
  }
  else if (MAX_LAUNCHES < m_launches)
  {
    std::cerr << "WARNING: --launches (-l) needs to be less than or equal to 1000.\n";
    m_launches = MAX_LAUNCHES;
  }

  if (m_interop < 0 || 3 < m_interop)
  {
    std::cerr << "ERROR: Invalid --interop (-i) argument (must be 0 to 3).\n";
    isInvalidArgument = true;
  }

  if (m_miss < 0 || 2 < m_miss)
  {
    std::cerr << "ERROR: Invalid --miss (-m) program argument (must be 0 to 2).\n";
    isInvalidArgument = true;
  }

  if (isInvalidArgument)
  {
    printUsage(argv[0]);
    return false;
  }

  return true;
}


std::filesystem::path Options::getFilename() const
{
  return m_filename;
}

int Options::getWidthClient() const
{
  return m_widthClient;
}

int Options::getHeightClient() const
{
  return m_heightClient;
}

int Options::getWidthResolution() const
{
  return m_widthResolution;
}

int Options::getHeightResolution() const
{
  return m_heightResolution;
}

int Options::getLaunches() const
{
  return m_launches;
}

int Options::getInterop() const
{
  return m_interop;
}

int Options::getMiss() const
{
  return m_miss;
}

std::string Options::getEnvironment() const
{
  return m_environment;
}

bool Options::getPunctual() const
{
  return m_punctual;
}

int Options::getSaveAfterFrames() const
{
  return m_saveAfterFrames;
}

std::string Options::getSaveAfterPath() const
{
  return m_saveAfterPath;
}

std::string Options::getRecordCameraPath() const
{
  return m_recordCameraPath;
}

std::string Options::getPlayCameraPath() const
{
  return m_playCameraPath;
}

float Options::getPlayDelay() const
{
  return m_playDelay;
}

bool Options::getPlayLoop() const
{
  return m_playLoop;
}

bool Options::getAnimate() const
{
  return m_animate;
}

int Options::getMotion() const
{
  return m_motion;
}

float Options::getMotionSeconds() const
{
  return m_motionSeconds;
}

float Options::getOrbitDegrees() const
{
  return m_orbitDegrees;
}

float Options::getCircleRadius() const
{
  return m_circleRadius;
}

int Options::getNrc() const
{
  return m_nrc;
}

float Options::getSphereRadiusFraction() const
{
  return m_sphereRadiusFraction;
}

// private:

void Options::printUsage(std::string const& argv0)
{
  std::cerr << "\nUsage: " << argv0 << " [options]\n";
  std::cerr <<
    "Options:\n"
    "   ? | --help | help     Print this usage message and exit.\n"
    "  -f | --file <filename> Filename of a glTF model (required). (empty)\n"
    "  -w | --width <int>     Client window width.  (512)\n"
    "  -h | --height <int>    Client window height. (512)\n"
    "  -x | --xres <int>      Render resolution width.  (512)\n"
    "  -y | --yres <int>      Render resolution height. (512)\n"
    "  -l | --launches <int>  Number of launches per render call, range [1, 1000] (1)\n"
    "  -i | --interop <int>   OpenGL interop: 0 = off, 1 = pbo, 2 = array copy, 3 = surface write. (0)\n"
    "  -p | --punctual <int>  Select KHR_lights_punctual support: 0 = off. (1)\n"
    "  -nrc                   Render mode: 0 = path tracing, 1 = nrc, 2 = bdrc (unified cache),\n"
    "                         3 = SPPM reference. (0)\n"
    "  -save-after <n> <file> Once n iterations have accumulated (since the last restart),\n"
    "                         write the linear image to <file> (PFM) and exit.\n"
    "  -record-camera <file>  Record the camera pose every frame from launch; written to <file>\n"
    "                         when recording stops (R key) or on exit.\n"
    "  -play-camera <file>    Replay a recorded camera path by wall-clock time, then exit.\n"
    "  -play-delay <seconds>  Hold the first pose this long before the path starts. (0)\n"
    "  -play-orbit <s> <deg>  Orbit the look-at point by <deg> degrees over <s> seconds,\n"
    "                         starting from the initial camera.\n"
    "  -play-circle <s> <r>   Move the camera once around a circle in its view plane over <s>\n"
    "                         seconds, radius <r> x distance to the look-at point, still aimed at it.\n"
    "  -play-loop             Repeat the camera path instead of exiting at its end.\n"
    "  -animate               Enable and play all glTF animations from the first rendered frame,\n"
    "                         on the same wall clock as the camera path.\n"
    "  -m | --miss <0|1|2>    Select the miss shader: 0 = null, 1 = white, 2 = environment. (2))\n"
    "  -e | --env <filename>  Filename of a spherical HDR texture. (empty)\n"
    "  -r | --radius <fraction>  Fraction of the scene diameter. (0.005)\n"
    "Viewport Interactions:\n"
    "  SPACE       Toggle GUI display.\n"
    "  P           Save image as tonemapped *.png.\n"
    "  H           Save image as linear *.hdr.\n"
    "  R           Start/stop recording the camera path.\n"
    "  LMB         Orbit\n"
    "  MMB         Pan\n"
    "  RMB         Dolly\n"
    "  MouseWheel  Zoom\n"
    "  Ctrl+LMB    Select material.\n";
}

float Options::getViewportScale() const
{
  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  float xscale, yscale;
  glfwGetMonitorContentScale(monitor, &xscale, &yscale);

  return xscale;
}