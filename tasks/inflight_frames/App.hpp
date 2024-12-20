#pragma once

#include <etna/GlobalContext.hpp>
#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/Image.hpp>
#include <etna/BlockingTransferHelper.hpp>
#include <etna/Sampler.hpp>
#include <etna/GraphicsPipeline.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;
  std::vector<etna::Buffer> paramsBuffers;
  int frame = 0;

  glm::uvec2 resolution;
  bool useVsync;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;

  etna::ComputePipeline pipeline;
  etna::Image image;
  etna::Image imageShader;

  etna::Buffer paramsBuffer;
  etna::Sampler sampler;
  etna::GraphicsPipeline mainPipeline;
  glm::vec2 mouse_pos;
  std::chrono::time_point<std::chrono::system_clock> init_time = std::chrono::system_clock::now();

  struct Params
  {
    glm::vec2 resolution;
    glm::vec2 mouse_pos;
    float time;
  };
};