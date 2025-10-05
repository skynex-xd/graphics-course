#include "Renderer.hpp"
#include "WorldRenderer.hpp" //!!!
#include <spdlog/spdlog.h>
#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <imgui.h>

#include <gui/ImGuiRenderer.hpp>
#include <imgui.h>


Renderer::Renderer(glm::uvec2 res)
  : resolution{res}
{
}

void Renderer::initVulkan(std::span<const char*> instance_extensions)
{
  std::vector<const char*> instanceExtensions;

  for (auto ext : instance_extensions)
    instanceExtensions.push_back(ext);

  std::vector<const char*> deviceExtensions;

  deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  vk::PhysicalDeviceFeatures devFeatures{};
  devFeatures.largePoints = VK_TRUE;

  etna::initialize(etna::InitParams{
    .applicationName = "ShadowmapSample",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .instanceExtensions = instanceExtensions,
    .deviceExtensions = deviceExtensions,
    .features = vk::PhysicalDeviceFeatures2{ .features = devFeatures },
    .physicalDeviceIndexOverride = {},
    .numFramesInFlight = 2,
  });
}

void Renderer::initFrameDelivery(vk::UniqueSurfaceKHR a_surface, ResolutionProvider res_provider)
{
  auto& ctx = etna::get_context();

  resolutionProvider = std::move(res_provider);
  commandManager = ctx.createPerFrameCmdMgr();

  window = ctx.createWindow(etna::Window::CreateInfo{
    .surface = std::move(a_surface),
  });

  auto [w, h] = window->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {resolution.x, resolution.y},
    .vsync = useVsync_,
  });
  resolution = {w, h};

  worldRenderer = std::make_unique<WorldRenderer>();

  worldRenderer->allocateResources(resolution);
  worldRenderer->loadShaders();
  worldRenderer->setupPipelines(window->getCurrentFormat());

  guiRenderer = std::make_unique<ImGuiRenderer>(window->getCurrentFormat());
}

void Renderer::recreateSwapchain(glm::uvec2 res)
{
  auto& ctx = etna::get_context();

  ETNA_CHECK_VK_RESULT(ctx.getDevice().waitIdle());

  auto [w, h] = window->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {res.x, res.y},
    .vsync = useVsync_,
  });
  resolution = {w, h};
  worldRenderer->allocateResources(resolution);
  worldRenderer->setupPipelines(window->getCurrentFormat());
}

void Renderer::loadScene(std::filesystem::path path)
{
  worldRenderer->loadScene(path);
}

void Renderer::reloadShaders()
{
  const int retval = std::system("cd " GRAPHICS_COURSE_ROOT "/build"
                                 " && cmake --build . --target shadowmap_shaders");
  if (retval != 0) {
    spdlog::warn("Shader recompilation returned a non-zero return code!");
  } else {
    ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
    etna::reload_shaders();
    spdlog::info("Successfully reloaded shaders!");
  }
}

void Renderer::debugInput(const Keyboard& kb)
{
  worldRenderer->debugInput(kb);

  if (kb[KeyboardKey::kB] == ButtonState::Falling)
    reloadShaders();
}

void Renderer::update(const FramePacket& packet)
{
  lastPacket_ = packet;
  worldRenderer->update(packet);
}

void Renderer::drawFrame()
{
  ZoneScoped;

  {
    ZoneScopedN("drawGui");
    guiRenderer->nextFrame();
    ImGui::NewFrame();
    drawGui();
    ImGui::Render();
  }

  auto currentCmdBuf = commandManager->acquireNext();
  etna::begin_frame();

  auto nextSwapchainImage = window->acquireNext();
  if (nextSwapchainImage)
  {
    auto [image, view, availableSem] = *nextSwapchainImage;

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));
    {
      ETNA_PROFILE_GPU(currentCmdBuf, renderFrame);

      worldRenderer->renderWorld(currentCmdBuf, image, view);

      {
        ImDrawData* pDrawData = ImGui::GetDrawData();
        guiRenderer->render(
          currentCmdBuf, {{0, 0}, {resolution.x, resolution.y}}, image, view, pDrawData);
      }

      etna::set_state(
        currentCmdBuf,
        image,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        {},
        vk::ImageLayout::ePresentSrcKHR,
        vk::ImageAspectFlagBits::eColor);

      etna::flush_barriers(currentCmdBuf);

      ETNA_READ_BACK_GPU_PROFILING(currentCmdBuf);
    }
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    auto renderingDone = commandManager->submit(std::move(currentCmdBuf), std::move(availableSem));

    const bool presented = window->present(std::move(renderingDone), view);

    if (!presented)
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  if (wantRecreateSwapchain_) {
    wantRecreateSwapchain_ = false;
    recreateSwapchain(resolution);
  }

  if (!nextSwapchainImage)
  {
    auto res = resolutionProvider();
    if (res.x != 0 && res.y != 0)
      recreateSwapchain(res);
  }
}

void Renderer::drawGui()
{
  ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(420, 620), ImGuiCond_FirstUseEver);
  ImGui::Begin("Particles", nullptr, ImGuiWindowFlags_None);
  ImGuiIO& io = ImGui::GetIO();
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
              1000.0f / (io.Framerate > 0 ? io.Framerate : 1.0f), io.Framerate);
  auto& emitters = worldRenderer->emitters();
  size_t totalAlive = 0;
  for (auto& e : emitters) totalAlive += e.particles.size();
  ImGui::Text("Particles alive: %zu", totalAlive);
  static float sGlobalAlpha = 1.0f;
  ImGui::SeparatorText("Global");
  ImGui::SliderFloat("Global alpha", &sGlobalAlpha, 0.0f, 1.0f);
  worldRenderer->setGlobalAlpha(sGlobalAlpha);

  ImGui::SeparatorText("Emitters");
  if (ImGui::Button("Add emitter")) {
    WorldRenderer::Emitter e;
    emitters.push_back(e);
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear all")) {
    emitters.clear();
  }

  for (int i = 0; i < (int)emitters.size(); ++i) {
    auto& e = emitters[i];
    if (ImGui::TreeNode((std::string("Emitter ") + std::to_string(i)).c_str())) {
      ImGui::Checkbox("Enabled", &e.enabled);
      ImGui::DragFloat3("Position", &e.position.x, 0.05f);
      ImGui::SliderFloat("Spawn rate", &e.spawnRate, 0.f, 2000.f);
      ImGui::DragFloat2("Life (min,max)", &e.lifeRange.x, 0.01f, 0.05f, 10.f);
      ImGui::DragFloat2("Speed (min,max)", &e.speedRange.x, 0.01f, 0.0f, 20.f);
      ImGui::DragFloat2("Size (min,max)", &e.sizeRange.x, 0.1f, 1.f, 128.f);
      ImGui::SliderFloat("Spread (deg)", &e.spread, 0.f, 90.f);
      ImGui::DragFloat3("Gravity", &e.gravity.x, 0.05f, -50.f, 50.f);

      ImGui::Separator();
      ImGui::Checkbox("Storm mode", &e.stormMode);
      const char* stormTypes[] = {"Sand", "Snow"};
      ImGui::Combo("Storm type", &e.stormType, stormTypes, IM_ARRAYSIZE(stormTypes));
      ImGui::DragFloat3("Wind", &e.wind.x, 0.1f, -50.f, 50.f);
      ImGui::DragFloat("Turbulence", &e.turbulence, 0.0f, 0.0f, 10.0f);
      ImGui::DragFloat3("Area min", &e.areaMin.x, 0.1f);
      ImGui::DragFloat3("Area max", &e.areaMax.x, 0.1f);

      ImGui::ColorEdit3("Particle color", &e.color.x);
      ImGui::SliderFloat("Base alpha", &e.baseAlpha, 0.0f, 1.0f);

      const char* shapeItems[] = { "Snowflake", "Dust" };
      ImGui::Combo("Particle type", &e.shapeType, shapeItems, IM_ARRAYSIZE(shapeItems));

      ImGui::Text("Particles: %zu", e.particles.size());
      if (ImGui::Button("Pop emitter")) {
        emitters.erase(emitters.begin() + i);
        ImGui::TreePop();
        break;
      }
      ImGui::TreePop();
    }
  }

  ImGui::End();
}



Renderer::~Renderer()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}
