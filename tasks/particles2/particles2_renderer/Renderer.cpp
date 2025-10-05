#include "Renderer.hpp"
#include "WorldRenderer.hpp" 

#include <spdlog/spdlog.h>
#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <gui/ImGuiRenderer.hpp>
#include <imgui.h>
#include <cstring>

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
  devFeatures.largePoints = VK_TRUE; // allow gl_PointSize > 1px

  etna::initialize(etna::InitParams{
    .applicationName = "ShadowmapSample",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .instanceExtensions = instanceExtensions,
    .deviceExtensions = deviceExtensions,
    .features = vk::PhysicalDeviceFeatures2{ .features = devFeatures },

    // Replace with an index if etna detects your preferred GPU incorrectly
    .physicalDeviceIndexOverride = {},
    // How much frames we buffer on the GPU without waiting for their completion on the CPU
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

  // Most resources depend on the current resolution, so we recreate them.
  worldRenderer->allocateResources(resolution);

  // Format of the swapchain CAN change on android
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

  // TODO: this makes literally 0 sense here, rename/refactor,
  // it doesn't actually begin anything, just resets descriptor pools
  etna::begin_frame();

  auto nextSwapchainImage = window->acquireNext();

  // NOTE: here, we skip frames when the window is in the process of being
  // re-sized. This is not mandatory, it is possible to submit frames to a
  // "sub-optimal" swap chain and still get something drawn while resizing,
  // but only on some platforms (not windows+nvidia, sadly).
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
  for (auto& e : emitters) {
    uint32_t val = 0;
    if (e.gpu.counterReadback.get()) {
      if (void* p = e.gpu.counterReadback.map()) {
        std::memcpy(&val, p, sizeof(uint32_t));
        e.gpu.counterReadback.unmap();
      }
    }
    totalAlive += val;
  }
  ImGui::Text("Particles alive (GPU): %zu", totalAlive);

  ImGui::SeparatorText("Emitters");

  ImGui::SetNextItemWidth(120);

  if (ImGui::Button("Add emitter")) {
    worldRenderer->addEmitter(10'000'000);
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear all")) {
    worldRenderer->clearEmitters();
  }


  for (int i = 0; i < (int)emitters.size(); ++i) {
    auto& e = emitters[i];
    if (ImGui::TreeNode((std::string("Emitter ") + std::to_string(i)).c_str())) {
      ImGui::Checkbox("Enabled", &e.enabled);
      ImGui::DragFloat3("Position", &e.position.x, 0.05f);
      ImGui::SeparatorText("Limits");
      ImGui::SliderInt("Max spawn / frame",
                      (int*)&worldRenderer->maxSpawnPerStep,
                      1, 131072, "%d", ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat("Spawn rate", &e.spawnRate, 0.f, 20000000.f, "%.0f",ImGuiSliderFlags_Logarithmic);

      ImGui::DragFloat2("Life (min,max)", &e.lifeRange.x, 0.01f, 0.05f, 10.f);
      ImGui::DragFloat2("Speed (min,max)", &e.speedRange.x, 0.01f, 0.0f, 20.f);
      ImGui::DragFloat2("Size (min,max)", &e.sizeRange.x, 0.1f, 1.f, 128.f);
      ImGui::SliderFloat("Spread (deg)", &e.spread, 0.f, 90.f);
      ImGui::DragFloat3("Gravity", &e.gravity.x, 0.05f, -50.f, 50.f);

      if (e.gpu.visibleCounterReadback.get()) {
        if (void* p = e.gpu.visibleCounterReadback.map()) {
          std::memcpy(&e.visibleCountCpuSide, p, sizeof(uint32_t));
          e.gpu.visibleCounterReadback.unmap();
        }
      }
      ImGui::SeparatorText("Frustum culling");
      ImGui::Checkbox("Frustum culling", &e.frustumCull);
      ImGui::Text("Visible (GPU): %u", e.visibleCountCpuSide);
      ImGui::SeparatorText("Sorting");
      if (ImGui::Button(e.sortAlways ? "Sorting: ON (click to stop)"
                               : "Sorting: OFF (click to start)")) {
          e.sortAlways = !e.sortAlways;
          if (e.sortAlways) e.sortRequest = true;
      }
      ImGui::Checkbox("Radix sort (O(n))", &e.radixSort);
      ImGui::Separator();
      if (ImGui::Button("Stop spawn now")) {
          e.spawnRate = 0.f;
          e.emitRemainder = 0.f;
      }

      {
        uint32_t alive = 0;
        if (e.gpu.counterReadback.get()) {
          if (void* p = e.gpu.counterReadback.map()) {
            std::memcpy(&alive, p, sizeof(uint32_t));
            e.gpu.counterReadback.unmap();
          }
        }
      }
      ImGui::SameLine();
      if (ImGui::Button("Pop emitter")) {
        worldRenderer->removeEmitter((size_t)i);
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
