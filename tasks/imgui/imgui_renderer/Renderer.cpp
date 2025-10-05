#include "Renderer.hpp"
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

  etna::initialize(etna::InitParams{
    .applicationName = "ShadowmapSample",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .instanceExtensions = instanceExtensions,
    .deviceExtensions = deviceExtensions,
    .features = vk::PhysicalDeviceFeatures2{.features = {}},
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
  ImGui::Begin("Render Settings");

  if (ImGui::CollapsingHeader("Application", ImGuiTreeNodeFlags_DefaultOpen))
  {
    if (ImGui::Checkbox("Use Vsync", &useVsync_))
      wantRecreateSwapchain_ = true;

    ImGui::SameLine();
    if (ImGui::Button("Reload shaders"))
      reloadShaders();

    ImGui::Checkbox("Show Demo Window", &showDemo_);
  }

  {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SeparatorText("Metrics");
    ImGui::Text("FPS: %.1f (%.3f ms)", io.Framerate, 1000.0f / (io.Framerate > 0 ? io.Framerate : 1.0f));
    ImGui::Text("Resolution: %ux%u", resolution.x, resolution.y);
    ImGui::Text("Time: %.3f", lastPacket_.currentTime);
  }

  if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
  {
    glm::vec3 lightPos = worldRenderer->getLightPos();
    if (ImGui::DragFloat3("Light source pos", &lightPos.x, 0.1f))
      worldRenderer->setLightPos(lightPos);

    glm::vec3 lightColor = worldRenderer->getLightColor();
    if (ImGui::ColorEdit3("Light color", &lightColor.x))
      worldRenderer->setLightColor(lightColor);

    float lightIntensity = worldRenderer->getLightIntensity();
    if (ImGui::SliderFloat("Intensity", &lightIntensity, 0.0f, 10.0f))
      worldRenderer->setLightIntensity(lightIntensity);
  }

  if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
  {
    bool wireframe = worldRenderer->getWireframe();
    if (ImGui::Checkbox("Wireframe", &wireframe))
      worldRenderer->setWireframe(wireframe);

    bool baseColorOnly = worldRenderer->getBaseColorOnly();
    if (ImGui::Checkbox("Meshes base color", &baseColorOnly))
      worldRenderer->setBaseColorOnly(baseColorOnly);

    bool showNormals = worldRenderer->getShowNormals();
    if (ImGui::Checkbox("Show normals", &showNormals))
      worldRenderer->setShowNormals(showNormals);

    float gamma = worldRenderer->getGamma();
    if (ImGui::SliderFloat("Gamma", &gamma, 1.6f, 2.6f))
      worldRenderer->setGamma(gamma);

    float exposure = worldRenderer->getExposure();
    if (ImGui::SliderFloat("Exposure", &exposure, 0.0f, 5.0f))
      worldRenderer->setExposure(exposure);
  }

  if (ImGui::CollapsingHeader("Shadows", ImGuiTreeNodeFlags_DefaultOpen))
  {
    int resIndex = worldRenderer->getShadowMapResIndex();
    const char* resItems[] = { "1024", "2048", "4096" };
    if (ImGui::Combo("Shadow map res", &resIndex, resItems, IM_ARRAYSIZE(resItems)))
      worldRenderer->setShadowMapResIndex(resIndex);

    float bias = worldRenderer->getShadowBias();
    if (ImGui::SliderFloat("Depth bias", &bias, 0.0f, 0.01f, "%.5f"))
      worldRenderer->setShadowBias(bias);

    bool pcf = worldRenderer->getShadowPCF();
    if (ImGui::Checkbox("PCF filtering", &pcf))
      worldRenderer->setShadowPCF(pcf);
  }

  if (ImGui::CollapsingHeader("Utilities"))
  {
    static bool pauseTime = false;
    ImGui::Checkbox("Pause time", &pauseTime);
    if (pauseTime) {
    }

    static float timeScale = 1.0f;
    if (ImGui::SliderFloat("Time scale", &timeScale, 0.0f, 2.0f))
      worldRenderer->setTimeScale(timeScale);

    if (ImGui::Button("Screenshot")) {
      worldRenderer->saveScreenshot();
    }
  }

  ImGui::End();
}



Renderer::~Renderer()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}
