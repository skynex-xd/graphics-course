#include "WorldRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <imgui.h>


WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
{
}

void WorldRenderer::allocateResources(glm::uvec2 swapchain_resolution)
{
  resolution = swapchain_resolution;

  auto& ctx = etna::get_context();

  mainViewDepth = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{resolution.x, resolution.y, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
  });

  shadowMap = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage =
      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  constants = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants",
  });

  constants.map();
}

void WorldRenderer::loadScene(std::filesystem::path path)
{
  sceneMgr->selectScene(path);
}

void WorldRenderer::loadShaders()
{
  etna::create_program(
    "simple_material",
    {IMGUI_RENDERER_SHADERS_ROOT "simple_shadow.frag.spv", IMGUI_RENDERER_SHADERS_ROOT "simple.vert.spv"});
  etna::create_program("simple_shadow", {IMGUI_RENDERER_SHADERS_ROOT "simple.vert.spv"});
}

void WorldRenderer::setupPipelines(vk::Format swapchain_format)
{
  quadRenderer = std::make_unique<QuadRenderer>(QuadRenderer::CreateInfo{
    .format = swapchain_format,
    .rect = {{0, 0}, {512, 512}},
  });

  etna::VertexShaderInputDescription sceneVertexInputDesc{
    .bindings = {etna::VertexShaderInputDescription::Binding{
      .byteStreamDescription = sceneMgr->getVertexFormatDescription(),
    }},
  };


  auto& pipelineManager = etna::get_context().getPipelineManager();

  basicForwardPipeline = {};
  basicForwardPipeline = pipelineManager.createGraphicsPipeline(
    "simple_material",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
          .depthAttachmentFormat = vk::Format::eD32Sfloat,
        },
    });

  shadowPipeline = {};
  shadowPipeline = pipelineManager.createGraphicsPipeline(
    "simple_shadow",
    etna::GraphicsPipeline::CreateInfo{
      .vertexShaderInput = sceneVertexInputDesc,
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eBack,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm,
        },
    });
}

void WorldRenderer::saveScreenshot()
{
    spdlog::info("saveScreenshot(): not implemented yet");
}

void WorldRenderer::debugInput(const Keyboard& kb)
{
  if (kb[KeyboardKey::kQ] == ButtonState::Falling)
    drawDebugFSQuad = !drawDebugFSQuad;

  if (kb[KeyboardKey::kP] == ButtonState::Falling)
    lightProps.usePerspectiveM = !lightProps.usePerspectiveM;
}

void WorldRenderer::update(const FramePacket& packet)
{
  ZoneScoped;

  {
    const float aspect = float(resolution.x) / float(resolution.y);
    worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  }

  {
    const glm::mat4 V = glm::lookAtLH(
        lightPos,
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f));

    const float fovDeg = 60.0f;
    const glm::mat4 P = lightProps.usePerspectiveM
      ? glm::perspectiveLH_ZO(-glm::radians(fovDeg), 1.0f, 1.0f, lightProps.lightTargetDist * 2.0f)
      : glm::orthoLH_ZO(
          +lightProps.radius, -lightProps.radius,
          +lightProps.radius, -lightProps.radius,
          0.0f, lightProps.lightTargetDist);

    lightMatrix = P * V;
  }

  {
    uniformParams.lightMatrix = lightMatrix;
    uniformParams.lightPos   = shader_vec4{ lightPos.x, lightPos.y, lightPos.z, lightIntensity_ };
    uniformParams.lightColor = shader_vec3{ lightColor_.x, lightColor_.y, lightColor_.z };
    uniformParams.time       = packet.currentTime * settings_.timeScale;
    uniformParams.baseColor  = shader_vec3{ uniformParams.baseColor.x,
                                            uniformParams.baseColor.y,
                                            uniformParams.baseColor.z };
    uniformParams.gamma      = settings_.gamma;
    uniformParams.exposure   = settings_.exposure;
    uniformParams.debugMode  = settings_.showNormals ? 1.0f : (settings_.baseColorOnly ? 2.0f : 0.0f);

    void* dst = constants.map();
    std::memcpy(dst, &uniformParams, sizeof(uniformParams));
    constants.unmap();
  }
}

void WorldRenderer::renderScene(
  vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout)
{
  if (!sceneMgr->getVertexBuffer())
    return;

  cmd_buf.bindVertexBuffers(0, {sceneMgr->getVertexBuffer()}, {0});
  cmd_buf.bindIndexBuffer(sceneMgr->getIndexBuffer(), 0, vk::IndexType::eUint32);

  pushConst2M.projView = glob_tm;

  auto instanceMeshes = sceneMgr->getInstanceMeshes();
  auto instanceMatrices = sceneMgr->getInstanceMatrices();

  auto meshes = sceneMgr->getMeshes();
  auto relems = sceneMgr->getRenderElements();

  for (std::size_t instIdx = 0; instIdx < instanceMeshes.size(); ++instIdx)
  {
    pushConst2M.model = instanceMatrices[instIdx];

    cmd_buf.pushConstants<PushConstants>(
      pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, {pushConst2M});

    const auto meshIdx = instanceMeshes[instIdx];

    for (std::size_t j = 0; j < meshes[meshIdx].relemCount; ++j)
    {
      const auto relemIdx = meshes[meshIdx].firstRelem + j;
      const auto& relem = relems[relemIdx];
      cmd_buf.drawIndexed(relem.indexCount, 1, relem.indexOffset, relem.vertexOffset, 0);
    }
  }
}

void WorldRenderer::renderWorld(
  vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view)
{
  ETNA_PROFILE_GPU(cmd_buf, renderWorld);

  {
    ETNA_PROFILE_GPU(cmd_buf, renderShadowMap);

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {2048, 2048}},
      {},
      {.image = shadowMap.get(), .view = shadowMap.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, shadowPipeline.getVkPipeline());
    renderScene(cmd_buf, lightMatrix, shadowPipeline.getVkPipelineLayout());
  }

  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    auto simpleMaterialInfo = etna::get_shader_program("simple_material");

    auto set = etna::create_descriptor_set(
      simpleMaterialInfo.getDescriptorLayoutId(0),
      cmd_buf,
      {etna::Binding{0, constants.genBinding()},
       etna::Binding{
         1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}});

    etna::RenderTargetState renderTargets(
      cmd_buf,
      {{0, 0}, {resolution.x, resolution.y}},
      {{.image = target_image, .view = target_image_view}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, basicForwardPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics,
      basicForwardPipeline.getVkPipelineLayout(),
      0,
      {set.getVkSet()},
      {});

    renderScene(cmd_buf, worldViewProj, basicForwardPipeline.getVkPipelineLayout());
  }

  if (drawDebugFSQuad)
    quadRenderer->render(cmd_buf, target_image, target_image_view, shadowMap, defaultSampler);
}

void WorldRenderer::drawGui()
{
  ImGui::Begin("Simple render settings");
  glm::vec3 base = { uniformParams.baseColor.r,
                     uniformParams.baseColor.g,
                     uniformParams.baseColor.b };
  if (ImGui::ColorEdit3("Meshes base color", &base.x,
        ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs))
  {
    uniformParams.baseColor = { base.r, base.g, base.b };
  }

  glm::vec3 lp = lightPos;
  if (ImGui::DragFloat3("Light source pos", &lp.x, 0.1f, -10.f, 10.f))
  {
    lightPos = lp;
    uniformParams.lightPos = { lp.x, lp.y, lp.z, lightIntensity_ };
  }

  ImGuiIO& io = ImGui::GetIO();
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
              1000.0f / (io.Framerate > 0 ? io.Framerate : 1.0f), io.Framerate);

  ImGui::NewLine();
  ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),
                     "Press 'B' to recompile and reload shaders");

  ImGui::End();
}

