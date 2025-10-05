#include "WorldRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <imgui.h>

namespace {
  inline float frand() { return rand() / float(RAND_MAX); }

  glm::vec3 randomDirInCone(const glm::vec3& axis, float deg) {
    const float rad = glm::radians(deg);
    const float u = frand();
    const float v = frand();
    const float theta = 2.f * glm::pi<float>() * u;
    const float z = 1.f - v * (1.f - std::cos(rad));
    const float r = std::sqrt(std::max(0.f, 1.f - z*z));
    const glm::vec3 local{ r*std::cos(theta), r*std::sin(theta), z };

    const glm::vec3 w = glm::normalize(axis);
    const glm::vec3 a = (std::abs(w.x) > 0.9f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    const glm::vec3 uvec = glm::normalize(glm::cross(a, w));
    const glm::vec3 vvec = glm::cross(w, uvec);
    return uvec*local.x + vvec*local.y + w*local.z;
  }

  inline float viewZ(const glm::mat4& view, const glm::vec3& wpos) {
    return (view * glm::vec4(wpos, 1.0f)).z;
  }
}

WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
{
  Emitter e;
  e.position  = {0.f, 5.f, 0.f};
  e.spawnRate = 600.f;
  e.lifeRange = {1.5f, 4.0f};
  e.speedRange= {0.5f, 2.0f};
  e.sizeRange = {2.f, 6.f};
  e.spread    = 0.f;
  e.stormMode = true;
  e.stormType = 0;
  e.areaMin   = {-25.f, 2.f, -25.f};
  e.areaMax   = {+25.f, 8.f, +25.f};
  e.wind      = {6.f, 0.f, 0.f};
  e.turbulence= 1.5f;
  e.color     = {0.92f, 0.86f, 0.60f};
  e.baseAlpha = 0.75f;

  emitters_.push_back(e);

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
    {PARTICLES_RENDERER_SHADERS_ROOT "simple_shadow.frag.spv", PARTICLES_RENDERER_SHADERS_ROOT "simple.vert.spv"});
  etna::create_program("simple_shadow", {PARTICLES_RENDERER_SHADERS_ROOT "simple.vert.spv"});
  etna::create_program(
    "particles",
    {PARTICLES_RENDERER_SHADERS_ROOT "particle.frag.spv", PARTICLES_RENDERER_SHADERS_ROOT "particle.vert.spv"});
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

  etna::VertexShaderInputDescription particleVertexInputDesc{
    .bindings = {},
  };

  vk::PipelineColorBlendAttachmentState particlesBlend{
    .blendEnable         = VK_TRUE,
    .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
    .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
    .colorBlendOp        = vk::BlendOp::eAdd,
    .srcAlphaBlendFactor = vk::BlendFactor::eOne,
    .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
    .alphaBlendOp        = vk::BlendOp::eAdd,
    .colorWriteMask      = vk::ColorComponentFlagBits::eR |
                            vk::ColorComponentFlagBits::eG |
                            vk::ColorComponentFlagBits::eB |
                            vk::ColorComponentFlagBits::eA,
  };
  vk::PipelineDepthStencilStateCreateInfo particlesDepth{
    .depthTestEnable  = VK_TRUE,
    .depthWriteEnable = VK_FALSE,
    .depthCompareOp   = vk::CompareOp::eLessOrEqual,
  };

  etna::GraphicsPipeline::CreateInfo particlesInfo{
    .vertexShaderInput = particleVertexInputDesc,
    .inputAssemblyConfig = vk::PipelineInputAssemblyStateCreateInfo{
      .topology = vk::PrimitiveTopology::ePointList,
      .primitiveRestartEnable = VK_FALSE,
    },
    .rasterizationConfig = vk::PipelineRasterizationStateCreateInfo{
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode    = vk::CullModeFlagBits::eNone,
      .frontFace   = vk::FrontFace::eCounterClockwise,
      .lineWidth   = 1.f,
    },
    .blendingConfig = etna::GraphicsPipeline::CreateInfo::Blending{
      .attachments   = { particlesBlend },
      .logicOpEnable = VK_FALSE,
    },
    .depthConfig = particlesDepth,
    .fragmentShaderOutput =
    {
      .colorAttachmentFormats = {swapchain_format},
      .depthAttachmentFormat  = vk::Format::eD32Sfloat,
    },
  };

  particlesPipeline = pipelineManager.createGraphicsPipeline("particles", particlesInfo);
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

  static float s_prevTime = -1.0f;
  const float currTime = packet.currentTime;
  const float dt = (s_prevTime < 0.0f) ? 0.0f : (currTime - s_prevTime);
  s_prevTime = currTime;
  const float dtClamped = glm::clamp(dt, 0.0f, 1.0f / 30.0f);
  const glm::mat4 mainView = packet.mainCam.viewTm();

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

  auto spawnFn = [&](Emitter& e, float dtLocal){
    e.accumulator += e.spawnRate * dtLocal;
    const int toSpawn = int(std::floor(e.accumulator));
    if (toSpawn <= 0) return;
    e.accumulator -= toSpawn;

    for (int i = 0; i < toSpawn; ++i) {
      Particle p;
      if (e.stormMode) {
        glm::vec3 r{ frand(), frand(), frand() };
        p.pos = glm::mix(e.areaMin, e.areaMax, r);
      } else {
        p.pos = e.position;
      }

      p.life = glm::mix(e.lifeRange.x,  e.lifeRange.y,  frand());
      p.size = glm::mix(e.sizeRange.x,  e.sizeRange.y,  frand());

      if (e.stormMode) {
        const bool isSnow = (e.stormType == 1);
        glm::vec3 jitter{ (frand()*2.f-1.f), (frand()*2.f-1.f), (frand()*2.f-1.f) };
        jitter = glm::normalize(jitter + glm::vec3(1e-3f)) * e.turbulence;

        glm::vec3 base = e.wind;
        if (isSnow) base += glm::vec3(0.f, -1.2f, 0.f);

        float spd = glm::mix(e.speedRange.x, e.speedRange.y, frand());
        p.vel = base * spd + jitter * 0.5f;
      } else {
        float speed = glm::mix(e.speedRange.x, e.speedRange.y, frand());
        p.vel  = randomDirInCone(glm::vec3(0,1,0), e.spread) * speed;
      }
      p.age  = 0.f;
      p.alpha= 1.f;
      e.particles.emplace_back(p);
    }
  };

  auto updateFn = [&](Emitter& e, float dtLocal){
    const bool isSnow = (e.stormMode && e.stormType == 1);

    for (auto& p : e.particles) {
      p.age += dtLocal;
      if (p.age < p.life) {
        float t = p.age * 6.28318f;
        glm::vec3 swirl{ std::sin(t*0.7f + p.pos.x), std::cos(t*0.5f + p.pos.z), std::sin(t*0.9f + p.pos.y) };
        p.vel += 0.2f * e.turbulence * swirl * dtLocal;
        glm::vec3 g = e.gravity;
        if (e.stormMode) {
          g = isSnow ? glm::vec3(0.f, -0.5f, 0.f) : glm::vec3(0.f);
        }
        p.vel += g * dtLocal;
        p.pos += p.vel * dtLocal;
        p.alpha = 1.f - (p.age / p.life);
      }
    }

    e.particles.erase(
      std::remove_if(e.particles.begin(), e.particles.end(),
        [](const Particle& p){ return p.age >= p.life; }),
      e.particles.end());
  };

  for (auto& e : emitters_) {
    if (!e.enabled) continue;
    spawnFn(e, dtClamped);
    updateFn(e, dtClamped);
  }

  for (auto& e : emitters_) e.viewDepth = viewZ(mainView, e.position);
  std::sort(emitters_.begin(), emitters_.end(),
    [](const Emitter& a, const Emitter& b){ return a.viewDepth > b.viewDepth; });

  for (auto& e : emitters_) {
    std::sort(e.particles.begin(), e.particles.end(),
      [&](const Particle& A, const Particle& B){
        return viewZ(mainView, A.pos) > viewZ(mainView, B.pos);
      });
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
    renderParticles(cmd_buf);
  }

  if (drawDebugFSQuad)
    quadRenderer->render(cmd_buf, target_image, target_image_view, shadowMap, defaultSampler);
}

void WorldRenderer::renderParticles(vk::CommandBuffer cmd_buf)
{
  size_t totalAlive = 0;
  for (auto& e : emitters_) totalAlive += e.particles.size();
  if (totalAlive == 0) return;

  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, particlesPipeline.getVkPipeline());

  struct ParticlePC {
    glm::mat4 projView;
    glm::vec4 posSize;
    glm::vec4 colorAlpha;
    glm::vec4 viewport;
  };

  for (const auto& e : emitters_) {
    if (!e.enabled) continue;

    const glm::vec3 eColor     = e.color;
    const float     eBaseAlpha = e.baseAlpha;
    const float     gAlpha     = globalAlpha_;
    const float     baseFinal  = glm::clamp(eBaseAlpha * gAlpha, 0.0f, 1.0f);
    const auto      signByType = (e.shapeType == 0) ? +1.0f : -1.0f;
    const float     alphaSignedBase = signByType * baseFinal;

    for (const auto& p : e.particles) {
      ParticlePC pc{
        .projView  = worldViewProj,
        .posSize   = glm::vec4(p.pos, p.size),
        .colorAlpha= glm::vec4(eColor, alphaSignedBase * p.alpha),
        .viewport   = glm::vec4(float(resolution.x), float(resolution.y), 0.f, 0.f),
      };

      cmd_buf.pushConstants<ParticlePC>(
        particlesPipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, {pc});

      cmd_buf.draw(1, 1, 0, 0);
    }
  }
}

