#include "WorldRenderer.hpp"

#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>
#include <etna/Profiling.hpp>
#include <glm/ext.hpp>
#include <imgui.h>

namespace {
  inline float viewZ(const glm::mat4& view, const glm::vec3& wpos) {
    return (view * glm::vec4(wpos, 1.0f)).z;
  }
}

void WorldRenderer::allocateEmitterGpu(Emitter& e, uint32_t capacity)
{
  auto& ctx = etna::get_context();
  e.gpu.capacity = capacity;

  auto makeSSBO = [&](const char* name){
    return ctx.createBuffer(etna::Buffer::CreateInfo{
      .size = capacity * sizeof(ParticleGpu),
      .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
      .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
      .name = name,
    });
  };

  e.gpu.ssboA = makeSSBO("particlesA");
  e.gpu.ssboB = makeSSBO("particlesB");

  e.gpu.counter = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eTransferSrc,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "aliveCounter",
  });

  e.gpu.indirect = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(uint32_t) * 4,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer |
                   vk::BufferUsageFlagBits::eIndirectBuffer |
                   vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "drawIndirect",
  });

  e.gpu.counterReadback = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "aliveCounterReadback",
  });

  e.gpu.indices = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = capacity * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "particleIndices",
  });
  e.gpu.keys = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = capacity * sizeof(float),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "particleDepthKeys",
  });

  e.gpu.indicesTmp = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = capacity * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "particleIndicesTmp",
  });

  e.gpu.radixBins = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = 256 * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "radixBins256",
  });

  e.gpu.visMask = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = capacity * sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "visMask",
  });

  e.gpu.visibleCounter = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer
                | vk::BufferUsageFlagBits::eTransferDst
                | vk::BufferUsageFlagBits::eTransferSrc,
    .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY,
    .name = "visibleCounter",
  });

  e.gpu.visibleCounterReadback = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = sizeof(uint32_t),
    .bufferUsage = vk::BufferUsageFlagBits::eTransferDst,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "visibleCounterReadback",
  });
}

WorldRenderer::WorldRenderer()
  : sceneMgr{std::make_unique<SceneManager>()}
{
  Emitter e;
  e.position  = {0.f, 5.f, 0.f};
  e.spawnRate = 1000.f;
  e.lifeRange = {1.5f, 10.0f};
  e.speedRange= {0.5f, 2.0f};
  e.sizeRange = {2.f, 6.f};
  e.spread    = 0.f;
  e.enabled = true;

  e.stormMode = true;
  e.stormType = 0;
  e.areaMin   = {-25.f, 2.f, -25.f};
  e.areaMax   = {25.f, 8.f, 25.f};
  e.wind      = {6.f, 0.f, 0.f};
  e.turbulence= 1.5f;
  e.color     = {0.92f, 0.86f, 0.60f};
  e.baseAlpha = 0.75f;

  allocateEmitterGpu(e, 20'000'000);
  emitters_.emplace_back(std::move(e));
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
  etna::create_program("simple_material",{PARTICLES2_RENDERER_SHADERS_ROOT "simple_shadow.frag.spv", PARTICLES2_RENDERER_SHADERS_ROOT "simple.vert.spv"});
  etna::create_program("simple_shadow", {PARTICLES2_RENDERER_SHADERS_ROOT "simple.vert.spv"});
  etna::create_program("particles",    {PARTICLES2_RENDERER_SHADERS_ROOT "particle.frag.spv", PARTICLES2_RENDERER_SHADERS_ROOT "particle.vert.spv"});
  etna::create_program("particles_instanced", {PARTICLES2_RENDERER_SHADERS_ROOT "particle_instanced.frag.spv", PARTICLES2_RENDERER_SHADERS_ROOT "particle_instanced.vert.spv"});
  etna::create_program("spawn",        {PARTICLES2_RENDERER_SHADERS_ROOT "spawn.gpu.comp.spv"});
  etna::create_program("simulate",     {PARTICLES2_RENDERER_SHADERS_ROOT "simulate.gpu.comp.spv"});
  etna::create_program("writeIndirect",{PARTICLES2_RENDERER_SHADERS_ROOT "writeIndirect.gpu.comp.spv"});
  etna::create_program("particleKeys", {PARTICLES2_RENDERER_SHADERS_ROOT "particle_keys.gpu.comp.spv"});
  etna::create_program("bitonic",      {PARTICLES2_RENDERER_SHADERS_ROOT "bitonic_sort.gpu.comp.spv"});
  etna::create_program("frustumMask",  {PARTICLES2_RENDERER_SHADERS_ROOT "frustum_mask.gpu.comp.spv"});
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
  auto& pm = etna::get_context().getPipelineManager();
  frustumMaskPipeline = pm.createComputePipeline("frustumMask", {});

  spawnPipeline        = pipelineManager.createComputePipeline("spawn", {});
  simulatePipeline     = pipelineManager.createComputePipeline("simulate", {});
  writeIndirectPipeline= pipelineManager.createComputePipeline("writeIndirect", {});
  keysPipeline         = pipelineManager.createComputePipeline("particleKeys", {});
  bitonicPipeline      = pipelineManager.createComputePipeline("bitonic", {});

  instancedParticlesPipeline = pipelineManager.createGraphicsPipeline(
    "particles_instanced",
    etna::GraphicsPipeline::CreateInfo{
      .inputAssemblyConfig = vk::PipelineInputAssemblyStateCreateInfo{
        .topology = vk::PrimitiveTopology::ePointList,
        .primitiveRestartEnable = VK_FALSE,
      },
      .rasterizationConfig =
        vk::PipelineRasterizationStateCreateInfo{
          .polygonMode = vk::PolygonMode::eFill,
          .cullMode = vk::CullModeFlagBits::eNone,
          .frontFace = vk::FrontFace::eCounterClockwise,
          .lineWidth = 1.f,
        },
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {swapchain_format},
          .depthAttachmentFormat  = vk::Format::eD32Sfloat,
        },
  });

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
  lastDtClamped = dtClamped;
  const glm::mat4 mainView = packet.mainCam.viewTm();

  {
    const float aspect = float(resolution.x) / float(resolution.y);
    worldViewProj = packet.mainCam.projTm(aspect) * packet.mainCam.viewTm();
  }

  {
    const glm::mat4 V = glm::lookAtLH(
        lightPos,
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );
    const float fovDeg = 60.0f;
    const glm::mat4 P = lightProps.usePerspectiveM
      ? glm::perspectiveLH_ZO(-glm::radians(fovDeg), 1.0f, 1.0f, lightProps.lightTargetDist * 2.0f)
      : glm::orthoLH_ZO(
          lightProps.radius, -lightProps.radius,
          lightProps.radius, -lightProps.radius,
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

  for (auto& e : emitters_) e.viewDepth = viewZ(mainView, e.position);
  std::sort(emitters_.begin(), emitters_.end(),
    [](const Emitter& a, const Emitter& b){ return a.viewDepth > b.viewDepth; });
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
      cmd_buf, {{0,0},{2048,2048}}, {},
      {.image = shadowMap.get(), .view = shadowMap.getView({})});
    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, shadowPipeline.getVkPipeline());
    renderScene(cmd_buf, lightMatrix, shadowPipeline.getVkPipelineLayout());
  }
  runParticlesCompute(cmd_buf);

  {
    ETNA_PROFILE_GPU(cmd_buf, renderForward);

    auto simpleMaterialInfo = etna::get_shader_program("simple_material");
    auto set = etna::create_descriptor_set(
      simpleMaterialInfo.getDescriptorLayoutId(0), cmd_buf,
      { etna::Binding{0, constants.genBinding()},
        etna::Binding{1, shadowMap.genBinding(defaultSampler.get(),
                        vk::ImageLayout::eShaderReadOnlyOptimal)} });

    etna::RenderTargetState renderTargets(
      cmd_buf, {{0,0},{resolution.x, resolution.y}},
      {{.image=target_image, .view=target_image_view}},
      {.image=mainViewDepth.get(), .view=mainViewDepth.getView({})});

    cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics, basicForwardPipeline.getVkPipeline());
    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               basicForwardPipeline.getVkPipelineLayout(),
                               0, {set.getVkSet()}, {});
    renderScene(cmd_buf, worldViewProj, basicForwardPipeline.getVkPipelineLayout());
    renderParticles(cmd_buf);
  }

  if (drawDebugFSQuad)
    quadRenderer->render(cmd_buf, target_image, target_image_view, shadowMap, defaultSampler);
}

void WorldRenderer::renderParticles(vk::CommandBuffer cmd_buf)
{
  if (emitters_.empty()) return;
  cmd_buf.bindPipeline(vk::PipelineBindPoint::eGraphics,
                        instancedParticlesPipeline.getVkPipeline());

  auto prog = etna::get_shader_program("particles_instanced");

  for (auto& e : emitters_)
  {
    if (!e.enabled) continue;
    auto& currIn = (!e.gpu.ping ? e.gpu.ssboA : e.gpu.ssboB);

    auto set = etna::create_descriptor_set(prog.getDescriptorLayoutId(0), cmd_buf, {
      etna::Binding{0, currIn.genBinding()},
      etna::Binding{1, e.gpu.indices.genBinding()},
    });

    cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                instancedParticlesPipeline.getVkPipelineLayout(),
                                0, { set.getVkSet() }, {});
    uint32_t drawCount = 0;
    if (e.frustumCull) {
      if (void* p = e.gpu.visibleCounterReadback.map()) {
        std::memcpy(&drawCount, p, sizeof(uint32_t));
        e.gpu.visibleCounterReadback.unmap();
      }
    } else {
      if (void* p = e.gpu.counterReadback.map()) {
        std::memcpy(&drawCount, p, sizeof(uint32_t));
        e.gpu.counterReadback.unmap();
      }
    }
    uint32_t colorMode = (e.sortAlways || e.sortRequest) ? (e.radixSort ? 2u : 1u) : 0u;
    struct VSPC {
      glm::mat4 projView;
      uint32_t  useIndices;
      uint32_t  drawCount;
      uint32_t  colorMode;
      uint32_t  _pad;
    };
    VSPC vpc{ worldViewProj, (e.useIndicesThisFrame ? 1u : 0u), drawCount, colorMode, 0u };

    cmd_buf.pushConstants(instancedParticlesPipeline.getVkPipelineLayout(),
                          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                          0, sizeof(VSPC), &vpc);
    cmd_buf.drawIndirect(e.gpu.indirect.get(), 0, 1, sizeof(uint32_t) * 4);
  }
}

void WorldRenderer::runParticlesCompute(vk::CommandBuffer cmd_buf)
{
  auto spawnProg         = etna::get_shader_program("spawn");
  auto simulateProg      = etna::get_shader_program("simulate");
  auto writeIndirectProg = etna::get_shader_program("writeIndirect");
  auto keysProg          = etna::get_shader_program("particleKeys");
  auto bitonicProg       = etna::get_shader_program("bitonic");
  auto doBarrier = [&](vk::PipelineStageFlags2 srcStage, vk::AccessFlags2 srcAccess,
                       vk::PipelineStageFlags2 dstStage, vk::AccessFlags2 dstAccess)
  {
    vk::MemoryBarrier2 mb{};
    mb.sType         = vk::StructureType::eMemoryBarrier2;
    mb.srcStageMask  = srcStage;
    mb.srcAccessMask = srcAccess;
    mb.dstStageMask  = dstStage;
    mb.dstAccessMask = dstAccess;

    vk::DependencyInfo dep{};
    dep.sType               = vk::StructureType::eDependencyInfo;
    dep.memoryBarrierCount  = 1;
    dep.pMemoryBarriers     = &mb;

    cmd_buf.pipelineBarrier2(dep);
  };

  struct ParticlesPushConst
  {
    float     dtSpawn;
    float     rate;
    float     emitRemainder;
    uint32_t  toSpawn;

    glm::vec3 emitterPos;  float _pad0;
    glm::vec3 emitterVel;  float _pad1;
    glm::vec3 emitterDir;  float emitterSpread;

    float lifeMin,  lifeMax;
    float speedMin, speedMax;
    float sizeMin,  sizeMax;

    uint32_t capacity;     uint32_t _pad2[3];
  };

  struct SimPushConst {
    float dt;
    uint32_t inCount;
    uint32_t capacity;
    uint32_t _pad;
  };

  for (auto& e : emitters_)
  {
    if (!e.enabled) continue;
    auto& outSSBO = (e.gpu.ping ? e.gpu.ssboA : e.gpu.ssboB);
    auto& inSSBO  = (e.gpu.ping ? e.gpu.ssboB : e.gpu.ssboA);

    uint32_t prevAlive = 0;
    if (void* p = e.gpu.counterReadback.map()) {
      std::memcpy(&prevAlive, p, sizeof(uint32_t));
      e.gpu.counterReadback.unmap();
    }

    if (e.gpu.counter.get() && e.gpu.indirect.get())
    {
      cmd_buf.fillBuffer(e.gpu.counter.get(),  0, VK_WHOLE_SIZE, 0);
      cmd_buf.fillBuffer(e.gpu.indirect.get(), 0, VK_WHOLE_SIZE, 0);
      if (e.gpu.visibleCounter.get()) {
        cmd_buf.fillBuffer(e.gpu.visibleCounter.get(), 0, VK_WHOLE_SIZE, 0);
      }

      doBarrier(vk::PipelineStageFlagBits2::eTransfer,       vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }

    // ---------- SIMULATE ----------
    if (simulatePipeline.getVkPipeline())
    {
      auto simulateSet = etna::create_descriptor_set(
        simulateProg.getDescriptorLayoutId(0), cmd_buf, {
          etna::Binding{0, inSSBO.genBinding()},
          etna::Binding{1, outSSBO.genBinding()},
          etna::Binding{2, e.gpu.counter.genBinding()},
        });

      SimPushConst spc{ lastDtClamped, prevAlive, e.gpu.capacity, 0u};

      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, simulatePipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 simulatePipeline.getVkPipelineLayout(),
                                 0, { simulateSet.getVkSet() }, {});
      cmd_buf.pushConstants(simulatePipeline.getVkPipelineLayout(),
                        vk::ShaderStageFlagBits::eCompute,
                        0, sizeof(SimPushConst), &spc);

      const uint32_t gx = (prevAlive + 63u) / 64u;
      if (gx > 0) cmd_buf.dispatch(gx, 1, 1);
      doBarrier(vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }

    // ---------- SPAWN ----------
    const bool spawnEnabled = (e.enabled && e.spawnRate > 0.f && maxSpawnPerStep > 0u);
    float     wantF   = spawnEnabled ? (e.spawnRate * lastDtClamped + e.emitRemainder) : 0.f;
    uint32_t  want    = (wantF > 0.f) ? uint32_t(wantF) : 0u;
    uint32_t  toSpawn = spawnEnabled ? std::min<uint32_t>(want, maxSpawnPerStep) : 0u;

    if (spawnPipeline.getVkPipeline())
    {
      if (!spawnEnabled) e.emitRemainder = 0.f;
      else               e.emitRemainder = wantF - float(toSpawn);

      if (spawnEnabled) {
        e.emitRemainder = wantF - float(toSpawn);
      }

      auto spawnSet = etna::create_descriptor_set(
        spawnProg.getDescriptorLayoutId(0), cmd_buf, {
          etna::Binding{0, outSSBO.genBinding()},
          etna::Binding{1, e.gpu.counter.genBinding()},
        });

      ParticlesPushConst pc{};
      pc.dtSpawn        = lastDtClamped;
      pc.rate           = e.spawnRate;
      pc.emitRemainder  = e.emitRemainder;
      pc.toSpawn        = toSpawn;

      pc.emitterPos     = e.position;
      pc.emitterVel     = e.wind;
      pc.emitterDir     = glm::vec3(0,1,0);
      pc.emitterSpread  = glm::radians(e.spread);

      pc.lifeMin        = e.lifeRange.x;     pc.lifeMax  = e.lifeRange.y;
      pc.speedMin       = e.speedRange.x;    pc.speedMax = e.speedRange.y;
      pc.sizeMin        = e.sizeRange.x;     pc.sizeMax  = e.sizeRange.y;

      pc.capacity       = e.gpu.capacity;

      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, spawnPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 spawnPipeline.getVkPipelineLayout(),
                                 0, { spawnSet.getVkSet() }, {});
      cmd_buf.pushConstants(spawnPipeline.getVkPipelineLayout(),
                            vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(pc), &pc);
      const uint32_t gx = (toSpawn + 63u) / 64u;
      if (gx > 0) cmd_buf.dispatch(gx, 1, 1);
      doBarrier(vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }

    const uint32_t aliveEst = std::min(e.gpu.capacity, prevAlive + toSpawn);
    bool doSort = (e.sortAlways || e.sortRequest);
    e.sortRequest = false;
    e.useIndicesThisFrame = doSort;

    // ---------- FRUSTUM CULL ----------
    if (e.frustumCull && frustumMaskPipeline.getVkPipeline())
    {
      auto fmProg = etna::get_shader_program("frustumMask");

      auto fmSet = etna::create_descriptor_set(
        fmProg.getDescriptorLayoutId(0), cmd_buf, {
          etna::Binding{0, outSSBO.genBinding()},
          etna::Binding{1, e.gpu.visMask.genBinding()},
          etna::Binding{2, e.gpu.visibleCounter.genBinding()},
        });

      struct FMPC { glm::mat4 projView; uint32_t count; } fmpc{ worldViewProj, aliveEst };

      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, frustumMaskPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                frustumMaskPipeline.getVkPipelineLayout(),
                                0, { fmSet.getVkSet() }, {});
      cmd_buf.pushConstants(frustumMaskPipeline.getVkPipelineLayout(),
                            vk::ShaderStageFlagBits::eCompute, 0, sizeof(FMPC), &fmpc);

      const uint32_t gxFM = (aliveEst + 63u) / 64u;
      if (gxFM > 0) cmd_buf.dispatch(gxFM, 1, 1);
      doBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
      cmd_buf.copyBuffer(e.gpu.visibleCounter.get(), e.gpu.visibleCounterReadback.get(),
                        { vk::BufferCopy{0, 0, sizeof(uint32_t)} });
    }

    // ---------- KEYS (depth) ----------
    if (doSort && keysPipeline.getVkPipeline())
    {
      auto keysSet = etna::create_descriptor_set(
        keysProg.getDescriptorLayoutId(0), cmd_buf, {
          etna::Binding{0, outSSBO.genBinding()},
          etna::Binding{1, e.gpu.keys.genBinding()},
          etna::Binding{2, e.gpu.indices.genBinding()},
          etna::Binding{3, e.gpu.counter.genBinding()},
          etna::Binding{4, e.gpu.visMask.genBinding()},
        });

      struct KPC { glm::mat4 projView; } kpc{ worldViewProj };

      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, keysPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 keysPipeline.getVkPipelineLayout(),
                                 0, { keysSet.getVkSet() }, {});
      cmd_buf.pushConstants(keysPipeline.getVkPipelineLayout(),
                            vk::ShaderStageFlagBits::eCompute,
                            0, sizeof(KPC), &kpc);

      auto nextPow2 = [](uint32_t v){ v = std::max(1u, v); v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; v++; return v; };
      const uint32_t N2  = nextPow2(aliveEst);
      const uint32_t gxK = (N2 + 63u) / 64u;

      if (gxK > 0) cmd_buf.dispatch(gxK, 1, 1);

      doBarrier(vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }

    // ---------- SORT ----------
    if (doSort && bitonicPipeline.getVkPipeline())
    {
      auto sortSet = etna::create_descriptor_set(
        bitonicProg.getDescriptorLayoutId(0), cmd_buf, {
          etna::Binding{0, e.gpu.keys.genBinding()},
          etna::Binding{1, e.gpu.indices.genBinding()},
          etna::Binding{2, e.gpu.counter.genBinding()},
          etna::Binding{3, e.gpu.radixBins.genBinding()},
          etna::Binding{4, e.gpu.indicesTmp.genBinding()},
        });

      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, bitonicPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 bitonicPipeline.getVkPipelineLayout(),
                                 0, { sortSet.getVkSet() }, {});

      auto nextPow2 = [](uint32_t v){ v = std::max(1u, v); v--; v|=v>>1; v|=v>>2; v|=v>>4; v|=v>>8; v|=v>>16; v++; return v; };
      struct SPC { uint32_t mode, j, k, count, shift; } spc{};
      if (doSort && e.radixSort)
      {
        const uint32_t count = prevAlive;
        for (uint32_t sh = 0; sh < 32; sh += 8)
        {
          cmd_buf.fillBuffer(e.gpu.radixBins.get(), 0, VK_WHOLE_SIZE, 0);
          doBarrier(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
          spc = {1u, 0u, 0u, count, sh};
          cmd_buf.pushConstants(bitonicPipeline.getVkPipelineLayout(),
                                vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPC), &spc);
          const uint32_t gxH = (count + 255u) / 256u;
          if (gxH > 0) cmd_buf.dispatch(gxH, 1, 1);
          doBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
          spc = {2u, 0u, 0u, 256u, 0u};
          cmd_buf.pushConstants(bitonicPipeline.getVkPipelineLayout(),
                                vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPC), &spc);
          cmd_buf.dispatch(1, 1, 1);
          doBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
          spc = {3u, 0u, 0u, count, sh};
          cmd_buf.pushConstants(bitonicPipeline.getVkPipelineLayout(),
                                vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPC), &spc);
          const uint32_t gxT = (count + 255u) / 256u;
          if (gxT > 0) cmd_buf.dispatch(gxT, 1, 1);
          std::swap(e.gpu.indices, e.gpu.indicesTmp);
          doBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
        }
      }
      else if (doSort)
      {
        const uint32_t N2  = nextPow2(prevAlive);
        const uint32_t gxS = (N2 + 255u) / 256u;

        for (uint32_t k = 2; k <= N2; k <<= 1) {
          for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            spc = {0u, j, k, 0u, 0u};
            cmd_buf.pushConstants(bitonicPipeline.getVkPipelineLayout(),
                                  vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPC), &spc);
            cmd_buf.dispatch(gxS, 1, 1);
          }
        }
      }

      doBarrier(vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eComputeShader,  vk::AccessFlagBits2::eShaderRead);
    }

    // ---------- WRITE INDIRECT ----------
    if (writeIndirectPipeline.getVkPipeline())
    {
      auto wiSet = etna::create_descriptor_set(
        writeIndirectProg.getDescriptorLayoutId(0), cmd_buf, {
          etna::Binding{0, e.gpu.counter.genBinding()},
          etna::Binding{1, e.gpu.indirect.genBinding()},
          etna::Binding{2, e.gpu.visibleCounter.genBinding()},
        });

      cmd_buf.bindPipeline(vk::PipelineBindPoint::eCompute, writeIndirectPipeline.getVkPipeline());
      cmd_buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                 writeIndirectPipeline.getVkPipelineLayout(),
                                 0, { wiSet.getVkSet() }, {});
      struct WIPC { uint32_t useVisible; } wipc{ e.frustumCull ? 1u : 0u };
      cmd_buf.pushConstants(writeIndirectPipeline.getVkPipelineLayout(),
                            vk::ShaderStageFlagBits::eCompute, 0, sizeof(WIPC), &wipc);
      cmd_buf.dispatch(1, 1, 1);
      doBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite);
      cmd_buf.copyBuffer(e.gpu.counter.get(), e.gpu.counterReadback.get(),
                         { vk::BufferCopy{ 0, 0, sizeof(uint32_t) } });
    }

    e.gpu.ping = !e.gpu.ping;
    vk::MemoryBarrier2 mbs[2]{};

    mbs[0].sType         = vk::StructureType::eMemoryBarrier2;
    mbs[0].srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
    mbs[0].srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    mbs[0].dstStageMask  = vk::PipelineStageFlagBits2::eVertexShader;
    mbs[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead;

    mbs[1].sType         = vk::StructureType::eMemoryBarrier2;
    mbs[1].srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
    mbs[1].srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
    mbs[1].dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect;
    mbs[1].dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead;

    vk::DependencyInfo dep{};
    dep.sType              = vk::StructureType::eDependencyInfo;
    dep.memoryBarrierCount = 2;
    dep.pMemoryBarriers    = mbs;

    cmd_buf.pipelineBarrier2(dep);
  }
}

WorldRenderer::Emitter& WorldRenderer::addEmitter(uint32_t capacity)
{
  emitters_.emplace_back();
  auto& e = emitters_.back();
  e.enabled = true;
  allocateEmitterGpu(e, capacity);
  return e;
}

void WorldRenderer::clearEmitters()
{
  for (auto& e : emitters_)
    destroyEmitterGpu(e);
  emitters_.clear();
}

bool WorldRenderer::removeEmitter(size_t index)
{
  if (index >= emitters_.size()) return false;
  destroyEmitterGpu(emitters_[index]);
  emitters_.erase(emitters_.begin() + index);
  return true;
}

bool WorldRenderer::resizeEmitter(size_t index, uint32_t newCapacity)
{
  if (index >= emitters_.size() || newCapacity == 0) return false;
  auto& e = emitters_[index];
  if (e.gpu.capacity == newCapacity) return true;
  destroyEmitterGpu(e);
  allocateEmitterGpu(e, newCapacity);
  return true;
}

void WorldRenderer::destroyEmitterGpu(Emitter& e)
{
  e.gpu.ssboA     = {};
  e.gpu.ssboB     = {};
  e.gpu.counter   = {};
  e.gpu.indirect  = {};
  e.gpu.capacity  = 0;
  e.gpu.ping      = false;
  e.gpu.counterReadback = {};
  e.gpu.indices  = {};
  e.gpu.keys     = {};
  e.gpu.indicesTmp = {};
  e.gpu.radixBins  = {};
  e.gpu.visMask = {};
  e.gpu.visibleCounter = {};
  e.gpu.visibleCounterReadback = {};
}