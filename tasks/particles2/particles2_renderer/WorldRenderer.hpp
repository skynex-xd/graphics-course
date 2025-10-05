#pragma once

#include <etna/Image.hpp>
#include <etna/Sampler.hpp>
#include <etna/Buffer.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <glm/glm.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/ComputePipeline.hpp>

#include "shaders/UniformParams.h"
#include "scene/SceneManager.hpp"
#include "render_utils/QuadRenderer.hpp"
#include "wsi/Keyboard.hpp"

#include "FramePacket.hpp"

class WorldRenderer
{
public:

  struct Particle {
    glm::vec3 pos{};
    glm::vec3 vel{};
    float age = 0.f;
    float life = 1.f;
    float size = 16.f;
    float alpha = 1.f;
  };

  struct ParticleGpu {
    glm::vec3 pos{};
    glm::vec3 vel{};
    float lifetime = 0.f;
    float age = 0.f;
    float size = 16.f;
    glm::vec3 color{1,1,1};
    float alpha = 1.f;
  };

  struct GpuBuffers {
    etna::Buffer ssboA;
    etna::Buffer ssboB;
    etna::Buffer counter;
    etna::Buffer indirect;
    etna::Buffer counterReadback;
    etna::Buffer indices, keys;
    etna::Buffer indicesTmp, radixBins;
    etna::Buffer visMask, visibleCounter, visibleCounterReadback;
    uint32_t     capacity = 0;
    bool         ping = false;
  } gpu;

  struct Emitter {
    GpuBuffers gpu;
    glm::vec3 position{0};
    float spawnRate = 80.f;
    glm::vec2 lifeRange{0.8f, 1.6f};
    glm::vec2 speedRange{1.0f, 3.5f};
    glm::vec2 sizeRange{8.f, 24.f};
    float spread = 30.f;
    glm::vec3 gravity{0.f, -4.f, 0.f};
    bool enabled = true;
    float emitRemainder  = 0.f;
    bool sortRequest = false;
    bool useIndicesThisFrame = false;

    bool sortAlways = false;
    bool frustumCull = true;
    uint32_t visibleCountCpuSide = 0;

    bool  stormMode = true;
    bool radixSort = false;
    int   stormType = 0;
    int shapeType = 0;
    glm::vec3 areaMin{-20.f, 2.f, -20.f};
    glm::vec3 areaMax{+20.f, 8.f, +20.f};
    glm::vec3 wind{6.f, 0.f, 0.f};
    float turbulence = 1.5f;
    glm::vec3 color{0.92f, 0.86f, 0.60f};
    float baseAlpha = 0.8f;

    std::vector<Particle> particles;
    float accumulator = 0.f;
    float viewDepth = 0.f;

    Emitter() = default;
    Emitter(Emitter&&) = default;
    Emitter& operator=(Emitter&&) = default;
    Emitter(const Emitter&) = delete;
    Emitter& operator=(const Emitter&) = delete;

  };

  std::vector<Emitter> emitters_;
  std::vector<Emitter>& emitters() { return emitters_; }
  const std::vector<Emitter>& emitters() const { return emitters_; }

  Emitter& addEmitter(uint32_t capacity = 100000);
  void     clearEmitters();
  bool     removeEmitter(size_t index);
  bool     resizeEmitter(size_t index, uint32_t newCapacity);

  WorldRenderer();

  void loadScene(std::filesystem::path path);
  void loadShaders();
  void allocateResources(glm::uvec2 swapchain_resolution);
  void setupPipelines(vk::Format swapchain_format);
  void debugInput(const Keyboard& kb);
  void update(const FramePacket& packet);
  void drawGui();
  void renderWorld(vk::CommandBuffer cmd_buf, vk::Image target_image, vk::ImageView target_image_view);
  bool  getWireframe() const            { return settings_.wireframe; }
  void  setWireframe(bool v)            { settings_.wireframe = v; }
  bool  getBaseColorOnly() const        { return settings_.baseColorOnly; }
  void  setBaseColorOnly(bool v)        { settings_.baseColorOnly = v; }
  bool  getShowNormals() const          { return settings_.showNormals; }
  void  setShowNormals(bool v)          { settings_.showNormals = v; }
  float getGamma() const                { return settings_.gamma; }
  void  setGamma(float v)               { settings_.gamma = v; }
  float getExposure() const             { return settings_.exposure; }
  void  setExposure(float v)            { settings_.exposure = v; }

  glm::vec3 getLightPos() const         { return lightPos; }
  void  setLightPos(const glm::vec3& v) { lightPos = v; }

  glm::vec3 getBaseColor() const        { return uniformParams.baseColor; }
  void      setBaseColor(const glm::vec3& v) { uniformParams.baseColor = v; }

  glm::vec3 getLightColor() const       { return lightColor_; }
  void  setLightColor(const glm::vec3& v) { lightColor_ = v; }

  float getLightIntensity() const   { return lightIntensity_; }
  void  setLightIntensity(float v)  { lightIntensity_ = v; }

  int   getShadowMapResIndex() const    { return settings_.shadowResIndex; }
  void  setShadowMapResIndex(int i)     { settings_.shadowResIndex = i; }

  float getShadowBias() const           { return settings_.shadowBias; }
  void  setShadowBias(float v)          { settings_.shadowBias = v; }

  bool  getShadowPCF() const            { return settings_.shadowPCF; }
  void  setShadowPCF(bool v)            { settings_.shadowPCF = v; }

  void  setTimeScale(float s)           { settings_.timeScale = s; }
  float getTimeScale() const            { return settings_.timeScale; }

  uint32_t maxSpawnPerStep = 4096;


private:
  void renderScene(
    vk::CommandBuffer cmd_buf, const glm::mat4x4& glob_tm, vk::PipelineLayout pipeline_layout);

private:
  std::unique_ptr<SceneManager> sceneMgr;

  etna::Image mainViewDepth;
  etna::Image shadowMap;
  etna::Sampler defaultSampler;
  etna::Buffer constants;

  struct PushConstants
  {
    glm::mat4x4 projView;
    glm::mat4x4 model;
  } pushConst2M;

  glm::mat4x4 worldViewProj;
  glm::mat4x4 lightMatrix;
  glm::vec3 lightPos{10.f, 10.f, 10.f};

  glm::vec3 lightColor_      = glm::vec3(1.0f, 1.0f, 1.0f);
  float lightIntensity_  = 1.0f;

  struct ShadowMapCam
  {
    float radius = 10;
    float lightTargetDist = 24;
    bool  usePerspectiveM = false;
  } lightProps;

  UniformParams uniformParams{
    .lightMatrix = {},
    .lightPos    = {},
    .time        = {},
    .baseColor   = {0.9f, 0.92f, 1.0f},
  };

  etna::GraphicsPipeline basicForwardPipeline{};
  etna::GraphicsPipeline shadowPipeline{};
  etna::GraphicsPipeline particlesPipeline{};

  std::unique_ptr<QuadRenderer> quadRenderer;
  bool drawDebugFSQuad = false;

  glm::uvec2 resolution;

  struct RenderSettings {
    bool  wireframe      = false;
    bool  baseColorOnly  = false;
    bool  showNormals    = false;

    float gamma          = 2.2f;
    float exposure       = 1.0f;

    int   shadowResIndex = 1;
    float shadowBias     = 0.0015f;
    bool  shadowPCF      = true;

    float timeScale      = 1.0f;
  } settings_;

  float lastDtClamped = 1.0f / 60.0f;

  void runParticlesCompute(vk::CommandBuffer cmd_buf);
  void renderParticles(vk::CommandBuffer cmd_buf);
  void allocateEmitterGpu(Emitter& e, uint32_t capacity);
  void destroyEmitterGpu(Emitter& e);

  etna::ComputePipeline  spawnPipeline{};
  etna::ComputePipeline  simulatePipeline{};
  etna::ComputePipeline  writeIndirectPipeline{};
  etna::ComputePipeline keysPipeline;
  etna::ComputePipeline bitonicPipeline;
  etna::ComputePipeline frustumMaskPipeline;
  etna::GraphicsPipeline instancedParticlesPipeline{};
};