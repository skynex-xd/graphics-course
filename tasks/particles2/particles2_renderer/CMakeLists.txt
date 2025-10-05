add_executable(particles2_renderer
  main.cpp
  Renderer.cpp
  WorldRenderer.cpp
  App.cpp
)

target_link_libraries(particles2_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(particles2_renderer
  shaders/simple.vert
  shaders/simple_shadow.frag
  shaders/particle.vert
  shaders/particle.frag
  shaders/particle_instanced.vert
  shaders/particle_instanced.frag
  shaders/spawn.gpu.comp
  shaders/simulate.gpu.comp
  shaders/writeIndirect.gpu.comp
  shaders/particle_keys.gpu.comp
  shaders/bitonic_sort.gpu.comp
  shaders/frustum_mask.gpu.comp
)
