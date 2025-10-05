add_executable(imgui_renderer
  main.cpp
  Renderer.cpp
  WorldRenderer.cpp
  App.cpp
)

target_link_libraries(imgui_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(imgui_renderer
  shaders/simple.vert
  shaders/simple_shadow.frag
)
