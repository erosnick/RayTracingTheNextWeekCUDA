#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define IMGUI_IMPL_OPENGL_LOADER_GLAD

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/ImFileDialog.h>
#include <imgui/stb_image.h>

#include <Shader.h>
#include <filesystem.h>

#include <iostream>

#include "main.h"
#include "Camera.h"

#include "CUDATypes.h"

#include "Processor.h"

void onFrameBufferResize(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
bool bShowDemoWindow = false;
bool bShowAnotherWindow = false;
bool bShowOpenMenuItem = true;

GLFWwindow* window = nullptr;

Float frameTime = 0.0f;
bool bRightMouseButtonDown = false;
bool bMiddleMouseButtonDown = false;

Float2 lastMousePosition = { 0.0f, 0.0f };

Float rotateSpeed = 1.0f;

void APIENTRY glDebugOutput(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam) {
    // 忽略一些不重要的错误/警告代码
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " << message << std::endl;

    switch (source)
    {
    case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
    case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
    case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;

    switch (type)
    {
    case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
    case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
    case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
    case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
    case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
    case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
    case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;

    switch (severity)
    {
    case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
    case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
    case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
    case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
}

void initImGui() {
    // Setup Dear ImGui context.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style.
    ImGui::StyleColorsDark();
    ImGui::GetStyle().ScaleAllSizes(1.0f);
    // ImGui::StyleColorsClassic();

    // Setup platform/Renderer bindings.
    const char* glsl_version = "#version 400";
    ImGui_ImplGlfw_InitForOpenGL(window, false);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'misc/fonts/README.txt' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //fonts.push_back(io.Fonts->AddFontFromFileTTF("./resources/imgui/misc/fonts/Roboto-Medium.ttf", 10.0f));
    //fonts.push_back(io.Fonts->AddFontFromFileTTF("./resources/imgui/misc/fonts/Cousine-Regular.ttf", 10.0f));
    //fonts.push_back(io.Fonts->AddFontFromFileTTF("./resources/imgui/misc/fonts/DroidSans.ttf", 10.0f));
    //fonts.push_back(io.Fonts->AddFontFromFileTTF("./resources/imgui/misc/fonts/ProggyTiny.ttf", 10.0f));
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    ifd::FileDialog::Instance().CreateTexture = [](uint8_t* data, int w, int h, char fmt) -> void* {
        GLuint tex;

        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, (fmt == 0) ? GL_BGRA : GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        return (void*)tex;
    };

    ifd::FileDialog::Instance().DeleteTexture = [](void* tex) {
        GLuint texID = (GLuint)tex;
        glDeleteTextures(1, &texID);
    };
}

void showMenuBar() {
    // Menu Bar
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open", "Ctrl+O")) {
                ifd::FileDialog::Instance().Open("ModelOpenDialog", "Open a model", "Model file (*.obj;){.obj},.*");
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    if (ifd::FileDialog::Instance().IsDone("ModelOpenDialog")) {
        if (ifd::FileDialog::Instance().HasResult()) {
            std::string fileName = ifd::FileDialog::Instance().GetResult().u8string();
            printf("OPEN[%s]\n", fileName.c_str());
        }
        ifd::FileDialog::Instance().Close();
    }
}

void buildImGuiWidgets() {
    // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (bShowDemoWindow)
        ImGui::ShowDemoWindow(&bShowDemoWindow);

    // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Hello, world!", nullptr, ImGuiWindowFlags_MenuBar);                          // Create a window called "Hello, world!" and append into it.

        showMenuBar();

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        //ImGui::PushFont(fonts[2]);
        ImGui::Text("This is some useful text use another font.");
        //ImGui::PopFont();
        ImGui::Checkbox("Demo Window", &bShowDemoWindow);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &bShowAnotherWindow);
        if (ImGui::SliderFloat("Aperture", &camera->getAperture(), 0.0f, 2.0f)) {
            camera->setDirty();
        }

        if (ImGui::SliderFloat("FOV", &camera->getFOV(), 20.0f, 90.0f)) {
            camera->setDirty();
        }

        ImGui::SliderFloat("Camera Speed", &camera->getMovingSpeed(), 1.0f, 5.0f);
        //ImGui::ColorEdit3("Ambient", (float*)&commonMaterial->Ka); // Edit 1 float using a slider from 0.1f to 1.0f
        //ImGui::SliderFloat("Reflection", &commonMaterial->reflectionFactor, 0.0f, 1.0f);
        //ImGui::SliderFloat("Refraction", &commonMaterial->refractionFactor, 0.0f, 1.0f);
        //ImGui::SliderFloat("Shininess", &commonMaterial->shininess, 32.0f, 128.0f);
        //ImGui::SliderFloat("Fog Density", &fog.density, 0.0f, 1.0f);
        //ImGui::SliderFloat("Edge Threshold", &edgeThreshold, 0.05f, 1.0f);
        //ImGui::SliderFloat("Luminance Threshold", &luminanceThreshold, 0.0f, 1.0f);
        //ImGui::SliderFloat("Scale", &scale, -1.0f, 1.0f);
        //ImGui::SliderFloat("SigmaSquared", &sigmaSquared, 0.001f, 10.0f);
        //ImGui::ColorEdit3("Edge color", (float*)&edgeColor);
        //ImGui::ColorEdit3("Clear color", (float*)&clearColor); // Edit 3 floats representing a color
        //ImGui::ColorEdit3("Point Light Color", (float*)&lights[0].color);
        //ImGui::ColorEdit3("Directional Light Color", (float*)&lights[1].color);
        //ImGui::DragFloat3("Light Direction", (float*)&lights[1].position, 0.1f, -1.0f, 1.f);
        //ImGui::DragFloat3("Light Position", (float*)&lights[0].position, 0.1f, -10.0f, 10.f);
        //ImGui::Checkbox("Projective Texture Mapping", &bShowProjector);
        //ImGui::Checkbox("Draw Normals", &bDrawNormals);
        //ImGui::Checkbox("Draw Bloom", &bDrawBloom);
        //ImGui::Checkbox("MSAA", &bMSAA);

        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        auto frameTimeMS = 1000.0f / ImGui::GetIO().Framerate;

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Text("Sample Count: %d", canvas->getSampleCount());

        ImGui::Text("Rendering Time: %.3f ms", canvas->getRenderingTime());

        frameTime = 1.0f / ImGui::GetIO().Framerate;

        //ImGui::Text("Camera Position %f, %f, %f", camera.getEye().x, camera.getEye().y, camera.getEye().z);
        ImGui::End();
    }

    // 3. Show another simple window.
    if (bShowAnotherWindow) {
        ImGui::Begin("Another Window", &bShowAnotherWindow);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)

        ImGui::Text("Hello from another window!");
        if (ImGui::Button("Close Me"))
            bShowAnotherWindow = false;
        ImGui::End();
    }
}

static void pxl_glfw_fps(GLFWwindow* window)
{
    // static fps counters
    static double stamp_prev = 0.0;
    static int    frame_count = 0;

    // locals
    const double stamp_curr = glfwGetTime();
    const double elapsed = stamp_curr - stamp_prev;

    if (elapsed > 0.5)
    {
        stamp_prev = stamp_curr;

        const double fps = (double)frame_count / elapsed;

        int  width, height;
        char tmp[64];

        glfwGetFramebufferSize(window, &width, &height);

        sprintf_s(tmp, 64, "CUDA Path Tracer (%u x %u) - FPS: %.2f", width, height, fps);

        glfwSetWindowTitle(window, tmp);

        frame_count = 0;
    }

    frame_count++;
}

void update() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    buildImGuiWidgets();

    pxl_glfw_fps(window);
}

void renderImGui() {
    // Rendering
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void render(unsigned int textureId, Shader& ourShader, unsigned int VAO) {
    // render
         // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureId);

    // render container
    ourShader.use();
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

}

void onKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void onMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        bRightMouseButtonDown = true;
        double x;
        double y;
        glfwGetCursorPos(window, &x, &y);
        lastMousePosition.x = static_cast<float>(x);
        lastMousePosition.y = static_cast<float>(y);
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        bRightMouseButtonDown = false;
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
        bMiddleMouseButtonDown = true;
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE) {
        bMiddleMouseButtonDown = false;
    }

    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
}

void onScrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
    ImGui_ImplGlfw_ScrollCallback(window, xOffset, yOffset);

    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    camera->walk(static_cast<float>(yOffset / 8.0f));
}

void onMouseMoveCallback(GLFWwindow* window, double x, double y) {
    double dx = (lastMousePosition.x - x) * frameTime;
    double dy = (lastMousePosition.y - y) * frameTime;

    if (bRightMouseButtonDown) {
        camera->yaw(Math::radians(dx * rotateSpeed));
        camera->pitch(Math::radians(dy * rotateSpeed));
    }

    if (bMiddleMouseButtonDown) {
        camera->strafe(static_cast<float>(-dx / 4.0f));
        camera->raise(static_cast<float>(dy / 4.0f));
    }

    lastMousePosition.x = static_cast<float>(x);
    lastMousePosition.y = static_cast<float>(y);
}

void bindCallbacks() {
    glfwSetFramebufferSizeCallback(window, onFrameBufferResize);
    glfwSetKeyCallback(window, onKeyCallback);
    glfwSetMouseButtonCallback(window, onMouseButtonCallback);
    glfwSetScrollCallback(window, onScrollCallback);
    glfwSetCursorPosCallback(window, onMouseMoveCallback);
}

int main() {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    window = glfwCreateWindow(width, height, "CUDA Ray Tracer", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    glfwSetFramebufferSizeCallback(window, onFrameBufferResize);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glDebugMessageControl(GL_DEBUG_SOURCE_API,
        GL_DEBUG_TYPE_ERROR,
        GL_DEBUG_SEVERITY_HIGH,
        0, nullptr, GL_TRUE);

    glDebugMessageCallback(glDebugOutput, nullptr);

    bindCallbacks();

    initImGui();

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("./resources/shaders/4.2.texture.vs", "./resources/shaders/4.2.texture.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    //float vertices[] = {
    //    // positions          // colors           // texture coords
    //     0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
    //     0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
    //    -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
    //    -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    //};
    //unsigned int indices[] = {
    //    0, 1, 3, // first triangle
    //    1, 2, 3  // second triangle
    //};

    float vertices[] = {
        // positions          // colors           // texture coords
              0.0f,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // top right
             width,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, // bottom right
             width, height, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // bottom left
              0.0f, height, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 2, // first triangle
        2, 3, 0  // second triangle
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // load image, create texture and generate mipmaps
    initialize(width, height);

    pathTracing();

    Processor processor;
    processor.setInput(imageData->data, imageData->width, imageData->height);

    //clearBackBuffers();

    // tell opengl for each sampler to which texture unit it belongs to (only has to be done once)
    // -------------------------------------------------------------------------------------------
    ourShader.use(); // don't forget to activate/use the shader before setting uniforms!
    // either set it manually like so:
    auto projectionMatrix = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height));

    ourShader.setMat4("projectionMatrix", projectionMatrix);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window)) {
        // input
        // -----
        processInput(window);

        update();

        pathTracing();

        processor.processData(imageData->data);

        render(processor.getInputTexture(), ourShader, VAO);

        renderImGui();

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();

    cleanup();

    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (ImGui::GetIO().WantCaptureMouse || ImGui::GetIO().WantCaptureKeyboard) {
        return;
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera->strafe(-frameTime);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera->strafe(frameTime);
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera->walk(frameTime);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera->walk(-frameTime);
    }

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        camera->raise(frameTime);
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        camera->raise(-frameTime);
    }

    //camera->orbit(make_float3(0.0f, 0.0f, -1.0f));

    camera->updateViewMatrix();
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void onFrameBufferResize(GLFWwindow* window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}