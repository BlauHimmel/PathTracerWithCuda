#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

#include <Windows.h>
#include <gl\glew.h>
#include <gl\glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lib\lodepng\lodepng.h"
#include "path_tracer.hpp"

using namespace std;

int width = 800, height = 800;

//===========================================

double mouse_old_x = 0, mouse_old_y = 0;
int mouse_button = 0, mouse_action = 0;

float rotate_x = 0.0f, rotate_y = 0.0f;
float translate_z = -30.0f;

bool is_info_window_show = true;

view_camera* view_cam = new view_camera();
render_camera* render_cam = new render_camera();
path_tracer* pt = new path_tracer();

void init_all(int argc, char *argv[]);

bool init_glew();
void init_camera();

void render();
void render_ui(image* render_image);

void callback_window_size(GLFWwindow* window, int w, int h);
void callback_key(GLFWwindow* window, int key, int scanCode, int action, int mods);
void callback_mouse_button(GLFWwindow* window, int key, int action, int mods);
void callback_mouse_position(GLFWwindow* window, double pos_x, double pos_y);
void callback_mouse_wheel(GLFWwindow* window, double offset_x, double offset_y);

bool screenshot();

int main(int argc, char* argv[])
{
	init_all(argc, argv);
	return 0;
}

void init_all(int argc, char* argv[])
{
	if (!glfwInit())
	{
		std::cout << "Failed to initialize GLFW!" << std::endl;
		return;
	}

	GLFWwindow* window = glfwCreateWindow(width, height, "CUDA Path Tracer - Yuming Zhou", nullptr, nullptr);
	if (window == nullptr)
	{
		glfwTerminate();
		std::cout << "Failed to create GLFW window!" << std::endl;
		return;
	}

	glfwMakeContextCurrent(window);

	glfwSetKeyCallback(window, callback_key);
	glfwSetWindowSizeCallback(window, callback_window_size);
	glfwSetMouseButtonCallback(window, callback_mouse_button);
	glfwSetCursorPosCallback(window, callback_mouse_position);
	glfwSetScrollCallback(window, callback_mouse_wheel);

	ImGui_ImplGlfwGL3_Init(window, false);
	ImGui::StyleColorsDark();

	if (!init_glew())
	{
		std::cout << "Failed to initialize GLEW!" << std::endl;
	}

	std::cout << "OpenGL initialized! Version: " << glGetString(GL_VERSION) << std::endl;

	srand(unsigned(time(0)));

	init_camera();
	view_cam->get_render_camera(render_cam);
	pt->init(render_cam);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		render();

		if (is_info_window_show)
		{
			ImGui::Render();
		}

		glfwSwapBuffers(window);
	}

	ImGui_ImplGlfwGL3_Shutdown();
	glfwTerminate();
	return;
}

bool init_glew()
{
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) 
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glBindTexture(GL_TEXTURE_2D, 13);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glEnable(GL_TEXTURE_2D);

	return true;
}

void init_camera()
{
	view_cam->set_resolution(static_cast<float>(width), static_cast<float>(height));
	view_cam->set_fov(45.0f);
}

void render()
{
	view_cam->get_render_camera(render_cam);
	image* render_image = pt->render();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	uchar3* pixels = new uchar3[render_image->pixel_count];
	for (auto y = 0; y < render_image->height; y++)
	{
		for (auto x = 0; x < render_image->width; x++)
		{
			color pixel_color = render_image->get_pixel(x, y);
			pixels[y * render_image->width + x] = math::float_to_8bit(pixel_color / static_cast<float>(render_image->pass_counter));
		}
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, render_image->width, render_image->height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	SAFE_DELETE_ARRAY(pixels);

	glBindTexture(GL_TEXTURE_2D, 13);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex3f(1.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glEnd();

	if (is_info_window_show)
	{
		render_ui(render_image);
	}
}

void render_ui(image* render_image)
{
	char buffer[2048];

	ImGui_ImplGlfwGL3_NewFrame();
	ImGui::Begin("Info(Press F1 to open/close this window)", 0, ImGuiWindowFlags_AlwaysAutoResize);
	
	sprintf(buffer, "FPS: %.2f", get_fps(render_image));
	ImGui::Text(buffer);
	sprintf(buffer, "Iteration: %u", render_image->pass_counter);
	ImGui::Text(buffer);
	sprintf(buffer, "Elapse: %.2f sec", get_elapsed_second(render_image));
	ImGui::Text(buffer);

	ImGui::Separator();

	if (ImGui::Button("Screenshot(F2)"))
	{
		screenshot();
	}

	ImGui::SameLine();
	if (ImGui::Button("Clear(Space)"))
	{
		pt->clear();
	}

	ImGui::Separator();

	ImGui::Text("Focal Distance:");
	float focal_distance = view_cam->get_focal_distance();
	if (ImGui::SliderFloat("(Up,Down)", &focal_distance, 0.0f, view_cam->get_max_focal_distance()))
	{
		view_cam->set_focal_distance(focal_distance);
		pt->clear();
	}

	ImGui::Text("Aperture Radius:");
	float aperture_radius = view_cam->get_aperture_radius();
	if (ImGui::SliderFloat("(Left,Right)", &aperture_radius, 0.0f, view_cam->get_max_aperture_radius()))
	{
		view_cam->set_aperture_radius(aperture_radius);
		pt->clear();
	}

	ImGui::Separator();

	pt->render_ui();

	ImGui::End();
}

void callback_window_size(GLFWwindow* window, int w, int h)
{
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);
}

void callback_key(GLFWwindow* window, int key, int scanCode, int action, int mods)
{
	if (key == GLFW_KEY_F1 && action == GLFW_PRESS)
	{
		is_info_window_show = !is_info_window_show;
		return;
	}

	if (key == GLFW_KEY_F2 && action == GLFW_PRESS)
	{
		screenshot();
		return;
	}

	if (key == GLFW_KEY_UP && action == GLFW_PRESS)
	{
		view_cam->modify_focal_distance(0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
	{
		view_cam->modify_focal_distance(-0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
	{
		view_cam->modify_aperture_radius(-0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
	{
		view_cam->modify_aperture_radius(0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		init_camera();
		pt->clear();
		return;
	}
}

void callback_mouse_button(GLFWwindow* window, int key, int action, int mods)
{
	if (ImGui::IsAnyWindowFocused() && is_info_window_show)
	{
		mouse_button = -1;
		mouse_action = -1;
		return;
	}

	mouse_button = key;
	mouse_action = action;
}

void callback_mouse_position(GLFWwindow* window, double pos_x, double pos_y)
{
	float dx = (float)(mouse_old_x - pos_x);
	float dy = (float)(mouse_old_y - pos_y);

	if (ImGui::IsAnyWindowFocused() && is_info_window_show)
	{
		mouse_old_x = pos_x;
		mouse_old_y = pos_y;
		mouse_button = -1;
		mouse_action = -1;
		return;
	}

	if (mouse_button == GLFW_MOUSE_BUTTON_LEFT && mouse_action != GLFW_RELEASE)
	{
		view_cam->modify_yaw(dx * 0.005f);
		view_cam->modify_pitch(-dy * 0.005f);
		pt->clear();
	}

	if (mouse_button == GLFW_MOUSE_BUTTON_MIDDLE && mouse_action != GLFW_RELEASE)
	{
		view_cam->modify_pan(dx * 0.005f, -dy * 0.005f);
		pt->clear();
	}

	mouse_old_x = pos_x;
	mouse_old_y = pos_y;
}

void callback_mouse_wheel(GLFWwindow* window, double offset_x, double offset_y)
{
	if (ImGui::IsAnyWindowFocused())
	{
		return;
	}

	float delta = offset_y > 0 ? -0.1f : (offset_y < 0 ? 0.1f : 0.0f);
	if (delta != 0)
	{
		view_cam->modify_radius(delta);
	}
	pt->clear();
}

bool screenshot()
{
	std::vector<uchar> pixel_buffer(width * height * 4);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer.data());
	for (auto i = 0; i < width; i++)
	{
		for (auto j = 0; j < height / 2; j++)
		{
			for (auto k = 0; k < 4; k++)
			{
				std::swap(pixel_buffer[(j * width + i) * 4 + k], pixel_buffer[((height - j - 1) * width + i) * 4 + k]);
			}
		}
	}
	lodepng::encode("gpu_path_tracer_screen_shot.png", pixel_buffer, static_cast<uint>(width), static_cast<uint>(height));
	MessageBox(nullptr, L"Screenshot Success!", L"Info", MB_OK);
	return true;
}