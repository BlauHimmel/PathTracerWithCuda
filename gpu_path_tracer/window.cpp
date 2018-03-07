#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

#include <gl\glew.h>
#include <gl\glfw3.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lib\lodepng\lodepng.h"
#include "path_tracer.hpp"

//===========================================

extern int width, height;

extern double mouse_old_x, mouse_old_y;
extern int mouse_button, mouse_action;

extern float rotate_x, rotate_y;
extern float translate_z;

extern bool is_info_window_show;

extern view_camera* view_cam;
extern render_camera* render_cam;
extern path_tracer* pt;

extern cudaDeviceProp* cuda_prop;

//===========================================

extern void init_all(int argc, char* argv[]);

extern void check_cuda_info();

extern bool init_glew();
extern bool init_config();
extern void init_camera();

extern void release_config();

extern bool render();
extern void render_ui(image* render_image);

extern void callback_window_size(GLFWwindow* window, int w, int h);
extern void callback_key(GLFWwindow* window, int key, int scan_code, int action, int mods);
extern void callback_mouse_button(GLFWwindow* window, int key, int action, int mods);
extern void callback_mouse_position(GLFWwindow* window, double pos_x, double pos_y);
extern void callback_mouse_wheel(GLFWwindow* window, double offset_x, double offset_y);

extern bool screenshot();

//===========================================

int width = 1024, height = 768;

//===========================================

double mouse_old_x = 0, mouse_old_y = 0;
int mouse_button = 0, mouse_action = 0;

float rotate_x = 0.0f, rotate_y = 0.0f;
float translate_z = -30.0f;

bool is_info_window_show = true;

view_camera* view_cam = new view_camera();
render_camera* render_cam = new render_camera();
path_tracer* pt = new path_tracer();
config_parser* config = new config_parser();
cudaDeviceProp* cuda_prop = new cudaDeviceProp();

int main(int argc, char* argv[])
{
	init_all(argc, argv);
	system("PAUSE");
	return 0;
}

void init_all(int argc, char* argv[])
{
	if (!init_config())
	{
		return;
	}

	if (!glfwInit())
	{
		std::cout << "[Info]Failed to initialize GLFW!" << std::endl;
		return;
	}

	char title[2048];
	sprintf(title, "CUDA Path Tracer(%d*%d) - Yuming Zhou", width, height);

	GLFWmonitor* monitor = config->get_config_device_ptr()->use_fullscreen ? glfwGetPrimaryMonitor() : nullptr;;
	GLFWwindow* window = glfwCreateWindow(width, height, title, monitor, nullptr);
	if (window == nullptr)
	{
		glfwTerminate();
		std::cout << "[Info]Failed to create GLFW window!" << std::endl;
		return;
	}

	GLFWimage images[1];
	std::vector<uchar> buffer;
	uint icon_width, icon_height;
	unsigned error = lodepng::decode(buffer, icon_width, icon_height, std::string("res\\icon\\icon.png"));
	if (error != 0)
	{
		std::cout << "[Error]Icon load fail : " << lodepng_error_text(error) << std::endl;
	}
	images[0].height = icon_height; images[0].width = icon_width; images[0].pixels = buffer.data();
	glfwSetWindowIcon(window, 1, images);
	buffer.clear();
	buffer.shrink_to_fit();

	glfwMakeContextCurrent(window);

	glfwSetWindowSizeCallback(window, callback_window_size);
	glfwSetCursorPosCallback(window, callback_mouse_position);

	ImGui_ImplGlfwGL3_Init(window, true);
	ImGui::StyleColorsDark();

	if (!init_glew())
	{
		std::cout << "[Info]Failed to initialize GLEW!" << std::endl;
		return;
	}

	std::cout << "[Info]OpenGL initialized! Version: " << glGetString(GL_VERSION) << std::endl;

	check_cuda_info();

	srand(unsigned(time(0)));

	init_camera();
	view_cam->get_render_camera(render_cam);
	pt->init(render_cam, config);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		if (render() && is_info_window_show)
		{
			ImGui::Render();
		}

		glfwSwapBuffers(window);
	}

	ImGui_ImplGlfwGL3_Shutdown();
	glfwTerminate();

	release_config();

	return;
}

void check_cuda_info()
{
	int device_count;
	CUDA_CALL((cudaGetDeviceCount(&device_count)));

	for (auto i = 0; i < device_count; i++)
	{
		CUDA_CALL(cudaGetDeviceProperties(cuda_prop, i));

		printf("[CUDA]Device%d-Name: %s\n",i , cuda_prop->name);
		printf("[CUDA]Device%d-Compute Capability: %d.%d\n", i, cuda_prop->major, cuda_prop->minor);
		printf("[CUDA]Device%d-Clock Rate: %.2fMHz\n", i, cuda_prop->clockRate / 1000.0f);
		printf("[CUDA]Device%d-Total Global Memory: %.2f GB\n", i, cuda_prop->totalGlobalMem / 1024.0f / 1024.0f);
		printf("[CUDA]Device%d-Bus Width: %d bits\n", i, cuda_prop->memoryBusWidth);
		printf("[CUDA]Device%d-MultiProcessor Count: %d\n", i, cuda_prop->multiProcessorCount);
		printf("[CUDA]Device%d-Threads In Warp: %d\n", i, cuda_prop->warpSize);
		printf("[CUDA]Device%d-Max Registers Per Block: %d\n", i, cuda_prop->regsPerBlock);
		printf("[CUDA]Device%d-Max Shared Memory Per Block: %.2f KB\n", i, cuda_prop->sharedMemPerBlock / 1024.0f);
		printf("[CUDA]Device%d-Max Threads Per Block: %d\n", i, cuda_prop->maxThreadsPerBlock);
		printf("[CUDA]Device%d-Max Threads Per MultiProcessor: %d\n", i, cuda_prop->maxThreadsPerMultiProcessor);
		printf("[CUDA]Device%d-Max Thread Dimensions: (%d, %d, %d)\n", i, cuda_prop->maxThreadsDim[0], cuda_prop->maxThreadsDim[1], cuda_prop->maxThreadsDim[2]);
		printf("[CUDA]Device%d-Max Grid Dimensions: (%d, %d, %d)\n", i, cuda_prop->maxGridSize[0], cuda_prop->maxGridSize[1], cuda_prop->maxGridSize[2]);
	}
}

bool init_glew()
{
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) 
	{
		printf("[Error]Support for necessary OpenGL extensions missing.");
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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glEnable(GL_TEXTURE_2D);

	return true;
}

bool init_config()
{
	double time;
	printf("[Info]Load config data...\n");
	TIME_COUNT_CALL_START();
	if (!config->load_config("res\\configuration\\config.json"))
	{
		std::cout << "[Error]Load config failed!" << std::endl;
		return false;
	}
	TIME_COUNT_CALL_END(time);
	printf("[Info]Load config completed, total time consuming: %.4f ms\n", time);
	config->create_config_device_data();
	width = config->get_config_device_ptr()->width;
	height = config->get_config_device_ptr()->height;
	return true;
}

void init_camera()
{
	view_cam->set_resolution(static_cast<float>(width), static_cast<float>(height));
	view_cam->set_fov(45.0f);
}

void release_config()
{
	config->create_config_device_data();
}

bool render()
{
	view_cam->get_render_camera(render_cam);
	image* render_image = pt->render();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (render_image == nullptr)
	{ 
		return false;
	}

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

	return true;
}

void render_ui(image* render_image)
{
	char buffer[2048];

	ImGui_ImplGlfwGL3_NewFrame();
	ImGui::Begin("Info(Press F1 to open/close this window)", 0, ImGuiWindowFlags_AlwaysAutoResize);
	
	sprintf(buffer, "FPS: %.2f", get_fps(render_image));
	ImGui::Text(buffer);
	sprintf(buffer, "Iteration: %d", render_image->pass_counter);
	ImGui::Text(buffer);
	sprintf(buffer, "Elapse: %.2f sec", get_elapsed_second(render_image));
	ImGui::Text(buffer);

	ImGui::Separator();

	configuration* conf = config->get_config_device_ptr();
	bool is_config_modified = false;
	is_config_modified |= ImGui::SliderInt("Block Size", &conf->block_size, 1, cuda_prop->maxThreadsPerBlock);
	is_config_modified |= ImGui::SliderInt("Trace Depth", &conf->max_tracer_depth, 0, 100);
	is_config_modified |= ImGui::DragFloat("Bias Length", &conf->vector_bias_length, 0.000001f, 0.0f, 1.0f, "%.6f");
	is_config_modified |= ImGui::DragFloat("Energy Threshold", &conf->energy_exist_threshold, 0.000001f, 0.0f, 1.0f, "%.6f");
	is_config_modified |= ImGui::DragFloat("SSS Threshold", &conf->sss_threshold, 0.000001f, 0.0f, 1.0f, "%.6f");
	is_config_modified |= ImGui::Checkbox("Skybox", &conf->use_sky_box);
	is_config_modified |= ImGui::Checkbox("Bilinear Sample", &conf->use_bilinear);
	is_config_modified |= ImGui::Checkbox("Ground", &conf->use_ground);

	if (is_config_modified)
	{
		pt->clear();
	}

	ImGui::Separator();

	if (ImGui::Button("Screenshot(F2)"))
	{
		screenshot();
	}

	ImGui::SameLine();
	if (ImGui::Button("Clear(Space)"))
	{
		init_camera();
		pt->clear();
	}

	ImGui::SameLine();
	if (ImGui::Button("Exit(Esc)"))
	{
		exit(0);
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

void callback_key(GLFWwindow* window, int key, int scan_code, int action, int mods)
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

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		exit(0);
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
	if (ImGui::IsAnyWindowFocused() && is_info_window_show)
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
	std::cout << "Screenshot Success!" << std::endl;
	return true;
}