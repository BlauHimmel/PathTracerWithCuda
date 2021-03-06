#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

#include "lib\glew\glew.h"
#include "lib\glfw\glfw3.h"

#include <deque>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvml.h>

#include "lib\lodepng\lodepng.h"
#include "Core\path_tracer.h"
#include "Core\material.h"
#include "Others\device_status.h"

//===========================================

extern std::deque<float> fps_deque;

extern int width, height;

extern float current_fps;
extern int current_iteration;
extern float current_render_time;

extern double mouse_old_x, mouse_old_y;
extern int mouse_button, mouse_action;

extern float rotate_x, rotate_y;
extern float translate_z;

extern bool is_info_window_show;
extern bool is_scene_choose;

extern int scene_index;
extern int scene_num;
extern char** scene_files;

extern view_camera* view_cam;
extern render_camera* render_cam;
extern path_tracer* pt;
extern config_parser* config;
extern cudaDeviceProp* cuda_prop;

extern color256* image_pixels_256_buffer;

//===========================================

extern void init_all(int argc, char* argv[]);

extern bool init_device();
extern bool init_glew();
extern bool init_config();
extern void init_camera();

extern void release_all();

extern bool render();
extern void render_choose_scene();
extern void render_ui(image* render_image);

extern void callback_window_size(GLFWwindow* window, int w, int h);
extern void callback_key(GLFWwindow* window, int key, int scan_code, int action, int mods);
extern void callback_mouse_button(GLFWwindow* window, int key, int action, int mods);
extern void callback_mouse_position(GLFWwindow* window, double pos_x, double pos_y);
extern void callback_mouse_wheel(GLFWwindow* window, double offset_x, double offset_y);

extern std::string screenshot();

//===========================================

std::deque<float> fps_deque;

int width = 1024, height = 768;

float current_fps = 0;
int current_iteration = 0;
float current_render_time = 0.0f;

double mouse_old_x = 0, mouse_old_y = 0;
int mouse_button = 0, mouse_action = 0;

float rotate_x = 0.0f, rotate_y = 0.0f;
float translate_z = -30.0f;

bool is_info_window_show = true;
bool is_scene_choose = false;

int scene_index = 0;
int scene_num = 0;
char** scene_files = nullptr;

view_camera* view_cam = new view_camera();
render_camera* render_cam = new render_camera();
path_tracer* pt = new path_tracer();
config_parser* config = new config_parser();
cudaDeviceProp* cuda_prop = new cudaDeviceProp();

color256* image_pixels_256_buffer = nullptr;

//===========================================
// Program Entry,ここは入口である↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
//===========================================
int main(int argc, char* argv[])		
{										
	init_all(argc, argv);				
	system("PAUSE");					
	return 0;							
}										
//===========================================
// Program Entry,ここは入口である↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
//===========================================

void init_all(int argc, char* argv[])
{
	if (!init_device())
	{
		return;
	}

	if (!glfwInit())
	{
		std::cout << "[Error]Failed to initialize GLFW!" << std::endl;
		return;
	}
	
	if (!init_config())
	{
		return;
	}

	char title[2048];
	sprintf(title, "CUDA Path Tracer(%d*%d) CUDA Acceleration: %s(F2) - Yuming Zhou", width, height, config->get_config_device_ptr()->cuda_acceleration ? "On" : "Off");

	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

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

	image_pixels_256_buffer = new color256[width * height];

	if (!init_glew())
	{
		std::cout << "[Info]Failed to initialize GLEW!" << std::endl;
		return;
	}

	std::cout << "[Info]OpenGL initialized! Version: " << glGetString(GL_VERSION) << std::endl;

	srand(unsigned(time(0)));

	init_camera();
	view_cam->get_render_camera(render_cam);
	std::vector<std::string> scene_files_vec = pt->init(render_cam, config, "res\\scene");
	scene_num = static_cast<int>(scene_files_vec.size());
	scene_files = new char*[scene_num];

	for (auto i = 0; i < scene_num; i++)
	{
		int last_separator_index = static_cast<int>(scene_files_vec[i].find_last_of('\\') + 1);
		std::string filename = scene_files_vec[i].substr(last_separator_index);
		scene_files[i] = new char[filename.length() + 1];
		strcpy(scene_files[i], filename.data());
	}

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (!is_scene_choose)
		{
			render_choose_scene();
			ImGui::Render();
		}
		else
		{
			render();

			if (is_info_window_show)
			{
				ImGui::Render();
			}
		}

		glfwSwapBuffers(window);
	}

	ImGui_ImplGlfwGL3_Shutdown();
	glfwTerminate();
	if (NVML_SUCCESS != nvmlShutdown())
	{
		std::cout << "[Error]Error occur when NVML shutdown." << std::endl;
	}
	release_all();
	return;
}

bool init_device()
{
	int device_count;
	CUDA_CALL((cudaGetDeviceCount(&device_count)));

	bool has_supported_device = false;
	int device_index = 0;

	for (auto i = 0; i < device_count; i++)
	{
		CUDA_CALL(cudaGetDeviceProperties(cuda_prop, i));

		printf("[CUDA]Device%d-Name: %s\n",i , cuda_prop->name);
		printf("[CUDA]Device%d-Compute Capability: %d.%d\n", i, cuda_prop->major, cuda_prop->minor);
		printf("[CUDA]Device%d-Clock Rate: %.2fMHz\n", i, cuda_prop->clockRate / 1000.0f);
		printf("[CUDA]Device%d-Total Global Memory: %.2f GB\n", i, cuda_prop->totalGlobalMem / 1024.0f / 1024.0f);
		printf("[CUDA]Device%d-Bus Width: %d bits\n", i, cuda_prop->memoryBusWidth);

		if (cuda_prop->major == 3)
		{
			printf("[CUDA]Device%d-CUDA Cores: %d Per MultiProcessor\n", i, 192);
		}
		else if (cuda_prop->major == 5)
		{
			printf("[CUDA]Device%d-CUDA Cores: %d Per MultiProcessor\n", i, 128);
		}
		else if (cuda_prop->major == 6)
		{
			if (cuda_prop->minor == 0)
			{
				printf("[CUDA]Device%d-CUDA Cores: %d Per MultiProcessor\n", i, 64);
			}
			else if (cuda_prop->minor == 1 || cuda_prop->minor == 2)
			{
				printf("[CUDA]Device%d-CUDA Cores: %d Per MultiProcessor\n", i, 128);
			}
		}
		else if (cuda_prop->major == 7)
		{
			printf("[CUDA]Device%d-FP32 Cores: %d Per MultiProcessor\n", i, 64);
			printf("[CUDA]Device%d-FP64 Cores: %d Per MultiProcessor\n", i, 32);
			printf("[CUDA]Device%d-INT32 Cores: %d Per MultiProcessor\n", i, 64);
			printf("[CUDA]Device%d-Mixed Precision Tensor Cores: %d Per MultiProcessor\n", i, 8);
		}

		printf("[CUDA]Device%d-MultiProcessor Count: %d\n", i, cuda_prop->multiProcessorCount);
		printf("[CUDA]Device%d-Threads In Warp: %d\n", i, cuda_prop->warpSize);
		printf("[CUDA]Device%d-Max Registers Per Block: %d\n", i, cuda_prop->regsPerBlock);
		printf("[CUDA]Device%d-Max Shared Memory Per Block: %.2f KB\n", i, cuda_prop->sharedMemPerBlock / 1024.0f);
		printf("[CUDA]Device%d-Max Threads Per Block: %d\n", i, cuda_prop->maxThreadsPerBlock);
		printf("[CUDA]Device%d-Max Threads Per MultiProcessor: %d\n", i, cuda_prop->maxThreadsPerMultiProcessor);
		printf("[CUDA]Device%d-Max Thread Dimensions: (%d, %d, %d)\n", i, cuda_prop->maxThreadsDim[0], cuda_prop->maxThreadsDim[1], cuda_prop->maxThreadsDim[2]);
		printf("[CUDA]Device%d-Max Grid Dimensions: (%d, %d, %d)\n", i, cuda_prop->maxGridSize[0], cuda_prop->maxGridSize[1], cuda_prop->maxGridSize[2]);

		if (cuda_prop->major >= 6)
		{
			device_index = i;
			has_supported_device = true;
			break;
		}
	}

	if (!has_supported_device)
	{
		std::cout << "[Error]Could not find supported device, Compute Capability must >= 6!" << std::endl;
		return false;
	}

	CUDA_CALL(cudaSetDevice(device_index));
	if (!device_status::init(device_index))
	{
		std::cout << "[Error]Could not initialize NVML." << std::endl;
		return false;
	}
	return true;
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

void release_all()
{
	config->release_config_device_data();

	SAFE_DELETE(view_cam);
	SAFE_DELETE(render_cam);
	SAFE_DELETE(pt);
	SAFE_DELETE(config);
	SAFE_DELETE(cuda_prop);

	SAFE_DELETE_ARRAY(image_pixels_256_buffer);

	for (auto i = 0; i < scene_num; i++)
	{
		SAFE_DELETE_ARRAY(scene_files[i]);
	}
	SAFE_DELETE_ARRAY(scene_files);
}

bool render()
{
	view_cam->get_render_camera(render_cam);
	image* render_image = pt->render();

	if (render_image == nullptr)
	{ 
		return false;
	}

	CUDA_CALL(cudaMemcpy(image_pixels_256_buffer, render_image->pixels_256_device, render_image->pixel_count * sizeof(color256), cudaMemcpyDefault));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_pixels_256_buffer);

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

void render_choose_scene()
{
	if (scene_num == 0)
	{
		return;
	}

	ImGui_ImplGlfwGL3_NewFrame();
	ImGui::Begin("Choose Render Scene", 0, ImGuiWindowFlags_AlwaysAutoResize);

	ImGui::ListBox("", &scene_index, scene_files, scene_num);
	if (ImGui::Button("OK"))
	{
		if (pt->init_scene_device_data(scene_index))
		{
			is_scene_choose = true;
		}
	}

	ImGui::SameLine();
	if (ImGui::Button("Exit(Esc)"))
	{
		exit(0);
	}

	ImGui::End();
}

void render_ui(image* render_image)
{
	char buffer[2048];

	ImGui_ImplGlfwGL3_NewFrame();

	ImGui::Begin("Info(F1)", 0, ImGuiWindowFlags_AlwaysAutoResize);

	sprintf(buffer, "Scene: %s", scene_files[scene_index]);
	ImGui::Text(buffer);

	current_fps = get_fps(render_image);
	sprintf(buffer, "FPS: %.2f", current_fps);
	ImGui::Text(buffer);

	fps_deque.push_back(current_fps);
	if (fps_deque.size() >= 60)
	{
		fps_deque.pop_front();
	}

	float fps_datas[60] = { 0.0f };
	for (auto i = 0; i < fps_deque.size(); i++)
	{
		fps_datas[i] = fps_deque[i];
	}
	ImGui::PlotLines("", fps_datas, static_cast<int>(fps_deque.size()), 0, 0, 0.0f, 60.0f, { 320, 100 });

	sprintf(buffer, "Frame Time: %.0f ms", 1000.0f / current_fps);
	ImGui::Text(buffer);

	current_iteration = render_image->pass_counter;
	sprintf(buffer, "Iteration: %d", current_iteration);
	ImGui::Text(buffer);

	current_render_time = get_elapsed_second(render_image);
	sprintf(buffer, "Render Time: %.2f sec", current_render_time);
	ImGui::Text(buffer);

	device_status::render_ui(250);

	if (ImGui::CollapsingHeader("Options"))
	{
		configuration* conf = config->get_config_device_ptr();
		bool is_config_modified = false;
		is_config_modified |= ImGui::SliderInt("Block Size", &conf->block_size, 1, conf->max_block_size < cuda_prop->maxThreadsPerBlock ? conf->max_block_size : cuda_prop->maxThreadsPerBlock);
		is_config_modified |= ImGui::SliderInt("Trace Depth", &conf->max_tracer_depth, 0, 100);
		is_config_modified |= ImGui::DragFloat("Bias Length", &conf->vector_bias_length, 0.000001f, 0.0f, 1.0f, "%.6f");
		is_config_modified |= ImGui::DragFloat("Energy Threshold", &conf->energy_exist_threshold, 0.000001f, 0.0f, 1.0f, "%.6f");
		is_config_modified |= ImGui::DragFloat("SSS Threshold", &conf->sss_threshold, 0.000001f, 0.0f, 1.0f, "%.6f");

		is_config_modified |= ImGui::Checkbox("Skybox", &conf->use_sky_box);
		is_config_modified |= ImGui::Checkbox("Bilinear Sample", &conf->use_bilinear);
		is_config_modified |= ImGui::Checkbox("Sky", &conf->use_sky);
		is_config_modified |= ImGui::Checkbox("Gamma Correction", &conf->gamma_correction);
		is_config_modified |= ImGui::Checkbox("Anti Alias", &conf->use_anti_alias);

		if (is_config_modified)
		{
			pt->clear();
		}
	}

	if (ImGui::CollapsingHeader("Operation"))
	{
		if (ImGui::Button("Screenshot(P)"))
		{
			screenshot();
		}

		ImGui::SameLine();
		if (ImGui::Button("Clear(Space)"))
		{
			init_camera();
			pt->clear();
		}

		if (ImGui::Button("Exit(Esc)"))
		{
			exit(0);
		}

		ImGui::SameLine();
		if (ImGui::Button("Choose Scene(L)"))
		{
			pt->release_scene_device_data();
			pt->clear();
			is_scene_choose = false;
			scene_index = 0;
		}
	}

	if (ImGui::CollapsingHeader("Camera"))
	{
		sprintf(buffer, "Distance: %f", view_cam->get_radius());
		ImGui::Text(buffer);

		float focal_distance = view_cam->get_focal_distance();
		if (ImGui::SliderFloat("Focal Distance(Up,Down)", &focal_distance, 0.0f, view_cam->get_max_focal_distance()))
		{
			view_cam->set_focal_distance(focal_distance);
			pt->clear();
		}
		
		float aperture_radius = view_cam->get_aperture_radius();
		if (ImGui::DragFloat("Aperture Radius(Left,Right)", &aperture_radius, 0.00001f, 0.0f, view_cam->get_max_aperture_radius(), "%.5f"))
		{
			view_cam->set_aperture_radius(aperture_radius);
			pt->clear();
		}

		if (ImGui::DragFloat("FOV", &view_cam->fov.x, 0.1f, 0.1f, 179.9f, "%.1f"))
		{
			view_cam->set_fov(view_cam->fov.x);
			pt->clear();
		}
	}

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
	if (key == GLFW_KEY_F1 && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		is_info_window_show = !is_info_window_show;
		return;
	}

	if (key == GLFW_KEY_F2 && action == GLFW_PRESS)
	{
		config->get_config_device_ptr()->cuda_acceleration = !config->get_config_device_ptr()->cuda_acceleration;

		char title[2048];
		sprintf(title, "CUDA Path Tracer(%d*%d) CUDA Acceleration: %s(F2) - Yuming Zhou", width, height, config->get_config_device_ptr()->cuda_acceleration ? "On" : "Off");
		glfwSetWindowTitle(window, title);
		return;
	}

	if (key == GLFW_KEY_P && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		screenshot();
		return;
	}

	if (key == GLFW_KEY_UP && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		view_cam->modify_focal_distance(0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		view_cam->modify_focal_distance(-0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		view_cam->modify_aperture_radius(-0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		view_cam->modify_aperture_radius(0.001f);
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS && is_scene_choose)
	{
		ImGui::SetWindowFocus();
		init_camera();
		pt->clear();
		return;
	}

	if (key == GLFW_KEY_L && action == GLFW_PRESS && is_scene_choose)
	{
		pt->release_scene_device_data();
		pt->clear();
		is_scene_choose = false;
		scene_index = 0;
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

std::string screenshot()
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

	char buffer[2048];
	sprintf(buffer, "FrameTime-%dms_SSP-%d_RenderTime-%ds_%dX%d.png", static_cast<int>(1000.0f / current_fps), current_iteration, static_cast<int>(current_render_time), width, height);
	unsigned error = lodepng::encode(buffer, pixel_buffer, static_cast<uint>(width), static_cast<uint>(height));
	if (error != 0)
	{
		std::cout << "[Error]Screenshot fail : " << lodepng_error_text(error) << std::endl;
		return "";
	}
	else
	{
		std::cout << "[Info]Screenshot success, file name " << buffer << std::endl;
		return buffer;
	}
}