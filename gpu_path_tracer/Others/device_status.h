#pragma once

#ifndef __DEVICE_STATUS__
#define __DEVICE_STATUS__

#include <string>
#include <nvml.h>
#include <ctime>

#include "Math\basic_math.hpp"
#include "Math\cuda_math.hpp"
#include "Others\utilities.hpp"
#include "lib\imgui\imgui.h"
#include "lib\imgui\imgui_impl_glfw_gl3.h"

class device_status
{
public:
	static bool init(int device_index);
	static void tick();
	static void render_ui(double tick_interval);

	static std::string device_name;

	static nvmlUtilization_st device_utilization;
	static bool device_utilization_supported;

	static nvmlMemory_t  device_memory;
	static bool device_memory_supported;

	/*
		BAR1 is used to map the FB(device memory) 
		so that it can be directly accessed by the 
		CPU or by 3rd party devices(peer - to - 
		peer on the PCIE bus).
	*/
	static nvmlBAR1Memory_t  device_bar1_memory;
	static bool device_bar1_memory_supported;
	
	static uint device_fan_speed;
	static bool device_fan_speed_supported;
	
	static uint device_temperature;
	static bool device_temperature_supported;
	
	static uint device_slowdown_temperature;
	static bool device_slowdown_temperature_supported;

	static uint device_shutdown_temperature;
	static bool device_shutdown_temperature_supported;

	static nvmlComputeMode_t device_compute_mode;
	static bool device_compute_mode_supported;

	static uint device_graphics_clock;
	static bool device_graphics_clock_supported;
	static uint device_graphics_max_clock;
	static bool device_graphics_max_clock_supported;

	static uint device_sm_clock;
	static bool device_sm_clock_supported;
	static uint device_sm_max_clock;
	static bool device_sm_max_clock_supported;

	static uint device_memory_clock;
	static bool device_memory_clock_supported;
	static uint device_memory_max_clock;
	static bool device_memory_max_clock_supported;

	static bool is_render_ui;

private:
	static nvmlDevice_t device;
	static bool is_initialized;

	static const char* compute_mode_convert_to_string(nvmlComputeMode_t compute_mode)
	{
		switch (compute_mode)
		{
		case NVML_COMPUTEMODE_DEFAULT:
			return "Default";
		case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
			return "Exclusive_Thread";
		case NVML_COMPUTEMODE_PROHIBITED:
			return "Prohibited";
		case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
			return "Exclusive Process";
		default:
			return "Unknown";
		}
	}
};

#endif // !__DEVICE_STATUS__
