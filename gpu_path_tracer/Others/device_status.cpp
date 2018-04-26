#include "Others\device_status.h"

bool device_status::init(int device_index)
{
	char buffer[256];
	NVML_CALL(nvmlInit());
	NVML_CALL(nvmlDeviceGetHandleByIndex(device_index, &device));
	NVML_CALL(nvmlDeviceGetName(device, buffer, 256));
	device_name = buffer;
	is_initialized = true;
	return true;
}

void device_status::tick()
{
	nvmlReturn_t error;

	error = nvmlDeviceGetComputeMode(device, &device_compute_mode);
	NVML_SUPPORTED(device_compute_mode);

	error = nvmlDeviceGetUtilizationRates(device, &device_utilization);
	NVML_SUPPORTED(device_utilization);

	error = nvmlDeviceGetMemoryInfo(device, &device_memory);
	NVML_SUPPORTED(device_memory);

	error = nvmlDeviceGetBAR1MemoryInfo(device, &device_bar1_memory);
	NVML_SUPPORTED(device_bar1_memory);

	error = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &device_shutdown_temperature);
	NVML_SUPPORTED(device_shutdown_temperature);

	error = nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &device_slowdown_temperature);
	NVML_SUPPORTED(device_slowdown_temperature);

	error = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &device_temperature);
	NVML_SUPPORTED(device_temperature);

	error = nvmlDeviceGetFanSpeed(device, &device_fan_speed);
	NVML_SUPPORTED(device_fan_speed);

	error = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &device_graphics_clock);
	NVML_SUPPORTED(device_graphics_clock);
	error = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &device_graphics_max_clock);
	NVML_SUPPORTED(device_graphics_max_clock);

	error = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &device_sm_clock);
	NVML_SUPPORTED(device_sm_clock);
	error = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &device_sm_max_clock);
	NVML_SUPPORTED(device_sm_max_clock);

	error = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &device_memory_clock);
	NVML_SUPPORTED(device_memory_clock);
	error = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &device_memory_max_clock);
	NVML_SUPPORTED(device_memory_max_clock);
}

void device_status::render_ui(double tick_interval)
{
	if (!is_initialized || !is_render_ui)
	{
		return;
	}

	static clock_t last_tick_time = clock();
	clock_t timer = clock();
	char buffer[2048];

	double elapsed = static_cast<double>(timer - last_tick_time) / CLOCKS_PER_SEC * 1000.0;
	if (elapsed >= tick_interval)
	{
		tick();
		last_tick_time = clock();
	}

	if (ImGui::CollapsingHeader("Device"))
	{
		sprintf(buffer, "Name: %s", device_name.c_str());
		ImGui::Text(buffer);

		if (device_compute_mode_supported)
		{
			sprintf(buffer, "Compute Mode: %s", compute_mode_convert_to_string(device_compute_mode));
			ImGui::Text(buffer);
			ImGui::NewLine();
		}

		if (device_fan_speed_supported)
		{
			sprintf(buffer, "Fan Speed: %d Percent", device_fan_speed);
			ImGui::Text(buffer);
			ImGui::NewLine();
		}

		if (device_utilization_supported)
		{
			ImGui::Text("Utilization");

			sprintf(buffer, "GPU: %d Percent", device_utilization.gpu);
			ImGui::Text(buffer);

			sprintf(buffer, "Memory: %d Percent", device_utilization.memory);
			ImGui::Text(buffer);

			ImGui::NewLine();
		}

		if (device_memory_supported)
		{
			ImGui::Text("Memory");

			sprintf(buffer, "Total: %.2f MB", static_cast<float>(device_memory.total) / 1024.0f / 1024.0f);
			ImGui::Text(buffer);

			sprintf(buffer, "Used: %.2f MB", static_cast<float>(device_memory.used) / 1024.0f / 1024.0f);
			ImGui::Text(buffer);

			sprintf(buffer, "Free: %.2f MB", static_cast<float>(device_memory.free) / 1024.0f / 1024.0f);
			ImGui::Text(buffer);

			ImGui::NewLine();
		}

		if (device_bar1_memory_supported)
		{
			ImGui::Text("Bar1 Memory");

			sprintf(buffer, "Total: %.2f MB", static_cast<float>(device_bar1_memory.bar1Total) / 1024.0f / 1024.0f);
			ImGui::Text(buffer);

			sprintf(buffer, "Used: %.2f MB", static_cast<float>(device_bar1_memory.bar1Used) / 1024.0f / 1024.0f);
			ImGui::Text(buffer);

			sprintf(buffer, "Free: %.2f MB", static_cast<float>(device_bar1_memory.bar1Free) / 1024.0f / 1024.0f);
			ImGui::Text(buffer);

			ImGui::NewLine();
		}

		if (device_temperature_supported || device_slowdown_temperature_supported || device_shutdown_temperature_supported)
		{
			ImGui::Text("Temperature");

			if (device_temperature_supported)
			{
				sprintf(buffer, "Current: %d Celsius", device_temperature);
				ImGui::Text(buffer);
			}

			if (device_slowdown_temperature_supported)
			{
				sprintf(buffer, "Slowdown Threshold: %d Celsius", device_slowdown_temperature);
				ImGui::Text(buffer);
			}

			if (device_shutdown_temperature_supported)
			{
				sprintf(buffer, "Shutdown Threshold: %d Celsius", device_shutdown_temperature);
				ImGui::Text(buffer);
			}

			ImGui::NewLine();
		}

		if (device_graphics_clock_supported || device_sm_clock_supported || device_memory_clock_supported)
		{
			ImGui::Text("Clock");

			if (device_graphics_clock_supported)
			{
				sprintf(buffer, "Graphics: %d Mhz", device_graphics_clock);
				ImGui::Text(buffer);

				if (device_graphics_max_clock_supported)
				{
					ImGui::SameLine();
					sprintf(buffer, "Max: %d Mhz", device_graphics_max_clock);
					ImGui::Text(buffer);
				}
			}

			if (device_sm_clock_supported)
			{
				sprintf(buffer, "SM:       %d Mhz", device_sm_clock);
				ImGui::Text(buffer);

				if (device_sm_max_clock_supported)
				{
					ImGui::SameLine();
					sprintf(buffer, "Max: %d Mhz", device_sm_max_clock);
					ImGui::Text(buffer);
				}
			}

			if (device_memory_clock_supported)
			{
				sprintf(buffer, "Memory:   %d Mhz", device_memory_clock);
				ImGui::Text(buffer);

				if (device_memory_max_clock_supported)
				{
					ImGui::SameLine();
					sprintf(buffer, "Max: %d Mhz", device_memory_max_clock);
					ImGui::Text(buffer);
				}
			}

			ImGui::NewLine();
		}
	}
}

std::string device_status::device_name = "Null";

nvmlUtilization_st device_status::device_utilization = { 0 };
bool device_status::device_utilization_supported = false;

nvmlMemory_t  device_status::device_memory = { 0 };
bool device_status::device_memory_supported = false;

nvmlBAR1Memory_t  device_status::device_bar1_memory = { 0 };
bool device_status::device_bar1_memory_supported = false;

uint device_status::device_fan_speed = 0;
bool device_status::device_fan_speed_supported = false;

uint device_status::device_temperature = 0;
bool device_status::device_temperature_supported = false;

uint device_status::device_slowdown_temperature = 0;
bool device_status::device_slowdown_temperature_supported = false;

uint device_status::device_shutdown_temperature = 0;
bool device_status::device_shutdown_temperature_supported = false;

nvmlComputeMode_t device_status::device_compute_mode;
bool device_status::device_compute_mode_supported = false;

uint device_status::device_graphics_clock = 0;
bool device_status::device_graphics_clock_supported = false;
uint device_status::device_graphics_max_clock = 0;
bool device_status::device_graphics_max_clock_supported = false;

uint device_status::device_sm_clock = 0;
bool device_status::device_sm_clock_supported = false;
uint device_status::device_sm_max_clock = 0;
bool device_status::device_sm_max_clock_supported = false;

uint device_status::device_memory_clock = 0;
bool device_status::device_memory_clock_supported = false;
uint device_status::device_memory_max_clock = 0;
bool device_status::device_memory_max_clock_supported = false;

bool device_status::is_render_ui = true;

nvmlDevice_t device_status::device = { 0 };
bool device_status::is_initialized = false;
