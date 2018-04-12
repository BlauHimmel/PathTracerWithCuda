#include "Core\cube_map_loader.h"

bool cube_map_loader::load_data(
	const std::string& filename_x_positive,
	const std::string& filename_x_negative,
	const std::string& filename_y_positive,
	const std::string& filename_y_negative,
	const std::string& filename_z_positive,
	const std::string& filename_z_negative
)
{
	std::vector<uchar> bmp_buffer;
	uint width, height;

	std::cout << "[Info]Loading file " << filename_x_positive << "...." << std::endl;
	if (!image_loader::load_image(filename_x_positive, m_width, m_height, m_x_positive_map) || m_width != m_height)
	{
		m_x_positive_map.clear();
		m_x_positive_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_x_positive << " succeeded." << std::endl;


	std::cout << "[Info]Loading file " << filename_x_negative << "...." << std::endl;
	if (!image_loader::load_image(filename_x_negative, width, height, m_x_negative_map) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_x_negative << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_y_positive << "...." << std::endl;
	if (!image_loader::load_image(filename_y_positive, width, height, m_y_positive_map) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_y_positive << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_y_negative << "...." << std::endl;
	if (!image_loader::load_image(filename_y_negative, width, height, m_y_negative_map) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_y_negative_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		m_y_negative_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_y_negative << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_z_positive << "...." << std::endl;
	if (!image_loader::load_image(filename_z_positive, width, height, m_z_positive_map) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_y_negative_map.clear();
		m_z_positive_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		m_y_negative_map.shrink_to_fit();
		m_z_positive_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_z_positive << " succeeded." << std::endl;

	std::cout << "[Info]Loading file " << filename_z_negative << "...." << std::endl;
	if (!image_loader::load_image(filename_z_negative, width, height, m_z_negative_map) || width != height || width != m_width || height != m_height)
	{
		m_x_positive_map.clear();
		m_x_negative_map.clear();
		m_y_positive_map.clear();
		m_y_negative_map.clear();
		m_z_positive_map.clear();
		m_z_negative_map.clear();
		m_x_positive_map.shrink_to_fit();
		m_x_negative_map.shrink_to_fit();
		m_y_positive_map.shrink_to_fit();
		m_y_negative_map.shrink_to_fit();
		m_z_positive_map.shrink_to_fit();
		m_z_negative_map.shrink_to_fit();
		return false;
	}
	std::cout << "[Info]Load file " << filename_z_negative << " succeeded." << std::endl;

	m_is_loaded = true;
	return true;
}

void cube_map_loader::unload_data()
{
	m_x_positive_map.clear();
	m_x_negative_map.clear();
	m_y_positive_map.clear();
	m_y_negative_map.clear();
	m_z_positive_map.clear();
	m_z_negative_map.clear();
	m_x_positive_map.shrink_to_fit();
	m_x_negative_map.shrink_to_fit();
	m_y_positive_map.shrink_to_fit();
	m_y_negative_map.shrink_to_fit();
	m_z_positive_map.shrink_to_fit();
	m_z_negative_map.shrink_to_fit();
	m_width = 0;
	m_height = 0;
	m_is_loaded = false;
}

cube_map * cube_map_loader::get_cube_map_device() const
{
	return m_cube_map_device;
}

bool cube_map_loader::create_cube_device_data()
{
	if (!m_is_loaded)
	{
		return false;
	}

	printf("[Info]Copy cube map data to gpu...\n");
	double time;
	TIME_COUNT_CALL_START();

	CUDA_CALL(cudaMallocManaged((void**)&m_cube_map_device, sizeof(cube_map)));

	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_x_positive_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_x_negative_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_y_positive_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_y_negative_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_z_positive_map), m_width * m_height * 4 * sizeof(uchar)));
	CUDA_CALL(cudaMallocManaged((void**)&(m_cube_map_device->m_z_negative_map), m_width * m_height * 4 * sizeof(uchar)));

	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_x_positive_map, m_x_positive_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_x_negative_map, m_x_negative_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_y_positive_map, m_y_positive_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_y_negative_map, m_y_negative_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_z_positive_map, m_z_positive_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));
	CUDA_CALL(cudaMemcpy(m_cube_map_device->m_z_negative_map, m_z_negative_map.data(), m_width * m_height * 4 * sizeof(uchar), cudaMemcpyDefault));

	m_cube_map_device->length = m_width;

	TIME_COUNT_CALL_END(time);
	printf("[Info]Completed, time consuming: %.4f ms\n", time);

	return true;
}

void cube_map_loader::release_cube_device_data()
{
	if (m_cube_map_device == nullptr)
	{
		return;
	}

	CUDA_CALL(cudaFree(m_cube_map_device->m_x_positive_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_x_negative_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_y_positive_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_y_negative_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_z_positive_map));
	CUDA_CALL(cudaFree(m_cube_map_device->m_z_negative_map));
	CUDA_CALL(cudaFree(m_cube_map_device));

	m_cube_map_device = nullptr;

	return;
}