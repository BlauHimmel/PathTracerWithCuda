#include "bvh.h"

bounding_box::bounding_box()
{

}

bounding_box::bounding_box(const float3& left_bottom, const float3& right_top) :
	left_bottom(left_bottom), right_top(right_top), centroid(0.5f * (right_top + left_bottom))
{

}

void bounding_box::expand_to_fit_box(const float3& other_left_bottom, const float3& other_right_top)
{
	left_bottom = fminf(left_bottom, other_left_bottom);
	right_top = fmaxf(right_top, other_right_top);
	centroid = 0.5f * (right_top + left_bottom);
}

void bounding_box::expand_to_fit_triangle(const float3& vertex0, const float3& vertex1, const float3& vertex2)
{
	left_bottom = fminf(left_bottom, make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z)));
	right_top = fmaxf(right_top, make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z)));
	centroid = 0.5f * (right_top + left_bottom);
}

void bounding_box::get_bounding_box(const float3& other_left_bottom, const float3& other_right_top)
{
	left_bottom = other_left_bottom;
	right_top = other_right_top;
	centroid = 0.5f * (right_top + left_bottom);
}

void bounding_box::get_bounding_box(const float3& vertex0, const float3& vertex1, const float3& vertex2)
{
	left_bottom = make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.y), min(vertex0.z, vertex1.z, vertex2.z));
	right_top = make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.y), max(vertex0.z, vertex1.z, vertex2.z));
	centroid = 0.5f * (right_top + left_bottom);
}

float bounding_box::get_surface_area()
{
	return 2.0f * (right_top.x - left_bottom.x) * (right_top.y - left_bottom.y) * (right_top.z - left_bottom.z);
}

float bounding_box::get_axis_length(int axis)
{
	if (axis == 0)	return right_top.x - left_bottom.x;
	else if (axis == 1) return right_top.y - left_bottom.y;
	else return right_top.z - left_bottom.z;
}

namespace bvh_naive_cpu
{

	INTERNAL_FUNC bvh_node_device get_bvh_node_device(bvh_node* node)
	{
		bvh_node_device node_device;
		node_device.box = node->box;
		node_device.is_leaf = node->is_leaf;
		node_device.triangle_indices = nullptr;
		node_device.next_node_index = -1;
		return node_device;
	}

	INTERNAL_FUNC void split_bounding_box(bvh_node* node, bounding_box* boxes, int start_index)
	{
		std::stack<bvh_node*> stack;
		stack.push(node);

		bvh_node* internals[3];
		internals[0] = new bvh_node[BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM];
		internals[1] = new bvh_node[BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM];
		internals[2] = new bvh_node[BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM];
		bool* is_box_init = new bool[BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM];

		while (!stack.empty())
		{
			bvh_node* current_node = stack.top();
			stack.pop();

			int split_axis;
			int split_internal_index;
			bounding_box split_box_left, split_box_right;
			int split_triangle_num_left, split_triangle_num_right;
			int split_divide_internal_num;

			float min_cost = INFINITY;

			//find the best partition
			for (auto axis = 0; axis < 3; axis++)
			{
				float axis_length = current_node->box.get_axis_length(axis);

				int divide_internal_num = min(static_cast<int>(BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM), static_cast<int>(current_node->triangle_indices.size()));
				float internal_length = axis_length / static_cast<float>(divide_internal_num);

				for (auto i = 0; i < divide_internal_num; i++)
				{
					is_box_init[i] = false;
					internals[axis][i].triangle_indices.clear();
				}

				for (auto triangle_index : current_node->triangle_indices)
				{
					triangle_index -= start_index;

					int internal_index = static_cast<int>((get(boxes[triangle_index].centroid, axis) - get(current_node->box.left_bottom, axis)) / internal_length);
					internal_index = internal_index >= divide_internal_num ? (divide_internal_num - 1) : (internal_index < 0 ? 0 : internal_index);

					if (!is_box_init[internal_index])
					{
						internals[axis][internal_index].box.get_bounding_box(boxes[triangle_index].left_bottom, boxes[triangle_index].right_top);
						is_box_init[internal_index] = true;
					}
					else
					{
						internals[axis][internal_index].box.expand_to_fit_box(boxes[triangle_index].left_bottom, boxes[triangle_index].right_top);
					}
					internals[axis][internal_index].triangle_indices.push_back(triangle_index + start_index);
				}

				for (auto i = 0; i < divide_internal_num; i++)
				{
					bounding_box box_left;
					for (auto j = 0; j < i; j++)
					{
						if (is_box_init[j])
						{
							box_left.get_bounding_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							break;
						}
					}

					size_t triangle_num_left = 0;

					for (auto j = 0; j < i; j++)
					{
						if (is_box_init[j])
						{
							box_left.expand_to_fit_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							triangle_num_left += internals[axis][j].triangle_indices.size();
						}
					}

					bounding_box box_right;
					for (auto j = i; j < divide_internal_num; j++)
					{
						if (is_box_init[j])
						{
							box_right.get_bounding_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							break;
						}
					}

					size_t triangle_num_right = 0;

					for (auto j = i; j < divide_internal_num; j++)
					{
						if (is_box_init[j])
						{
							box_right.expand_to_fit_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							triangle_num_right += internals[axis][j].triangle_indices.size();
						}
					}

					float cost = box_left.get_surface_area() * triangle_num_left + box_right.get_surface_area() * triangle_num_right;

					if (cost < min_cost && cost > 0.0f)
					{
						min_cost = cost;
						split_axis = axis;
						split_internal_index = i;
						split_box_left = box_left;
						split_box_right = box_right;
						split_triangle_num_left = static_cast<int>(triangle_num_left);
						split_triangle_num_right = static_cast<int>(triangle_num_right);
						split_divide_internal_num = divide_internal_num;
					}
				}
			}

			//build the subnode of current node
			if (split_triangle_num_left > 0)
			{
				bvh_node* left = new bvh_node();
				left->box = split_box_left;
				for (auto i = 0; i < split_internal_index; i++)
				{
					left->triangle_indices.insert(left->triangle_indices.end(), internals[split_axis][i].triangle_indices.begin(), internals[split_axis][i].triangle_indices.end());
				}
				if (split_triangle_num_left <= BVH_LEAF_NODE_TRIANGLE_NUM)
				{
					left->is_leaf = true;
					left->triangle_indices.resize(BVH_LEAF_NODE_TRIANGLE_NUM, -1);
				}
				else
				{
					left->is_leaf = false;
					stack.push(left);
				}
				current_node->left = left;
			}

			if (split_triangle_num_right > 0)
			{
				bvh_node* right = new bvh_node();
				right->box = split_box_right;
				for (auto i = split_internal_index; i < split_divide_internal_num; i++)
				{
					right->triangle_indices.insert(right->triangle_indices.end(), internals[split_axis][i].triangle_indices.begin(), internals[split_axis][i].triangle_indices.end());
				}
				if (split_triangle_num_right <= BVH_LEAF_NODE_TRIANGLE_NUM)
				{
					right->is_leaf = true;
					right->triangle_indices.resize(BVH_LEAF_NODE_TRIANGLE_NUM, -1);
				}
				else
				{
					right->is_leaf = false;
					stack.push(right);
				}
				current_node->right = right;
			}
		}

		SAFE_DELETE_ARRAY(internals[0]);
		SAFE_DELETE_ARRAY(internals[1]);
		SAFE_DELETE_ARRAY(internals[2]);
		SAFE_DELETE_ARRAY(is_box_init);
	}

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index)
	{
		if (triangle_num == 0)
		{
			return nullptr;
		}

		bvh_node* root_node = new bvh_node();
		root_node->box.get_bounding_box(triangles[0].vertex0, triangles[0].vertex1, triangles[0].vertex2);

		for (auto i = 0; i < triangle_num; i++)
		{
			root_node->box.expand_to_fit_triangle(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
			root_node->triangle_indices.push_back(i + start_index);
		}

		if (root_node->triangle_indices.size() <= BVH_LEAF_NODE_TRIANGLE_NUM)
		{
			root_node->is_leaf = true;
			return root_node;
		}

		bounding_box* boxes = new bounding_box[triangle_num];

		for (auto i = 0; i < triangle_num; i++)
		{
			boxes[i].get_bounding_box(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
		}

		split_bounding_box(root_node, boxes, start_index);

		SAFE_DELETE_ARRAY(boxes);

		return root_node;
	}

	API_ENTRY void release_bvh(bvh_node* root_node)
	{
		if (root_node != nullptr)
		{
			if (root_node->left == nullptr && root_node->right == nullptr)
			{
				SAFE_DELETE(root_node)
			}
			else
			{
				if (root_node->left != nullptr)
				{
					release_bvh(root_node->left);
				}
				if (root_node->right != nullptr)
				{
					release_bvh(root_node->right);
				}
				SAFE_DELETE(root_node);
			}
		}
	}

	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root)
	{
		std::stack<bvh_node*> stack;
		std::vector<int> bvh_leaf_node_triangles_index_vec_device;
		int traversal_index = 0;
		int node_num = 0;
		stack.push(root);
		while (!stack.empty())
		{
			bvh_node* current_node = stack.top();
			stack.pop();
			node_num++;

			current_node->traversal_index = traversal_index;
			traversal_index++;

			if (current_node->is_leaf)
			{
				bvh_leaf_node_triangles_index_vec_device.insert(bvh_leaf_node_triangles_index_vec_device.end(),
					current_node->triangle_indices.begin(), current_node->triangle_indices.end());
			}

			if (current_node->right != nullptr)
			{
				stack.push(current_node->right);
			}

			if (current_node->left != nullptr)
			{
				stack.push(current_node->left);
			}
		}

		int* leaf_node_triangle_indices;
		CUDA_CALL(cudaMallocManaged((void**)&leaf_node_triangle_indices, bvh_leaf_node_triangles_index_vec_device.size() * sizeof(int)));
		CUDA_CALL(cudaMemcpy(leaf_node_triangle_indices, bvh_leaf_node_triangles_index_vec_device.data(), bvh_leaf_node_triangles_index_vec_device.size() * sizeof(int), cudaMemcpyDefault));

		std::vector<bvh_node_device> bvh_nodes_vec_device(node_num);
		traversal_index = -1;
		int leaf_node_index = 0;
		stack.push(root);
		while (!stack.empty())
		{
			bvh_node* current_node = stack.top();
			stack.pop();
			traversal_index++;

			bvh_node_device node_device = get_bvh_node_device(current_node);

			if (current_node->is_leaf)
			{
				node_device.triangle_indices = leaf_node_triangle_indices + leaf_node_index * BVH_LEAF_NODE_TRIANGLE_NUM;
				leaf_node_index++;
			}

			if (stack.empty())
			{
				node_device.next_node_index = node_num;
			}
			else
			{
				node_device.next_node_index = stack.top()->traversal_index;
			}
			bvh_nodes_vec_device[traversal_index] = node_device;

			if (current_node->right != nullptr)
			{
				stack.push(current_node->right);
			}

			if (current_node->left != nullptr)
			{
				stack.push(current_node->left);
			}
		}

		bvh_node_device* bvh_nodes_array_device;
		CUDA_CALL(cudaMallocManaged((void**)&bvh_nodes_array_device, bvh_nodes_vec_device.size() * sizeof(bvh_node_device)));
		CUDA_CALL(cudaMemcpy(bvh_nodes_array_device, bvh_nodes_vec_device.data(), bvh_nodes_vec_device.size() * sizeof(bvh_node_device), cudaMemcpyDefault));
		return bvh_nodes_array_device;
	}
}

namespace bvh_morton_code_cpu
{	
	/*
		Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
		From from http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
	*/
	INTERNAL_FUNC uint expand_bits(uint value)
	{
		value = (value * 0x00010001u) & 0xFF0000FFu;
		value = (value * 0x00000101u) & 0x0F00F00Fu;
		value = (value * 0x00000011u) & 0xC30C30C3u;
		value = (value * 0x00000005u) & 0x49249249u;
		return value;
	}

	/*
		Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
		From from http://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
	*/
	INTERNAL_FUNC uint morton_code(const float3& point)
	{
		float x = fminf(fmaxf(point.x * 1024.0f, 0.0f), 1023.0f);
		float y = fminf(fmaxf(point.y * 1024.0f, 0.0f), 1023.0f);
		float z = fminf(fmaxf(point.z * 1024.0f, 0.0f), 1023.0f);

		uint xx = expand_bits(static_cast<uint>(x));
		uint yy = expand_bits(static_cast<uint>(y));
		uint zz = expand_bits(static_cast<uint>(z));

		return xx << 2 + yy << 1 + zz;
	}

	/*
		Counts the number of leading zero bits in a 32-bit integer.
		From : http://embeddedgurus.com/state-space/2014/09/fast-deterministic-and-portable-counting-leading-zeros/
	*/
	INTERNAL_FUNC uint clz(uint value)
	{
		static const uchar clz_table[] = 
		{
			32u, 31u, 30u, 30u, 29u, 29u, 29u, 29u,
			28u, 28u, 28u, 28u, 28u, 28u, 28u, 28u,
			27u, 27u, 27u, 27u, 27u, 27u, 27u, 27u,
			27u, 27u, 27u, 27u, 27u, 27u, 27u, 27u,
			26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
			26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
			26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
			26u, 26u, 26u, 26u, 26u, 26u, 26u, 26u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			25u, 25u, 25u, 25u, 25u, 25u, 25u, 25u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u,
			24u, 24u, 24u, 24u, 24u, 24u, 24u, 24u
		};

		uint n;

		if (value >= (1u << 16))
		{
			if (value >= (1u << 24)) 
			{
				n = 24u;
			}
			else 
			{
				n = 16u;
			}
		}
		else 
		{
			if (value >= (1u << 8)) 
			{
				n = 8u;
			}
			else 
			{
				n = 0u;
			}
		}

		return static_cast<uint>(clz_table[value >> n]) - n;
	}

	/*
		From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
	*/
	INTERNAL_FUNC uint find_split(uint* morton_codes, uint first_index, uint last_index)
	{
		uint first_morton_code = morton_codes[first_index];
		uint last_morton_code = morton_codes[last_index];

		if (first_morton_code == last_morton_code)
		{
			return first_index;
		}

		uint common_prefix_length = clz(first_morton_code ^ last_morton_code);
		uint split_index = first_index;
		uint step = last_index - first_index;

		do
		{
			step = (step + 1) >> 1;
			uint new_split_index = split_index + step;

			if (new_split_index < last_index)
			{
				uint split_morton_code = morton_codes[new_split_index];
				int split_prefix_length = clz(first_morton_code ^ split_morton_code);

				if (split_prefix_length > common_prefix_length)
				{
					split_index = new_split_index;
				}
			}


		} while (step > 1);

		return split_index;
	}

	/*
		From the paper: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees
	*/
	uint2 find_range(uint* morton_codes, uint morton_codes_size, uint index)
	{
		if (index == 0)
		{
			return make_uint2(0, morton_codes_size - 1);
		}

		int direction;
		uint delta_min;
		uint initial_index = index;

		uint previous_morton_code = morton_codes[index - 1];
		uint current_morton_code = morton_codes[index];
		uint next_morton_code = morton_codes[index + 1];

		if (previous_morton_code == current_morton_code && current_morton_code == next_morton_code)
		{
			while (index > 0 && index < morton_codes_size - 1)
			{
				index++;

				if (index >= morton_codes_size - 1)
				{
					break;
				}

				if (morton_codes[index] != morton_codes[index + 1])
				{
					break;
				}
			}

			return make_uint2(initial_index, index);
		}
		else
		{
			uint prefix_common_length_left = clz(current_morton_code ^ previous_morton_code);
			uint prefix_common_length_right = clz(current_morton_code ^ next_morton_code);

			if (prefix_common_length_left > prefix_common_length_right)
			{
				direction = -1;
				delta_min = prefix_common_length_right;
			}
			else
			{
				direction = 1;
				delta_min = prefix_common_length_left;
			}
		}

		uint l_max = 2;
		uint test_index = index + l_max * direction;

		while (test_index < morton_codes_size && test_index >= 0 &&
			clz(current_morton_code ^ morton_codes[test_index] > delta_min))
		{
			l_max *= 2;
			test_index = index + l_max * direction;
		}

		int l = 0;

		for (int divisor = 2; l_max / divisor >= 1; divisor *= 2)
		{
			int t = l_max / divisor;
			test_index = index + (l + t) * direction;

			if (test_index < morton_codes_size && test_index >= 0)
			{
				if (clz(current_morton_code ^ morton_codes[test_index]))
				{
					l = l + t;
				}
			}
		}

		if (direction == 1)
		{
			return make_uint2(index, index + l * direction);
		}
		else/* if (direction == -1) */
		{
			return make_uint2(index + l * direction, index);
		}
	}
}