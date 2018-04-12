#include "Bvh\bvh.h"

int bvh_build_config::bvh_leaf_node_triangle_num = 1;
int bvh_build_config::bvh_bucket_max_divide_internal_num = 12;
int bvh_build_config::bvh_build_block_size = 32;

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
		internals[0] = new bvh_node[bvh_build_config::bvh_bucket_max_divide_internal_num];
		internals[1] = new bvh_node[bvh_build_config::bvh_bucket_max_divide_internal_num];
		internals[2] = new bvh_node[bvh_build_config::bvh_bucket_max_divide_internal_num];
		bool* is_box_init = new bool[bvh_build_config::bvh_bucket_max_divide_internal_num];

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
			for (int axis = 0; axis < 3; axis++)
			{
				float axis_length = current_node->box.get_axis_length(axis);

				int divide_internal_num = min(bvh_build_config::bvh_bucket_max_divide_internal_num, static_cast<int>(current_node->triangle_indices.size()));
				float internal_length = axis_length / static_cast<float>(divide_internal_num);

				for (int i = 0; i < divide_internal_num; i++)
				{
					is_box_init[i] = false;
					internals[axis][i].triangle_indices.clear();
				}

				for (int triangle_index : current_node->triangle_indices)
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

				for (int i = 0; i < divide_internal_num; i++)
				{
					bounding_box box_left;
					for (int j = 0; j < i; j++)
					{
						if (is_box_init[j])
						{
							box_left.get_bounding_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							break;
						}
					}

					size_t triangle_num_left = 0;

					for (int j = 0; j < i; j++)
					{
						if (is_box_init[j])
						{
							box_left.expand_to_fit_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							triangle_num_left += internals[axis][j].triangle_indices.size();
						}
					}

					bounding_box box_right;
					for (int j = i; j < divide_internal_num; j++)
					{
						if (is_box_init[j])
						{
							box_right.get_bounding_box(internals[axis][j].box.left_bottom, internals[axis][j].box.right_top);
							break;
						}
					}

					size_t triangle_num_right = 0;

					for (int j = i; j < divide_internal_num; j++)
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
				for (int i = 0; i < split_internal_index; i++)
				{
					left->triangle_indices.insert(left->triangle_indices.end(), internals[split_axis][i].triangle_indices.begin(), internals[split_axis][i].triangle_indices.end());
				}
				if (split_triangle_num_left <= bvh_build_config::bvh_leaf_node_triangle_num)
				{
					left->is_leaf = true;
					left->triangle_indices.resize(bvh_build_config::bvh_leaf_node_triangle_num, -1);
				}
				else
				{
					left->is_leaf = false;
					stack.push(left);
				}
				current_node->left = left;
				left->parent = current_node;
			}

			if (split_triangle_num_right > 0)
			{
				bvh_node* right = new bvh_node();
				right->box = split_box_right;
				for (int i = split_internal_index; i < split_divide_internal_num; i++)
				{
					right->triangle_indices.insert(right->triangle_indices.end(), internals[split_axis][i].triangle_indices.begin(), internals[split_axis][i].triangle_indices.end());
				}
				if (split_triangle_num_right <= bvh_build_config::bvh_leaf_node_triangle_num)
				{
					right->is_leaf = true;
					right->triangle_indices.resize(bvh_build_config::bvh_leaf_node_triangle_num, -1);
				}
				else
				{
					right->is_leaf = false;
					stack.push(right);
				}
				current_node->right = right;
				right->parent = current_node;
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

		for (int i = 0; i < triangle_num; i++)
		{
			root_node->box.expand_to_fit_triangle(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
			root_node->triangle_indices.push_back(i + start_index);
		}

		if (root_node->triangle_indices.size() <= bvh_build_config::bvh_leaf_node_triangle_num)
		{
			root_node->is_leaf = true;
			return root_node;
		}

		bounding_box* boxes = new bounding_box[triangle_num];

		for (int i = 0; i < triangle_num; i++)
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
		std::vector<int> bvh_leaf_node_triangles_index_vector;
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
				bvh_leaf_node_triangles_index_vector.insert(bvh_leaf_node_triangles_index_vector.end(),
					current_node->triangle_indices.begin(), current_node->triangle_indices.end());
			
				if (current_node->box.is_thin_bounding_box())
				{
					current_node->box.expand_to_fit_box(current_node->parent->box.left_bottom, current_node->parent->box.right_top);
				}
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
		CUDA_CALL(cudaMallocManaged((void**)&leaf_node_triangle_indices, bvh_leaf_node_triangles_index_vector.size() * sizeof(int)));
		CUDA_CALL(cudaMemcpy(leaf_node_triangle_indices, bvh_leaf_node_triangles_index_vector.data(), bvh_leaf_node_triangles_index_vector.size() * sizeof(int), cudaMemcpyDefault));

		std::vector<bvh_node_device> bvh_nodes_device_vector(node_num);
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
				node_device.triangle_indices = leaf_node_triangle_indices + leaf_node_index * bvh_build_config::bvh_leaf_node_triangle_num;
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
			bvh_nodes_device_vector[traversal_index] = node_device;

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
		CUDA_CALL(cudaMallocManaged((void**)&bvh_nodes_array_device, 2 * bvh_nodes_device_vector.size() * sizeof(bvh_node_device)));
		CUDA_CALL(cudaMemcpy(bvh_nodes_array_device, bvh_nodes_device_vector.data(), bvh_nodes_device_vector.size() * sizeof(bvh_node_device), cudaMemcpyDefault));
		CUDA_CALL(cudaMemcpy(bvh_nodes_array_device + bvh_nodes_device_vector.size(), bvh_nodes_device_vector.data(), bvh_nodes_device_vector.size() * sizeof(bvh_node_device), cudaMemcpyDefault));
		return bvh_nodes_array_device;
	}

	API_ENTRY void update_bvh(
		const glm::mat4& initial_transform_mat,
		const glm::mat4& transform_mat,
		bvh_node_device* initial_root,
		bvh_node_device* transformed_root
	)
	{
		int node_num = initial_root->next_node_index;

		for (auto i = 0; i < node_num; i++)
		{
			float3 left_bottom = initial_root[i].box.left_bottom;
			float3 right_top = initial_root[i].box.right_top;

			glm::vec4 left_bottom_vec4 = glm::vec4(left_bottom.x, left_bottom.y, left_bottom.z, 1.0f);
			glm::vec4 right_top_vec4 = glm::vec4(right_top.x, right_top.y, right_top.z, 1.0f);

			left_bottom_vec4 = transform_mat * initial_transform_mat * left_bottom_vec4;
			right_top_vec4 = transform_mat * initial_transform_mat * right_top_vec4;

			transformed_root[i].box.left_bottom = make_float3(left_bottom_vec4.x, left_bottom_vec4.y, left_bottom_vec4.z);
			transformed_root[i].box.right_top = make_float3(right_top_vec4.x, right_top_vec4.y, right_top_vec4.z);
			transformed_root[i].box.centroid = 0.5f * (transformed_root[i].box.left_bottom + transformed_root[i].box.right_top);
		}
	}
}

namespace bvh_morton_code_cpu
{
	bool bvh_node_morton_node_comparator(const bvh_node& left, const bvh_node& right)
	{
		return left.morton_code < right.morton_code;
	}

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

		return xx * 4 + yy * 2 + zz;
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
	INTERNAL_FUNC uint find_split(bvh_node* sorted_leaf_nodes, uint first_index, uint last_index)
	{
		uint first_morton_code = sorted_leaf_nodes[first_index].morton_code;
		uint last_morton_code = sorted_leaf_nodes[last_index].morton_code;

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
				uint split_morton_code = sorted_leaf_nodes[new_split_index].morton_code;
				uint split_prefix_length = clz(first_morton_code ^ split_morton_code);

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
	INTERNAL_FUNC uint2 find_range(bvh_node* sorted_leaf_nodes, uint node_num, uint index)
	{
		if (index == 0)
		{
			return make_uint2(0, node_num - 1);
		}

		int direction;
		uint delta_min;
		uint initial_index = index;

		uint previous_morton_code = sorted_leaf_nodes[index - 1].morton_code;
		uint current_morton_code = sorted_leaf_nodes[index].morton_code;
		uint next_morton_code = sorted_leaf_nodes[index + 1].morton_code;

		if (previous_morton_code == current_morton_code && current_morton_code == next_morton_code)
		{
			while (index > 0 && index < node_num - 1)
			{
				index++;

				if (index >= node_num - 1)
				{
					break;
				}

				if (sorted_leaf_nodes[index].morton_code != sorted_leaf_nodes[index + 1].morton_code)
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

		while (test_index < node_num && test_index >= 0 &&
			clz(current_morton_code ^ sorted_leaf_nodes[test_index].morton_code) > delta_min)
		{
			l_max *= 2;
			test_index = index + l_max * direction;
		}

		int l = 0;

		for (int divisor = 2; l_max / divisor >= 1; divisor *= 2)
		{
			int t = l_max / divisor;
			test_index = index + (l + t) * direction;

			if (test_index < node_num && test_index >= 0)
			{
				if (clz(current_morton_code ^ sorted_leaf_nodes[test_index].morton_code) > delta_min)
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

	INTERNAL_FUNC void generate_internal_node(bvh_node* internal_nodes, bvh_node* leaf_nodes, uint leaf_node_num, uint index)
	{
		uint2 range = find_range(leaf_nodes, leaf_node_num, index);
		uint split_index = find_split(leaf_nodes, range.x, range.y);

		bvh_node* left;
		bvh_node* right;

		if (split_index == range.x)
		{
			left = &leaf_nodes[split_index];
		}
		else
		{
			left = &internal_nodes[split_index];
		}

		if (split_index + 1 == range.y)
		{
			right = &leaf_nodes[split_index + 1];
		}
		else
		{
			right = &internal_nodes[split_index + 1];
		}

		left->parent = &internal_nodes[index];
		right->parent = &internal_nodes[index];

		internal_nodes[index].left = left;
		internal_nodes[index].right = right;
	}

	INTERNAL_FUNC void generate_bounding_box_for_internal_node(bvh_node* node)
	{
		if (node->is_leaf)
		{
			if (node->parent != nullptr)
			{
				generate_bounding_box_for_internal_node(node->parent);
			}
		}
		else
		{
			bool is_ret = false;

			#ifdef BVH_MORTON_CODE_BUILD_OPENMP
			#pragma omp critical(node)
			#endif
			{
				if (node->is_visited == 0)
				{
					node->is_visited = 1;
					is_ret = true;
				}
			}

			if (is_ret)
			{
				return;
			}

			node->box.get_bounding_box(node->left->box.left_bottom, node->left->box.right_top);
			node->box.expand_to_fit_box(node->right->box.left_bottom, node->right->box.right_top);

			if (node->parent != nullptr)
			{
				generate_bounding_box_for_internal_node(node->parent);
			}
		}
	}

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index)
	{
		static int threadnum = omp_get_max_threads();

		uint leaf_node_num = triangle_num / bvh_build_config::bvh_leaf_node_triangle_num;
		leaf_node_num = ((leaf_node_num == 0) ? 1 : leaf_node_num);
		leaf_node_num = ((triangle_num % bvh_build_config::bvh_leaf_node_triangle_num == 0) ? leaf_node_num : (leaf_node_num + 1));

		uint internal_node_num = leaf_node_num - 1;

		bvh_node* triangle_nodes = new bvh_node[triangle_num];
		bvh_node* leaf_nodes = new bvh_node[leaf_node_num];
		bvh_node* internal_nodes = new bvh_node[internal_node_num];

		//Compute the total bounding box of the given mesh
		internal_nodes[0].box.get_bounding_box(triangles[0].vertex0, triangles[0].vertex1, triangles[0].vertex2);
		for (int i = 1; i < triangle_num; i++)
		{
			internal_nodes[0].box.expand_to_fit_triangle(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
		}

		//Compute the bounding box of each triangle on the given mesh
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < triangle_num; i++)
		{
			triangle_nodes[i].triangle_index = i + start_index;
			triangle_nodes[i].box.get_bounding_box(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
			triangle_nodes[i].morton_code = morton_code(
				(triangle_nodes[i].box.centroid - internal_nodes[0].box.left_bottom) / 
				(internal_nodes[0].box.right_top - internal_nodes [0].box.left_bottom)
			);
		}

		//Sort the bvh_node according to the morton code of its centroid
		std::sort(triangle_nodes, triangle_nodes + triangle_num, bvh_node_morton_node_comparator);

		//Batch the bvh_node(each batch contains BVH_LEAF_NODE_TRIANGLE_NUM original bvh_node)(Can be parallel)
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(leaf_node_num); i++)
		{
			int triangle_start_index = i * bvh_build_config::bvh_leaf_node_triangle_num;

			leaf_nodes[i].triangle_indices.resize(bvh_build_config::bvh_leaf_node_triangle_num, -1);

			leaf_nodes[i].is_leaf = true;
			leaf_nodes[i].box.get_bounding_box(
				triangle_nodes[triangle_start_index].box.left_bottom,
				triangle_nodes[triangle_start_index].box.right_top
			);
			leaf_nodes[i].triangle_indices[0] = triangle_nodes[triangle_start_index].triangle_index;

			for (int j = 1; j < bvh_build_config::bvh_leaf_node_triangle_num; j++)
			{
				if (j + triangle_start_index < triangle_num)
				{
					leaf_nodes[i].box.expand_to_fit_box(triangle_nodes[j + triangle_start_index].box.left_bottom, triangle_nodes[j + triangle_start_index].box.right_top);
					leaf_nodes[i].triangle_indices[j] = triangle_nodes[j + triangle_start_index].triangle_index;
				}
			}

			leaf_nodes[i].morton_code = morton_code(
				(leaf_nodes[i].box.centroid - internal_nodes[0].box.left_bottom) /
				(internal_nodes[0].box.right_top - internal_nodes[0].box.left_bottom)
			);
		}

		std::sort(leaf_nodes, leaf_nodes + leaf_node_num, bvh_node_morton_node_comparator);

		//Generate the tree(Can be parallel)
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(internal_node_num); i++)
		{
			generate_internal_node(internal_nodes, leaf_nodes, leaf_node_num, i);
		}

		//Generate bounding box for the internal node of the tree
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(leaf_node_num); i++)
		{
			generate_bounding_box_for_internal_node(&leaf_nodes[i]);
		}

		SAFE_DELETE_ARRAY(triangle_nodes);

		return &internal_nodes[0];
	}

	API_ENTRY void release_bvh(bvh_node* root_node)
	{
		bvh_node* leaf_nodes = root_node;
		while (!leaf_nodes->is_leaf)
		{
			leaf_nodes = leaf_nodes->left;
		}
		
		SAFE_DELETE_ARRAY(leaf_nodes);
		SAFE_DELETE_ARRAY(root_node);
	}

	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root)
	{
		return bvh_naive_cpu::build_bvh_device_data(root);
	}

	API_ENTRY void update_bvh(
		const glm::mat4& initial_transform_mat,
		const glm::mat4& transform_mat,
		bvh_node_device* initial_root,
		bvh_node_device* transformed_root
	)
	{
		return bvh_naive_cpu::update_bvh(initial_transform_mat, transform_mat, initial_root, transformed_root);
	}
}

namespace bvh_morton_code_cuda
{
	bool bvh_node_morton_node_comparator(const bvh_node_morton_code_cuda& left, const bvh_node_morton_code_cuda& right)
	{
		return left.morton_code < right.morton_code;
	}

	INTERNAL_FUNC void generate_bounding_box_for_internal_node(
		bvh_node_morton_code_cuda* node,
		bvh_node_morton_code_cuda* leaf_nodes,
		bvh_node_morton_code_cuda* internal_nodes
	)
	{
		if (node->is_leaf)
		{
			if (node->parent_index != -1)
			{
				generate_bounding_box_for_internal_node(&internal_nodes[node->parent_index], leaf_nodes, internal_nodes);
			}
		}
		else
		{
			bool is_ret = false;
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
			#pragma omp critical(node)
#endif
			{
				if (node->is_visited == 0)
				{
					node->is_visited = 1;
					is_ret = true;
				}
			}

			if (is_ret)
			{
				return;
			}

			bvh_node_morton_code_cuda* left = node->is_left_leaf ? &leaf_nodes[node->left_index] : &internal_nodes[node->left_index];
			bvh_node_morton_code_cuda* right = node->is_right_leaf ? &leaf_nodes[node->right_index] : &internal_nodes[node->right_index];

			node->box.get_bounding_box(left->box.left_bottom, left->box.right_top);
			node->box.expand_to_fit_box(right->box.left_bottom, right->box.right_top);

			if (node->parent_index != -1)
			{
				generate_bounding_box_for_internal_node(&internal_nodes[node->parent_index], leaf_nodes, internal_nodes);
			}
		}
	}

	API_ENTRY bvh_node* build_bvh(triangle* triangles, int triangle_num, int start_index)
	{
		static int threadnum = omp_get_max_threads();

		uint leaf_node_num = triangle_num / bvh_build_config::bvh_leaf_node_triangle_num;
		leaf_node_num = ((leaf_node_num == 0) ? 1 : leaf_node_num);
		leaf_node_num = ((triangle_num % bvh_build_config::bvh_leaf_node_triangle_num == 0) ? leaf_node_num : (leaf_node_num + 1));

		uint internal_node_num = leaf_node_num - 1;

		bvh_node_morton_code_cuda* triangle_morton_code_nodes_device = nullptr;
		bvh_node_morton_code_cuda* leaf_morton_code_nodes_device = nullptr;
		bvh_node_morton_code_cuda* internal_morton_code_nodes_device = nullptr;

		bvh_node_morton_code_cuda* triangle_morton_code_nodes = new bvh_node_morton_code_cuda[triangle_num];
		bvh_node_morton_code_cuda* leaf_morton_code_nodes = new bvh_node_morton_code_cuda[leaf_node_num];
		bvh_node_morton_code_cuda* internal_morton_code_nodes = new bvh_node_morton_code_cuda[internal_node_num];

		std::vector<int>* leaf_nodes_triangle_indices = new std::vector<int>[leaf_node_num];

 		CUDA_CALL(cudaMallocManaged((void**)&triangle_morton_code_nodes_device, triangle_num * sizeof(bvh_node_morton_code_cuda)));
		CUDA_CALL(cudaMallocManaged((void**)&leaf_morton_code_nodes_device, leaf_node_num * sizeof(bvh_node_morton_code_cuda)));
		CUDA_CALL(cudaMallocManaged((void**)&internal_morton_code_nodes_device, internal_node_num * sizeof(bvh_node_morton_code_cuda)));

		//Compute the total bounding box of the given mesh
		internal_morton_code_nodes_device[0].box.get_bounding_box(triangles[0].vertex0, triangles[0].vertex1, triangles[0].vertex2);
		internal_morton_code_nodes_device[0].parent_index = -1;
		for (int i = 1; i < triangle_num; i++)
		{
			internal_morton_code_nodes_device[0].box.expand_to_fit_triangle(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
		}

		//Compute the bounding box of each triangle on the given mesh
		compute_triangle_bounding_box_kernel(
			triangles,
			triangle_num,
			triangle_morton_code_nodes_device,
			&(internal_morton_code_nodes_device[0].box),
			start_index,
			bvh_build_config::bvh_build_block_size
		);

		CUDA_CALL(cudaMemcpy(triangle_morton_code_nodes, triangle_morton_code_nodes_device, triangle_num * sizeof(bvh_node_morton_code_cuda), cudaMemcpyDefault));

		//Sort the bvh_node according to the morton code of its centroid
		std::sort(triangle_morton_code_nodes, triangle_morton_code_nodes + triangle_num, bvh_node_morton_node_comparator);

		//Batch the bvh_node(each batch contains BVH_LEAF_NODE_TRIANGLE_NUM original bvh_node)
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(leaf_node_num); i++)
		{
			int triangle_start_index = i * bvh_build_config::bvh_leaf_node_triangle_num;

			leaf_nodes_triangle_indices[i].resize(bvh_build_config::bvh_leaf_node_triangle_num, -1);

			//triangle_index here is used to mark its corresponding index of leaf_nodes_triangle_indices
			//because leaf_morton_code_nodes will be sorted and its sequence no longer match the leaf_morton_code_nodes
			leaf_morton_code_nodes[i].triangle_index = i;	
			leaf_morton_code_nodes[i].is_leaf = true;
			leaf_morton_code_nodes[i].box.get_bounding_box(
				triangle_morton_code_nodes[triangle_start_index].box.left_bottom,
				triangle_morton_code_nodes[triangle_start_index].box.right_top
			);
			leaf_nodes_triangle_indices[i][0] = triangle_morton_code_nodes[triangle_start_index].triangle_index;

			for (int j = 1; j < bvh_build_config::bvh_leaf_node_triangle_num; j++)
			{
				if (j + triangle_start_index < triangle_num)
				{
					leaf_morton_code_nodes[i].box.expand_to_fit_box(
						triangle_morton_code_nodes[j + triangle_start_index].box.left_bottom,
						triangle_morton_code_nodes[j + triangle_start_index].box.right_top
					);
					leaf_nodes_triangle_indices[i][j] = triangle_morton_code_nodes[j + triangle_start_index].triangle_index;
				}
			}

			leaf_morton_code_nodes[i].morton_code = bvh_morton_code_cpu::morton_code(
				(leaf_morton_code_nodes[i].box.centroid - internal_morton_code_nodes_device[0].box.left_bottom) /
				(internal_morton_code_nodes_device[0].box.right_top - internal_morton_code_nodes_device[0].box.left_bottom)
			);
		}
		
		std::sort(leaf_morton_code_nodes, leaf_morton_code_nodes + leaf_node_num, bvh_node_morton_node_comparator);
		CUDA_CALL(cudaMemcpy(leaf_morton_code_nodes_device, leaf_morton_code_nodes, leaf_node_num * sizeof(bvh_node_morton_code_cuda), cudaMemcpyDefault));

		//Generate the tree
		generate_internal_node_kernel(internal_morton_code_nodes_device, internal_node_num, leaf_morton_code_nodes_device, leaf_node_num, bvh_build_config::bvh_build_block_size);

		CUDA_CALL(cudaMemcpy(leaf_morton_code_nodes, leaf_morton_code_nodes_device, leaf_node_num * sizeof(bvh_node_morton_code_cuda), cudaMemcpyDefault));
		CUDA_CALL(cudaMemcpy(internal_morton_code_nodes, internal_morton_code_nodes_device, internal_node_num * sizeof(bvh_node_morton_code_cuda), cudaMemcpyDefault));

		//Generate bounding box for the internal node of the tree
#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(leaf_node_num); i++)
		{
			generate_bounding_box_for_internal_node(&leaf_morton_code_nodes[i], leaf_morton_code_nodes, internal_morton_code_nodes);
		}

		bvh_node* leaf_nodes = new bvh_node[leaf_node_num];
		bvh_node* internal_nodes = new bvh_node[internal_node_num];

#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(leaf_node_num); i++)
		{
			leaf_nodes[i].box = leaf_morton_code_nodes[i].box;
			leaf_nodes[i].is_leaf = true;
			leaf_nodes[i].parent = &internal_nodes[leaf_morton_code_nodes[i].parent_index];
			//because the sequence of leaf_morton_code_nodes not match leaf_morton_code_nodes, so the index of leaf_nodes_triangle_indices
			//should be redirect
			leaf_nodes[i].triangle_indices = leaf_nodes_triangle_indices[leaf_morton_code_nodes[i].triangle_index];
		}

#ifdef BVH_MORTON_CODE_BUILD_OPENMP
		#pragma omp parallel for num_threads(threadnum) schedule(guided) 
#endif
		for (int i = 0; i < static_cast<int>(internal_node_num); i++)
		{
			internal_nodes[i].box = internal_morton_code_nodes[i].box;

			if (internal_morton_code_nodes[i].parent_index != -1)
			{
				internal_nodes[i].parent = &internal_nodes[internal_morton_code_nodes[i].parent_index];
			}

			if (internal_morton_code_nodes[i].is_left_leaf)
			{
				internal_nodes[i].left = &leaf_nodes[internal_morton_code_nodes[i].left_index];
			}
			else
			{
				internal_nodes[i].left = &internal_nodes[internal_morton_code_nodes[i].left_index];
			}

			if (internal_morton_code_nodes[i].is_right_leaf)
			{
				internal_nodes[i].right = &leaf_nodes[internal_morton_code_nodes[i].right_index];
			}
			else
			{
				internal_nodes[i].right = &internal_nodes[internal_morton_code_nodes[i].right_index];
			}
		}

		CUDA_CALL(cudaFree(triangle_morton_code_nodes_device));
		CUDA_CALL(cudaFree(leaf_morton_code_nodes_device));
		CUDA_CALL(cudaFree(internal_morton_code_nodes_device));

		SAFE_DELETE_ARRAY(triangle_morton_code_nodes);
		SAFE_DELETE_ARRAY(leaf_morton_code_nodes);
		SAFE_DELETE_ARRAY(internal_morton_code_nodes);
		
		SAFE_DELETE_ARRAY(leaf_nodes_triangle_indices);

		return &internal_nodes[0];
	}

	API_ENTRY void release_bvh(bvh_node* root_node)
	{
		bvh_morton_code_cpu::release_bvh(root_node);
	}
	
	API_ENTRY bvh_node_device* build_bvh_device_data(bvh_node* root)
	{
		return bvh_naive_cpu::build_bvh_device_data(root);
	}

	API_ENTRY void update_bvh(
		const glm::mat4& initial_transform_mat,
		const glm::mat4& transform_mat,
		bvh_node_device* initial_root,
		bvh_node_device* transformed_root
	)
	{
		bvh_naive_cpu::update_bvh(initial_transform_mat, transform_mat, initial_root, transformed_root);
	}
}