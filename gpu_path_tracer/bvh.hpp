#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>

#include <stack>

#include "triangle_mesh.hpp"
#include "cuda_math.hpp"

#define BVH_LEAF_NODE_TRIANGLE_NUM 6
#define BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM 12

struct bounding_box
{
	float3 left_bottom;
	float3 right_top;
	float3 centroid;

	bounding_box() {}

	bounding_box(const float3& left_bottom, const float3& right_top) :
		left_bottom(left_bottom), right_top(right_top), centroid(0.5f * (right_top - left_bottom))
	{

	}

	void expand_to_fit_box(const float3& other_left_bottom, const float3& other_right_top)
	{
		left_bottom = fminf(left_bottom, other_left_bottom);
		right_top = fmaxf(right_top, other_right_top);
		centroid = 0.5f * (right_top - left_bottom);
	}

	void expand_to_fit_triangle(const float3& vertex0, const float3& vertex1, const float3& vertex2)
	{
		left_bottom = make_float3(min(vertex0.x, vertex1.x, vertex2.x), min(vertex0.y, vertex1.y, vertex2.z), min(vertex0.x, vertex1.y, vertex2.z));
		right_top = make_float3(max(vertex0.x, vertex1.x, vertex2.x), max(vertex0.y, vertex1.y, vertex2.z), max(vertex0.x, vertex1.y, vertex2.z));
		centroid = 0.5f * (right_top - left_bottom);
	}

	float get_surface_area()
	{
		return 2.0f * (right_top.x - left_bottom.x) * (right_top.y - left_bottom.y) * (right_top.z - left_bottom.z);
	}

	//0:x 1:y 2:z
	int get_longest_axis(int& length /*out*/)
	{
		float dx = right_top.x - left_bottom.x;
		float dy = right_top.y - left_bottom.y;
		float dz = right_top.z - left_bottom.z;
		length = max(dx, dy, dz);

		if (dx == length)	return 0;
		else if (dy == length) return 1;
		else return 2;
	}
};


struct bvh_node
{
	bounding_box box;
	bvh_node* left = nullptr;
	bvh_node* right = nullptr;
	bool is_leaf = false;
	thrust::device_vector<int> triangle_indices;
};

int bounding_box::get_longest_axis(int& length /*out*/)
{
	float dx = right_top.x - left_bottom.x;
	float dy = right_top.y - left_bottom.y;
	float dz = right_top.z - left_bottom.z;
	length = max(dx, dy, dz);

	if (dx == length)	return 0;
	else if (dy == length) return 1;
	else return 2;
}

bounding_box get_bounding_box(const triangle& triangle)
{
	bounding_box box;
	box.left_bottom = make_float3(
		min(triangle.vertex0.x, triangle.vertex1.x, triangle.vertex2.x),
		min(triangle.vertex0.y, triangle.vertex1.y, triangle.vertex2.z),
		min(triangle.vertex0.x, triangle.vertex1.y, triangle.vertex2.z)
	);
	box.right_top = make_float3(
		max(triangle.vertex0.x, triangle.vertex1.x, triangle.vertex2.x),
		max(triangle.vertex0.y, triangle.vertex1.y, triangle.vertex2.z),
		max(triangle.vertex0.x, triangle.vertex1.y, triangle.vertex2.z)
	);
	box.centroid = 0.5f * (box.right_top - box.left_bottom);
	return box;
}


bvh_node* build_bvh(triangle* triangles, int triangle_num)
{
	bvh_node* root_note = new bvh_node();

	for (auto i = 0; i < triangle_num; i++)
	{
		root_note->box.expand_to_fit_triangle(triangles[i].vertex0, triangles[i].vertex1, triangles[i].vertex2);
		root_note->triangle_indices.push_back(i);
	}

	if (root_note->triangle_indices.size() < BVH_LEAF_NODE_TRIANGLE_NUM)
	{
		root_note->is_leaf = true;
		return root_note;
	}

	bounding_box* boxes = new bounding_box[triangle_num];

	for (auto i = 0; i < triangle_num; i++)
	{
		boxes[i] = get_bounding_box(triangles[i]);
	}

	split_bounding_box(triangles, triangle_num, root_note, boxes);

	SAFE_DELETE_ARRAY(boxes);
}

void split_bounding_box(triangle* triangles, int triangle_num, bvh_node* node, bounding_box* boxes)
{
	std::stack<bvh_node*> stack;
	stack.push(node);

	while (!stack.empty())
	{
		node = stack.top();
		stack.pop();

		//divide the bounding along the longest axis into several internal
		int longest_axis_length;
		int longest_axis = node->box.get_longest_axis(longest_axis_length);
		int divide_internal_num = min(BVH_BUCKET_MAX_DIVIDE_INTERNAL_NUM, node->triangle_indices.size());
		int internal_length = longest_axis_length / divide_internal_num;

		bvh_node* internal = new bvh_node[divide_internal_num];

		for (size_t i = 0; i < node->triangle_indices.size(); i++)
		{
			int triangle_index = node->triangle_indices[i];

			int internal_index = (get(boxes[triangle_index].centroid, longest_axis) - get(boxes[triangle_index].left_bottom, longest_axis)) / internal_length;
			internal[internal_index].box.expand_to_fit_box(boxes[triangle_index].left_bottom, boxes[triangle_index].right_top);
			internal[internal_index].triangle_indices.push_back(node->triangle_indices[i]);
		}

		//find the minimal cost of split operation
		int split_internal_index;
		bounding_box split_box_left, split_box_right;
		int split_triangle_num_left, split_triangle_num_right;

		int min_cost = 2147483647;

		for (auto i = 0; i < divide_internal_num; i++)
		{
			bounding_box box_left;
			size_t triangle_num_left = 0;

			for (auto j = 0; j < i; j++)
			{
				box_left.expand_to_fit_box(internal[j].box.left_bottom, internal[j].box.right_top);
				triangle_num_left += internal[j].triangle_indices.size();
			}

			bounding_box box_right;
			size_t triangle_num_right = 0;

			for (auto j = i; j < divide_internal_num; j++)
			{
				box_right.expand_to_fit_box(internal[j].box.left_bottom, internal[j].box.right_top);
				triangle_num_right += internal[j].triangle_indices.size();
			}

			float cost = box_left.get_surface_area() * triangle_num_left + box_right.get_surface_area() * triangle_num_right;

			if (cost < min_cost)
			{
				split_internal_index = cost;
				split_box_left = box_left;
				split_box_right = box_right;
				split_triangle_num_left = triangle_num_left;
				split_triangle_num_right = triangle_num_right;
			}
		}

		//build the subnode of current node
		if (split_triangle_num_left > 0)
		{
			bvh_node* left = new bvh_node();
			left->box = split_box_left;
			for (auto i = 0; i < split_internal_index; i++)
			{
				left->triangle_indices.insert(left->triangle_indices.end(), internal[i].triangle_indices.begin(), internal[i].triangle_indices.end());
			}
			if (split_triangle_num_left <= BVH_LEAF_NODE_TRIANGLE_NUM)
			{
				left->is_leaf = true;
			}
			node->left = left;
			stack.push(left);
		}

		if (split_triangle_num_right > 0)
		{
			bvh_node* right = new bvh_node();
			right->box = split_box_left;
			for (auto i = 0; i < split_internal_index; i++)
			{
				right->triangle_indices.insert(right->triangle_indices.end(), internal[i].triangle_indices.begin(), internal[i].triangle_indices.end());
			}
			if (split_triangle_num_left <= BVH_LEAF_NODE_TRIANGLE_NUM)
			{
				right->is_leaf = true;
			}
			node->right = right;
			stack.push(right);
		}

		SAFE_DELETE_ARRAY(internal);
	}
}