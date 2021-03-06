#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

/*
	1. Drag an obj file in
	3. output an txt file to list all used material's name:
	["Mat1", "Mat2", ...., "MatN"]
*/

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "Parameters invalid!" << endl;
		for (auto i = 0; i < argc; i++)
		{
			cout << "Param " << i << " -> " << argv[i] << endl;
		}
		return 0;
	}

	string input_path = argv[1];
	string path_base = input_path.substr(0, input_path.find_last_of('\\') + 1);
	string output_path = path_base + "material_json_array.txt";

	ifstream fin(input_path, ios::in);
	ofstream fout(output_path, ios::out);

	vector<string> material_names;

	char buffer[2048];
	while (fin.getline(buffer, 2048))
	{
		string text(buffer);
		istringstream stream(text);
		string token;
		stream >> token;

		if (token == "usemtl")
		{
			string material_name;
			stream >> material_name;
			material_names.push_back(material_name);
		}
	}

	fout << "[";

	for (auto i = 0; i < material_names.size(); i++)
	{
		fout << "\"";
		fout << material_names[i];
		fout << "\"";
		if (i != material_names.size() - 1)
		{
			fout << ", ";
		}
	}

	fout << "]";

	fout.close();
	fin.close();

	cout << "Material num : " << material_names.size() << endl;

	system("Pause");

    return 0;
}

