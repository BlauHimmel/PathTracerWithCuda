#include <gl\glew.h>
#include <gl\freeglut.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "path_tracer.hpp"

using namespace std;

int width = 512, height = 512;

int mouse_old_x = 0, mouse_old_y = 0;
int mouse_button = 0, modifier_button = 0;

float rotate_x = 0.0f, rotate_y = 0.0f;
float translate_z = -30.0f;

view_camera* view_cam = new view_camera();
render_camera* render_cam = new render_camera();
path_tracer* pt = new path_tracer();

void init_all(int argc, char *argv[]);

bool init_gl();
void init_camera();

void display();
void keyboard(uchar key, int x, int y);
void mouse(int button, int state, int x, int y);
void mouse_wheel(int button, int dir, int x, int y);
void motion(int x, int y);

int main(int argc, char* argv[])
{
	init_all(argc, argv);
	return 0;
}

void init_all(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA Path Tracer");

	srand(unsigned(time(0)));

	init_camera();
	view_cam->get_render_camera(render_cam);
	pt->init(render_cam);

	if (!init_gl())
	{
		return;
	}

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc(mouse_wheel);
	glutMotionFunc(motion);

	glutMainLoop();
}

bool init_gl()
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
	view_cam->eye = glm::vec4(-2.0f, 1.13f, 5.8f, 0.0f);
	view_cam->up = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
	view_cam->target = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
	view_cam->set_resolution(static_cast<float>(width), static_cast<float>(height));
	view_cam->set_fov_x(40.0f);
}

void display()
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

	glColor4f(0.0f, 0.0f, 0.0f, 0.0f);
	glRasterPos2f(0.01f, 0.975f);
	char log[2048];
	sprintf(log, "FPS: %.2f", get_fps(render_image));
	for (uint i = 0; i < strlen(log); i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, log[i]);
	}

	glRasterPos2f(0.01f, 0.945f);
	sprintf(log, "Iteration: %u", render_image->pass_counter);
	for (uint i = 0; i < strlen(log); i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, log[i]);
	}

	glRasterPos2f(0.01f, 0.915f);
	sprintf(log, "Elapse: %.2f sec", get_elapsed_second(render_image));
	for (uint i = 0; i < strlen(log); i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, log[i]);
	}

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
	glutPostRedisplay();
}

void keyboard(uchar key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(1);
		break;
	case ' ':
		init_camera();
		pt->clear();
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	mouse_button = button;
	modifier_button = glutGetModifiers();

	mouse_old_x = x;
	mouse_old_y = y;

	motion(x, y);
}

void mouse_wheel(int button, int dir, int x, int y)
{
	if (dir > 0)
	{
		view_cam->zoom_in(0.5f);
	}
	else
	{
		view_cam->zoom_out(0.5f);
	}

	pt->clear();
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx = (float)(mouse_old_x - x);
	float dy = (float)(mouse_old_y - y);

	if (dx == 0 && dy == 0)
	{
		return;
	}

	if (mouse_button == GLUT_LEFT_BUTTON)
	{

		if (dx > 0)
		{
			view_cam->orbit_right(dx * 0.005f);
		}
		else
		{
			view_cam->orbit_left(-dx * 0.005f);
		}

		if (dy > 0)
		{
			view_cam->orbit_up(dy * 0.005f);
		}
		else
		{
			view_cam->orbit_down(-dy * 0.005f);
		}

		pt->clear();
		glutPostRedisplay();
	}
	else if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		view_cam->move(dx * 0.005f, -dy * 0.005f);
		pt->clear();
		glutPostRedisplay();
	}

	mouse_old_x = x;
	mouse_old_y = y;

	
}
