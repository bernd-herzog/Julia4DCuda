
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>


////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;
/* maximum
const unsigned int mesh_width = 512;
const unsigned int mesh_height = 512;
const unsigned int mesh_depth = 1024;
*/

#define threads_per_block 64

const unsigned int mesh_depth = threads_per_block*2; // multiple of 265! depth³ must fit in signed int (265*5=1280)

GLuint vbo;
GLuint vboNormals;

int once = 0;
int countd = 0;

unsigned char *dbuffer;
unsigned char *dbuffer2;
unsigned char *dhbuffer;
int *countbuffer;
int *hcountbuffer;
unsigned int _displayListId;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

float4 C = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
float e = 0.0f;

////////////////////////////////////////////////////////////////////////////////
// kernels
#include <Julia4D_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createBuffers();
void deleteVBO( GLuint* vbo);

// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda( );
void checkResultCuda( int argc, char** argv, const GLuint& vbo);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    runTest( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "Cuda GL interop");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);

    // create VBO
    createBuffers( );
	dhbuffer = new unsigned char[mesh_depth*mesh_depth*mesh_depth/8];
	hcountbuffer = new int[mesh_depth*mesh_depth];
	
    // run the cuda part
    runCuda( );

    // check result of Cuda step
 //   checkResultCuda( argc, argv, vbo);

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
	if (once != 0)	return;

    // execute the kernel
    /*
	64 threads per block is minimal and makes sense only if there are multiple active
	blocks per multiprocessor. 192 or 256 threads per block is better and usually allows
	for enough registers to compile.
	*/	

	dim3 grid(mesh_depth/threads_per_block, mesh_depth);
	dim3 block(threads_per_block);

	kernel<<< grid, block>>>(dbuffer, mesh_depth, C, e, 2);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	kernel2<<< grid, block>>>(dbuffer, countbuffer, mesh_depth);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(hcountbuffer, countbuffer, mesh_depth*mesh_depth*4, cudaMemcpyDeviceToHost));

	countd = 0;
	for (int i = 0; i < mesh_depth*mesh_depth; i++){
		countd += hcountbuffer[i];
	}

    // initialize buffer object
    glBindBuffer( GL_ARRAY_BUFFER, vbo);
    glBufferData( GL_ARRAY_BUFFER, countd * 4 * 4 * sizeof( float), 0, GL_DYNAMIC_DRAW);
    
    glBindBuffer( GL_ARRAY_BUFFER, vboNormals);
    glBufferData( GL_ARRAY_BUFFER, countd * 4 * 4 * sizeof( float), 0, GL_DYNAMIC_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA

    float4 *dptr;
    float3 *dptrNormals;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, vbo));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptrNormals, vboNormals));

	kernel3<<< grid, block>>>(dbuffer, countbuffer, dptr, dptrNormals, mesh_depth);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vboNormals));
	
	once = 1;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    
 
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
//    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);
   // glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createBuffers()
{
	CUDA_SAFE_CALL(cudaMalloc((void **) &dbuffer,  mesh_depth*mesh_depth*mesh_depth/8));
	//CUDA_SAFE_CALL(cudaMalloc((void **) &dbuffer2, mesh_height*mesh_width*mesh_depth/8));
	CUDA_SAFE_CALL(cudaMalloc((void **) &countbuffer, mesh_depth*mesh_depth*4));
	
    glGenBuffers( 1, &vbo);
    glGenBuffers( 1, &vboNormals);

    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vbo));
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(vboNormals));


    CUT_CHECK_ERROR_GL();
	
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{

}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat mat_shininess[] = { 50.0 };
    GLfloat light_position[] = { 1.0, -1.0, -4.0, 0.0 };

    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);


    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, vboNormals);
	glNormalPointer(GL_FLOAT, 0, 0);
//	glColorPointer(4, GL_FLOAT, 0, 0);
	
    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
//	glEnableClientState(GL_COLOR_ARRAY);
    
    glColor3f(1.0, 1.0, 1.0);
    glDrawArrays(GL_QUADS, 0, countd*4);
    
    glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
//	glDisableClientState(GL_COLOR_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        exit( 0);
        break;
    case( 'q') :
		once = 0;
		e += 0.1f;
		break;
    case( 'a') :
		once = 0;
		e -= 0.1f;
		break;
    case( 'w') :
		once = 0;
		C.x += 0.1f;
		break;
    case( 's') :
		once = 0;
		C.x -= 0.1f;
		break;
    case( 'e') :
		once = 0;
		C.y += 0.1f;
		break;
    case( 'd') :
		once = 0;
		C.y -= 0.1f;
		break;
    case( 'r') :
		once = 0;
		C.z += 0.1f;
		break;
    case( 'f') :
		once = 0;
		C.z -= 0.1f;
		break;
    case( 't') :
		once = 0;
		C.w += 0.1f;
		break;
    case( 'g') :
		once = 0;
		C.w -= 0.1f;
		break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda( int argc, char** argv, const GLuint& vbo)
{
    CUT_CHECK_ERROR_GL();
}
