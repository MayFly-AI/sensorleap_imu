import numpy as np
import queue
import sys
import os
import threading
import time
import math
import argparse

from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *
import glfw

from csgl import *
import common
import controls_no_mouse
import cylinder

from get_data import DataSensorleap, DataRecording
from kalman import KalmanWrapper

# Global window
window = None
null = c_void_p(0)

def opengl_init():
    global window
    # Initialize the library
    if not glfw.init():
        print("Failed to initialize GLFW\n",file=sys.stderr)
        return False

    # Open Window and create its OpenGL context
    window = glfw.create_window(1024, 768, "Tutorial 06", None, None) #(in the accompanying source code this variable will be global)
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    if not window:
        print("Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n",file=sys.stderr)
        glfw.terminate()
        return False

    # Initialize GLEW
    glfw.make_context_current(window)
 #   glewExperimental = True

    # GLEW is a framework for testing extension availability.  Please see tutorial notes for
    # more information including why can remove this code.a
 #   if glewInit() != GLEW_OK:
 #       print("Failed to initialize GLEW\n",file=sys.stderr);
 #       return False
    return True

def floor(Nx,Nz,min_x,max_x,min_z,max_z):
    x = np.linspace(min_x,max_x,Nx)
    z = np.linspace(min_z,max_z,Nz)
    vertices = []
    colors = []
    for i in range(len(x)-1):
        black = False
        if i%2 == 0:
            black = True
        for j in range(len(z)-1):
            # first triangle in quad
            vertices.append([x[i],0.,z[j]])
            vertices.append([x[i],0.,z[j+1]])
            vertices.append([x[i+1],0.,z[j]])
            # second triangle in quad
            vertices.append([x[i+1],0.,z[j]])
            vertices.append([x[i],0.,z[j+1]])
            vertices.append([x[i+1],0.,z[j+1]])
            if black:
                colors.append([1.,1.,1.])
                colors.append([1.,1.,1.])
                colors.append([1.,1.,1.])
                colors.append([1.,1.,1.])
                colors.append([1.,1.,1.])
                colors.append([1.,1.,1.])
            else:
                colors.append([0.,0.,0.])
                colors.append([0.,0.,0.])
                colors.append([0.,0.,0.])
                colors.append([0.,0.,0.])
                colors.append([0.,0.,0.])
                colors.append([0.,0.,0.])
            black = not black
    vertices = np.array(vertices).flatten()
    colors = np.array(colors).flatten()
    return vertices, colors

def box(pos):
    vertices = []
    colors = []

    vertices = [-1.,-1.,-1.,
                -1.,-1., 1.,
                -1., 1., 1., 
                1., 1.,-1., 
                -1.,-1.,-1.,
                -1., 1.,-1.,
                1.,-1., 1.,
                -1.,-1.,-1.,
                1.,-1.,-1.,
                1., 1.,-1.,
                1.,-1.,-1.,
                -1.,-1.,-1.,
                -1.,-1.,-1.,
                -1., 1., 1.,
                -1., 1.,-1.,
                1.,-1., 1.,
                -1.,-1., 1.,
                -1.,-1.,-1.,
                -1., 1., 1.,
                -1.,-1., 1.,
                1.,-1., 1.,
                1., 1., 1.,
                1.,-1.,-1.,
                1., 1.,-1.,
                1.,-1.,-1.,
                1., 1., 1.,
                1.,-1., 1.,
                1., 1., 1.,
                1., 1.,-1.,
                -1., 1.,-1.,
                1., 1., 1.,
                -1., 1.,-1.,
                -1., 1., 1.,
                1., 1., 1.,
                -1., 1., 1.,
                1.,-1., 1.]
    vertices_lines = [-1.,-1.,-1., # face
                      -1.,-1.,1.,
                      -1.,-1.,-1.,
                      -1.,1.,-1.,
                      -1.,-1.,1.,
                      -1.,1.,1.,
                      -1.,1.,1.,
                      -1.,1.,-1.,
                      
                      1.,1.,1., #face
                      1.,1.,-1,
                      1.,1.,1.,
                      1.,-1,1.,
                      1.,1.,-1.,
                      1.,-1,-1.,
                      1,-1,1,
                      1,-1,-1,
                      
                      1,1,1, # connecting lines between faces
                      -1,1,1,
                      1,-1,1,
                      -1,-1,1,
                      1,1,-1,
                      -1,1,-1,
                      1,-1,-1,
                      -1,-1,-1
                      ]
        
    vertices = np.array(vertices)
    vertices_lines = np.array(vertices_lines)
    colors = np.ones([vertices.shape[0]])*0.5
    colors_lines = np.ones([vertices_lines.shape[0]])*0.
    return vertices, vertices_lines, colors, colors_lines    

def triangles_and_lines_for_camera():
    box_vertices, box_vertices_lines, box_colors, box_colors_lines = box(0) 
    v, v_lines = cylinder.cylinder(np.array([2.,0.,0.]), np.array([1.01,0.,0.]), 0.5, 0.5)

    vertex_data = []
    vertex_data_lines = []
    color_data = []
    color_data_lines = []

    vertex_data.append(box_vertices)
    vertex_data_lines.append(box_vertices_lines) 
    color_data.append(box_colors)
    color_data_lines.append(box_colors_lines)
    
    v = v.flatten()
    v_lines = v_lines.flatten()
    vertex_data.append(v)
    vertex_data_lines.append(v_lines)
    color_data.append(np.ones(len(v),dtype=np.float32)*0.5)
    color_data_lines.append(np.ones(len(v_lines),dtype=np.float32)*0.0)
    
    vertex_data = np.concatenate(vertex_data)
    vertex_data_lines = np.concatenate(vertex_data_lines)
    color_data = np.concatenate(color_data)
    color_data_lines = np.concatenate(color_data_lines)
    
    return vertex_data, vertex_data_lines, color_data, color_data_lines

# vertices to draw coordinate system
xyz_vertices = np.array([ 
    0.,0.,0.,
    5.,0.,0.,
    0.,0.,0.,
    0.,5.,0.,
    0.,0.,0.,
    0.,0.,5.], dtype=np.float32)
xyz_color = np.array([ 
    0.,0.,0.,
    1.,0.,0.,
    0.,0.,0.,
    0.,1.,0.,
    0.,0.,0.,
    0.,0.,1.], dtype=np.float32)
#floor_vertices, floor_colors = floor(21,21,-10.,10.,-10.,10.)

cam_vertices, cam_vertices_lines, cam_colors, cam_colors_lines = triangles_and_lines_for_camera()

def all_stuff_to_draw():
    vertex_data = []
    vertex_data_lines = []
    color_data = []
    color_data_lines = []

    vertex_data_lines.append(xyz_vertices)
    color_data_lines.append(xyz_color)

    #vertex_data.append(floor_vertices)
    #color_data.append(floor_colors)
    
    vertex_data_lines.append(cam_vertices_lines)
    color_data_lines.append(cam_colors_lines)
    vertex_data.append(cam_vertices)
    color_data.append(cam_colors)

    vertex_data = np.concatenate(vertex_data)
    vertex_data_lines = np.concatenate(vertex_data_lines)
    color_data = np.concatenate(color_data)
    color_data_lines = np.concatenate(color_data_lines)
        
    return vertex_data, vertex_data_lines, color_data, color_data_lines


def run(q_vis):
    if not opengl_init():
        return

    # Enable key events
    glfw.set_input_mode(window,glfw.STICKY_KEYS,GL_TRUE) 

    # Set opengl clear color to something other than red (color used by the fragment shader)
    glClearColor(1.0,1.0,1.0,0.0)
    
    # Enable depth test
    glEnable(GL_DEPTH_TEST)
    # Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS)
    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)
    glCullFace(GL_BACK)

    vertex_array_id = glGenVertexArrays(1)
    glBindVertexArray( vertex_array_id )

    program_id = common.LoadShaders( "./shaders/TransformVertexShader.vertexshader",
        "./shaders/ColorFragmentShader.fragmentshader" )
    
    # Get a handle for our "MVP" uniform
    matrix_id = glGetUniformLocation(program_id, "MVP");

    vertex_data, vertex_data_lines, color_data, color_data_lines = all_stuff_to_draw()

    vertex_buffer = glGenBuffers(1);
    array_type = GLfloat * len(vertex_data)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4, array_type(*vertex_data), GL_STATIC_DRAW)

    color_buffer = glGenBuffers(1);
    array_type = GLfloat * len(color_data)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    glBufferData(GL_ARRAY_BUFFER, len(color_data) * 4, array_type(*color_data), GL_STATIC_DRAW)

    vertex_buffer_lines = glGenBuffers(1);
    array_type = GLfloat * len(vertex_data_lines)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_lines)
    glBufferData(GL_ARRAY_BUFFER, len(vertex_data_lines) * 4, array_type(*vertex_data_lines), GL_STATIC_DRAW)

    color_buffer_lines = glGenBuffers(1);
    array_type = GLfloat * len(color_data_lines)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer_lines)
    glBufferData(GL_ARRAY_BUFFER, len(color_data_lines) * 4, array_type(*color_data_lines), GL_STATIC_DRAW)

    # vsync and glfw do not play nice.  when vsync is enabled mouse movement is jittery.
    #common.disable_vsyc()
    
    vis_data = np.zeros([3],dtype=np.float32) # roll, pitch and yaw

    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        t_start = time.time()
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)

        glUseProgram(program_id)

        controls_no_mouse.computeMatricesFromInputs(window)
        ProjectionMatrix = controls_no_mouse.getProjectionMatrix();
        ViewMatrix = controls_no_mouse.getViewMatrix();
        ModelMatrix = mat4.identity();

        if not q_vis.empty():
            vis_data = q_vis.get()
        Rx = mat4.identity()
        Ry = mat4.identity()
        Rz = mat4.identity()
        Rx.rotateX(vis_data[0])
        Ry.rotateY(vis_data[2])
        Rz.rotateZ(vis_data[1])
        ModelMatrix = Rx.__mul__(Ry).__mul__(Rz)
        
        mvp = ProjectionMatrix * ViewMatrix * ModelMatrix;

        # Send our transformation to the currently bound shader, 
        # in the "MVP" uniform
        glUniformMatrix4fv(matrix_id, 1, GL_FALSE,mvp.data)
       
        if not q_vis.empty():
            vertex_data, vertex_data_lines, color_data, color_data_lines = all_stuff_to_draw()

            array_type = GLfloat * len(vertex_data)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
            glBufferData(GL_ARRAY_BUFFER, len(vertex_data) * 4, array_type(*vertex_data), GL_STATIC_DRAW)

            array_type = GLfloat * len(color_data)
            glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
            glBufferData(GL_ARRAY_BUFFER, len(color_data) * 4, array_type(*color_data), GL_STATIC_DRAW)

            array_type = GLfloat * len(vertex_data_lines)
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_lines)
            glBufferData(GL_ARRAY_BUFFER, len(vertex_data_lines) * 4, array_type(*vertex_data_lines), GL_STATIC_DRAW)

            array_type = GLfloat * len(color_data_lines)
            glBindBuffer(GL_ARRAY_BUFFER, color_buffer_lines)
            glBufferData(GL_ARRAY_BUFFER, len(color_data_lines) * 4, array_type(*color_data_lines), GL_STATIC_DRAW)

        #1rst attribute buffer : vertices
        glEnableVertexAttribArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glVertexAttribPointer(
            0,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  # len(vertex_data)
            GL_FLOAT,           # type
            GL_FALSE,           # ormalized?
            0,                  # stride
            null                # array buffer offset (c_type == void*)
            )

		# 2nd attribute buffer : colors
        glEnableVertexAttribArray(1)
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
        glVertexAttribPointer(
			1,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  # len(vertex_data)
			GL_FLOAT,           # type
			GL_FALSE,           # ormalized?
			0,                  # stride
			null           		# array buffer offset (c_type == void*)
			)
        # Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, len(vertex_data)) #3 indices starting at 0 -> 1 triangle
        
        #1rst attribute buffer : vertices
        glEnableVertexAttribArray(2)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_lines);
        glVertexAttribPointer(
            0,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
            3,                  # len(vertex_data)
            GL_FLOAT,           # type
            GL_FALSE,           # ormalized?
            0,                  # stride
            null                # array buffer offset (c_type == void*)
            )

		# 2nd attribute buffer : colors
        glEnableVertexAttribArray(3)
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_lines);
        glVertexAttribPointer(
			1,                  # attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  # len(vertex_data)
			GL_FLOAT,           # type
			GL_FALSE,           # ormalized?
			0,                  # stride
			null           		# array buffer offset (c_type == void*)
			)
        glDrawArrays(GL_LINES, 0, len(vertex_data_lines)) 

        # Not strictly necessary because we only have 
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDisableVertexAttribArray(3)
        
        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

        # sleep to render at 60 fps (only use this if this function is running on separate thread)
        t_elapsed = (time.time()-t_start)/1000.
        print('t_elapsed: ',t_elapsed)
     #   if t_elapsed > 1000./60.:
     #       time.sleep((1000./60.-t_elapsed)/1000.)


    # !Note braces around vertex_buffer and uv_buffer.  
    # glDeleteBuffers expects a list of buffers to delete
    glDeleteBuffers(1, [vertex_buffer])
    glDeleteBuffers(1, [color_buffer])
    glDeleteBuffers(1, [vertex_buffer_lines])
    glDeleteBuffers(1, [color_buffer_lines])
    glDeleteProgram(program_id)
    glDeleteVertexArrays(1, [vertex_array_id])

    glfw.terminate()

def run_kalman(data, q_vis):
    kalman_wrapper = KalmanWrapper()
    t_prev = None
    while True:
        ts = []
        acc_x = []
        acc_y = []
        acc_z = []
        rads_x = []
        rads_y = []
        rads_z = []
        N = data.get_data(ts, acc_x, acc_y, acc_z, rads_x, rads_y, rads_z)
        if N == 0:
            time.sleep(1./100.)
        for j in range(N):
            t = ts[j]
            ax = acc_x[j]-means[0]
            ay = acc_y[j]-means[1]
            az = acc_z[j]-means[2]
            gx = rads_x[j]-means[3]
            gy = rads_y[j]-means[4]
            gz = rads_z[j]-means[5]
            if t_prev is not None:
                dt = (t_prev-t)/1000000. # microseconds to seconds
                roll_pitch_yaw = kalman_wrapper([gz,gx,gy,ax,az,ay], dt)
                q_vis.put(roll_pitch_yaw)
                time.sleep(1./20.)
            t_prev = t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--means', type=str, required=False)
    parser.add_argument('--recording', type=str, required=False)
    args = parser.parse_args()

    q_vis = queue.Queue(10)
    thread_run = threading.Thread(target=run, args=(q_vis,), daemon=False)
    thread_run.start()

    means = np.zeros([6])
    if args.means is not None:
        means = np.loadtxt(args.means)
        print('Loaded means',means)

    if args.recording is None:
        data = DataSensorleap()
    else:
        data = DataRecording(args.recording)

    run_kalman(data, q_vis)


