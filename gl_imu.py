import numpy as np
import queue
import sys
import os
import time
import math
import argparse
from multiprocessing import Process, Queue

from OpenGL.GL import *
from OpenGL.GL.ARB import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT.special import *
from OpenGL.GL.shaders import *
import glfw

import common
import controls
import cylinder
import box

from get_data import DataSensorleap, DataRecording
from kalman import KalmanWrapper
from conversion import accelerometer_to_attitude

import glm

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
    window = glfw.create_window(1024, 768, "IMU GL Demo", None, None) #(in the accompanying source code this variable will be global)
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

def triangles_and_lines_for_camera():
    # -1,1 box at (0,0,0)
    box_vertices, box_vertices_lines, box_colors, box_colors_lines = box.box() 
    # cylinder bug when x_top == x_base. therefore following hack (fix this)
    v, v_lines = cylinder.cylinder(np.array([0.00001,0.,-1.01]), np.array([0.00002,0,-2.]), 0.5, 0.5)

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
    
    tmp = 0.
    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        t_start = time.time()
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT)

        glUseProgram(program_id)

        controls.computeMatricesFromInputs(window)
        ProjectionMatrix = controls.getProjectionMatrix();
        ViewMatrix = controls.getViewMatrix();
 
        if not q_vis.empty():
            vis_data = q_vis.get()

        if False: # quat to euler to rotation matrix
            euler = glm.eulerAngles(vis_data) # pitch, yaw, roll
            rot_matrix = glm.mat4() #identity
            rot_matrix = glm.rotate(rot_matrix, euler.x, glm.vec3(1., 0., 0.))
            rot_matrix = glm.rotate(rot_matrix, euler.y, glm.vec3(0., 1., 0.))
            rot_matrix = glm.rotate(rot_matrix, euler.z, glm.vec3(0., 0., 1.))
        else: # quat to rotation matrix
            Q = vis_data
            rot_matrix = glm.mat4(Q)

        #rot_matrix = glm.mat4() #identity
        #rot_matrix = glm.rotate(rot_matrix, tmp, glm.vec3(1., 0., 0.))
        #tmp += 0.001

        mvp = ProjectionMatrix * ViewMatrix * rot_matrix;

        # Send our transformation to the currently bound shader, 
        # in the "MVP" uniform
        glUniformMatrix4fv(matrix_id, 1, GL_FALSE,glm.value_ptr(mvp))
       
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
        #print('t_elapsed: ',t_elapsed)
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
            ax = acc_x[j]-calib[0]
            ay = acc_y[j]-calib[1]
            az = acc_z[j]-calib[2]
            gx = rads_x[j]-calib[3]
            gy = rads_y[j]-calib[4]
            gz = rads_z[j]-calib[5]
            if t_prev is not None:
                dt = (t-t_prev)/1000000. # microseconds to seconds
                Q = kalman_wrapper([gx,gy,gz,ax,az,ay], dt)

                Q = glm.quat(*Q) # convert to GLM quat
                Q = glm.normalize(Q)
                q_vis.put(Q)
            t_prev = t

def run_unfiltered(data, q_vis):
    t_prev = None
    g_roll = 0.
    g_pitch = 0.
    g_yaw = 0.
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
            ax = acc_x[j]-calib[0]
            ay = acc_y[j]-calib[1]
            az = acc_z[j]-calib[2]
            gx = rads_x[j]-calib[3]
            gy = rads_y[j]-calib[4]
            gz = rads_z[j]-calib[5]
            if t_prev is not None:
                dt = (t-t_prev)/1000000. # microseconds to seconds

                a_roll, a_pitch, _ = accelerometer_to_attitude(ax,az,ay)

                g_roll = g_roll + gz*dt
                g_pitch = g_pitch + gx*dt
                g_yaw = g_yaw + gy*dt

                Q = glm.quat(glm.vec3(g_pitch, g_yaw, g_roll))
                #Q = glm.quat(glm.vec3(a_pitch, g_yaw, a_roll))
                Q = glm.normalize(Q)
                q_vis.put(Q)
            t_prev = t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', type=str, required=False)
    parser.add_argument('--recording', type=str, required=False)
    args = parser.parse_args()

    q_vis = Queue(10)
    p = Process(target=run, args=(q_vis,))
    p.start()

    calib = np.zeros([6])
    if args.calib is not None:
        calib = np.loadtxt(args.calib)
        print('Loaded calibration values:',calib)

    if args.recording is None:
        data = DataSensorleap()
    else:
        data = DataRecording(args.recording)

    run_kalman(data, q_vis)
    #run_unfiltered(data, q_vis)


