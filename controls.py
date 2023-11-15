#! /usr/bin/env python
from OpenGL.GL import *
import glm

import glfw
import math as mathf

ViewMatrix = glm.mat4()
ProjectionMatrix = glm.mat4()

def getViewMatrix():
    return ViewMatrix

def getProjectionMatrix():
    return ProjectionMatrix


#position = vec3( 0, 0, 5 )
#position = vec3( 0, 2.5, 8 ) #MAX
#position = vec3( 8, 2.5, 0 ) #MAX looking at our model from +X
position = glm.vec3( 8, 2.5, 0 ) #MAX looking at our model from +X
# Initial horizontal angle : toward -Z
#horizontalAngle = 3.14
horizontalAngle = 1.5*3.14 #MAX
# Initial vertical angle : none
#verticalAngle = 0.0
verticalAngle = -0.2 #MAX
# Initial Field of View
initialFoV = 45.0

speed = 3.0 # 3 units / second
#mouseSpeed = 0.005
#mouseSpeed = 0.001 #MAX

lastTime = None

def computeMatricesFromInputs(window):
    global lastTime
    global position
    global horizontalAngle
    global verticalAngle
    global initialFoV
    global ViewMatrix
    global ProjectionMatrix

    # glfwGetTime is called only once, the first time this function is called
    if lastTime == None:
        lastTime = glfw.get_time()

    # Compute time difference between current and last frame
    currentTime = glfw.get_time()
    deltaTime = currentTime - lastTime

    # Get mouse position
#    xpos,ypos = glfw.get_cursor_pos(window)

    # Reset mouse position for next frame
#    glfw.set_cursor_pos(window, 1024/2, 768/2);

    # Compute new orientation
    #horizontalAngle += mouseSpeed * float(1024.0/2.0 - xpos );
    #verticalAngle   += mouseSpeed * float( 768.0/2.0 - ypos );

    if glfw.get_key( window, glfw.KEY_RIGHT ) == glfw.PRESS:
        horizontalAngle -= deltaTime * speed * 0.2

    if glfw.get_key( window, glfw.KEY_LEFT ) == glfw.PRESS:
        horizontalAngle += deltaTime * speed * 0.2

    if glfw.get_key( window, glfw.KEY_UP ) == glfw.PRESS:
        verticalAngle -= deltaTime * speed * 0.2

    if glfw.get_key( window, glfw.KEY_DOWN ) == glfw.PRESS:
        verticalAngle += deltaTime * speed * 0.2

    # Direction : Spherical coordinates to Cartesian coordinates conversion
    direction = glm.vec3(
        mathf.cos(verticalAngle) * mathf.sin(horizontalAngle), 
        mathf.sin(verticalAngle),
        mathf.cos(verticalAngle) * mathf.cos(horizontalAngle)
    )
    
    # Right vector
    right = glm.vec3(
        mathf.sin(horizontalAngle - 3.14/2.0), 
        0.0,
        mathf.cos(horizontalAngle - 3.14/2.0)
    )

    # Up vector
    up = glm.cross( right, direction )

    # Move forward
    if glfw.get_key( window, glfw.KEY_W ) == glfw.PRESS:
        position += direction * deltaTime * speed;
    
    # Move backward
    if glfw.get_key( window, glfw.KEY_S ) == glfw.PRESS:
        position -= direction * deltaTime * speed
    
    # Strafe right
    if glfw.get_key( window, glfw.KEY_D ) == glfw.PRESS:
        position += right * deltaTime * speed
    
    # Strafe left
    if glfw.get_key( window, glfw.KEY_A ) == glfw.PRESS:
        position -= right * deltaTime * speed
    
    # Move up
    if glfw.get_key( window, glfw.KEY_R ) == glfw.PRESS:
        position += up * deltaTime * speed

    # Move down
    if glfw.get_key( window, glfw.KEY_F ) == glfw.PRESS:
        position -= up * deltaTime * speed


    FoV = initialFoV# - 5 * glfwGetMouseWheel(); # Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

    # Projection matrix : 45 Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    ProjectionMatrix = glm.perspective(glm.radians(FoV), 4.0 / 3.0, 0.1, 100.0)
    # Camera matrix
    ViewMatrix       = glm.lookAt(
                                position,           # Camera is here
                                position+direction, # and looks here : at the same position, plus "direction"
                                up                  # Head is up (set to 0,-1,0 to look upside-down)
                           )

    # For the next frame, the "last time" will be "now"
    lastTime = currentTime
