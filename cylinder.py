import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

def cylinder(base_center, top_center, base_radius, top_radius):
    # Find two unit vectors u1,u2 on the plane normal to the vector d1
    d1 = top_center-base_center
    # u1 is a perpendicular vector to d1. We could have obtained this by computing the cross product between d1
    # and an arbitrary vector (different from 0 and not parallel to d1), fx: u1 = np.cross(d1,np.random.rand(3)).
    # Instead of cross product, we use a more numerically stable approach from "Real-time Rendering" section 3.2.4,
    # This guarantees that u1 is perpendicular to d1 and later that (d1,u1,u2) form an orthonormal basis
    if np.abs(d1[0]) < np.abs(d1[1]) and np.abs(d1[0]) < np.abs(d1[2]):
        u1 = [0,-d1[2],d1[1]]
    elif np.abs(d1[1]) < np.abs(d1[0]) and np.abs(d1[1]) < np.abs(d1[2]):
        u1 = [-d1[2],0.,d1[0]]
    else:
        u1 = [-d1[1],d1[0],0.]
    u1 = np.array(u1)
    u2 = np.cross(d1,u1)
    u1 = u1/np.sqrt(np.sum(u1**2))
    u2 = u2/np.sqrt(np.sum(u2**2))
    N=10
    vertices_base = np.zeros((N,3))
    vertices_top = np.zeros((N,3))
    for idx, t in enumerate(np.linspace(0,2.*np.pi,N)):
        vertices_base[idx,:] = base_center + base_radius*np.cos(t)*u1 + base_radius*np.sin(t)*u2
        vertices_top[idx,:] = top_center + top_radius*np.cos(t)*u1 + top_radius*np.sin(t)*u2

    # tube: N-1 quads, 2 triangles per quad, 3 vertices per triangle
    # base: N-1 triangles, 3 vertices per triangle
    # top: N-1 triangles, 3 vertices per triangle
    triangle_vertices_tube = np.zeros(((N-1)*2*3,3), dtype=np.float32)
    triangle_vertices_base = np.zeros(((N-1)*3,3), dtype=np.float32)
    triangle_vertices_top  = np.zeros(((N-1)*3,3), dtype=np.float32)
    for i in range(N-1):
        triangle_vertices_tube[i*6+0,:] = vertices_top[i]
        triangle_vertices_tube[i*6+1,:] = vertices_base[i]
        triangle_vertices_tube[i*6+2,:] = vertices_base[i+1]
        triangle_vertices_tube[i*6+3,:] = vertices_base[i+1]
        triangle_vertices_tube[i*6+4,:] = vertices_top[i+1]
        triangle_vertices_tube[i*6+5,:] = vertices_top[i]

        triangle_vertices_base[i*3+0,:] = base_center
        triangle_vertices_base[i*3+1,:] = vertices_base[i+1]
        triangle_vertices_base[i*3+2,:] = vertices_base[i]

        triangle_vertices_top[i*3+0,:] = top_center
        triangle_vertices_top[i*3+1,:] = vertices_top[i]
        triangle_vertices_top[i*3+2,:] = vertices_top[i+1]
    '''
    line_vertices_tube = np.zeros((N*2,3), dtype=np.float32)
    line_vertices_base = np.zeros((N*2,3), dtype=np.float32)
    line_vertices_top  = np.zeros((N*2,3), dtype=np.float32)
    for i in range(N):
        line_vertices_tube[i*2+0] = vertices_base[i]
        line_vertices_tube[i*2+1] = vertices_top[i]

        line_vertices_base[i*2+0] = vertices_base[i]
        line_vertices_base[i*2+1] = base_center

        line_vertices_top[i*2+0] = vertices_top[i]
        line_vertices_top[i*2+1] = top_center
    '''
    line_vertices_tube = np.zeros((N*2,3), dtype=np.float32)
    #line_vertices_base = np.zeros((N*2,3), dtype=np.float32)
    #line_vertices_top  = np.zeros((N*2,3), dtype=np.float32)
    line_vertices_base2 = np.zeros(((N-1)*2*2,3), dtype=np.float32)
    line_vertices_top2 = np.zeros(((N-1)*2*2,3), dtype=np.float32)
    for i in range(N):
        line_vertices_tube[i*2+0] = vertices_base[i]
        line_vertices_tube[i*2+1] = vertices_top[i]
        
        #line_vertices_base[i*2+0] = vertices_base[i]
        #line_vertices_base[i*2+1] = base_center

        #line_vertices_top[i*2+0] = vertices_top[i]
        #line_vertices_top[i*2+1] = top_center
    for i in range(N-1):
        line_vertices_base2[i*2+0] = vertices_base[i]
        line_vertices_base2[i*2+1] = vertices_base[i+1]
        line_vertices_top2[i*2+0] = vertices_top[i]
        line_vertices_top2[i*2+1] = vertices_top[i+1]
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vertices_base[:,0],vertices_base[:,1],vertices_base[:,2], color='blue')
        ax.scatter(vertices_top[:,0],vertices_top[:,1],vertices_top[:,2], color='purple')
        ax.scatter(*base_center, color='red',s=100)
        ax.scatter(*top_center, color='green',s=100)
        ax.plot(triangle_vertices_tube[:,0], triangle_vertices_tube[:,1], triangle_vertices_tube[:,2], color='black')
        ax.plot(triangle_vertices_base[:,0], triangle_vertices_base[:,1], triangle_vertices_base[:,2], color='green')
        ax.plot(triangle_vertices_top[:,0], triangle_vertices_top[:,1], triangle_vertices_top[:,2], color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    vertices = np.concatenate((triangle_vertices_tube,triangle_vertices_base,triangle_vertices_top), axis=0)
    vertices_line = np.concatenate((line_vertices_tube,line_vertices_base2,line_vertices_top2), axis=0)
    return vertices, vertices_line

def cylinder_with_cones(base_center, top_center, base_radius, top_radius):
    # locations for pointy ends of cones
    base_cone = base_center
    top_cone = top_center

    # Find two unit vectors u1,u2 on the plane normal to the vector d1
    d1 = top_center-base_center

    top_center = base_center + 0.66*d1
    base_center = base_center + 0.33*d1

    # u1 is a perpendicular vector to d1. We could have obtained this by computing the cross product between d1
    # and an arbitrary vector (different from 0 and not parallel to d1), fx: u1 = np.cross(d1,np.random.rand(3)).
    # Instead of cross product, we use a more numerically stable approach from "Real-time Rendering" section 3.2.4,
    # This guarantees that u1 is perpendicular to d1 and later that (d1,u1,u2) form an orthonormal basis
    if np.abs(d1[0]) < np.abs(d1[1]) and np.abs(d1[0]) < np.abs(d1[2]):
        u1 = [0,-d1[2],d1[1]]
    elif np.abs(d1[1]) < np.abs(d1[0]) and np.abs(d1[1]) < np.abs(d1[2]):
        u1 = [-d1[2],0.,d1[0]]
    else:
        u1 = [-d1[1],d1[0],0.]
    u1 = np.array(u1)
    u2 = np.cross(d1,u1)
    u1 = u1/np.sqrt(np.sum(u1**2))
    u2 = u2/np.sqrt(np.sum(u2**2))
    N=10
    vertices_base = np.zeros((N,3))
    vertices_top = np.zeros((N,3))
    for idx, t in enumerate(np.linspace(0,2.*np.pi,N)):
        vertices_base[idx,:] = base_center + base_radius*np.cos(t)*u1 + base_radius*np.sin(t)*u2
        vertices_top[idx,:] = top_center + top_radius*np.cos(t)*u1 + top_radius*np.sin(t)*u2

    # tube: N-1 quads, 2 triangles per quad, 3 vertices per triangle
    # base: N-1 triangles, 3 vertices per triangle
    # top: N-1 triangles, 3 vertices per triangle
    triangle_vertices_tube = np.zeros(((N-1)*2*3,3), dtype=np.float32)
    triangle_vertices_base = np.zeros(((N-1)*3,3), dtype=np.float32)
    triangle_vertices_top  = np.zeros(((N-1)*3,3), dtype=np.float32)
    for i in range(N-1):
        triangle_vertices_tube[i*6+0,:] = vertices_top[i]
        triangle_vertices_tube[i*6+1,:] = vertices_base[i]
        triangle_vertices_tube[i*6+2,:] = vertices_base[i+1]
        triangle_vertices_tube[i*6+3,:] = vertices_base[i+1]
        triangle_vertices_tube[i*6+4,:] = vertices_top[i+1]
        triangle_vertices_tube[i*6+5,:] = vertices_top[i]

        triangle_vertices_base[i*3+0,:] = base_cone
        triangle_vertices_base[i*3+1,:] = vertices_base[i+1]
        triangle_vertices_base[i*3+2,:] = vertices_base[i]

        triangle_vertices_top[i*3+0,:] = top_cone
        triangle_vertices_top[i*3+1,:] = vertices_top[i]
        triangle_vertices_top[i*3+2,:] = vertices_top[i+1]

    line_vertices_tube = np.zeros((N*2,3), dtype=np.float32)
    line_vertices_base = np.zeros((N*2,3), dtype=np.float32)
    line_vertices_top  = np.zeros((N*2,3), dtype=np.float32)
    for i in range(N):
        line_vertices_tube[i*2+0] = vertices_base[i]
        line_vertices_tube[i*2+1] = vertices_top[i]

        line_vertices_base[i*2+0] = vertices_base[i]
        line_vertices_base[i*2+1] = base_cone

        line_vertices_top[i*2+0] = vertices_top[i]
        line_vertices_top[i*2+1] = top_cone
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vertices_base[:,0],vertices_base[:,1],vertices_base[:,2], color='blue')
        ax.scatter(vertices_top[:,0],vertices_top[:,1],vertices_top[:,2], color='purple')
        ax.scatter(*base_center, color='red',s=100)
        ax.scatter(*top_center, color='green',s=100)
        ax.scatter(*base_cone, color='blue',s=100)
        ax.scatter(*top_cone, color='black',s=100)
        ax.plot(triangle_vertices_tube[:,0], triangle_vertices_tube[:,1], triangle_vertices_tube[:,2], color='black')
        ax.plot(triangle_vertices_base[:,0], triangle_vertices_base[:,1], triangle_vertices_base[:,2], color='green')
        ax.plot(triangle_vertices_top[:,0], triangle_vertices_top[:,1], triangle_vertices_top[:,2], color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    vertices = np.concatenate((triangle_vertices_tube,triangle_vertices_base,triangle_vertices_top), axis=0)
    vertices_line = np.concatenate((line_vertices_tube,line_vertices_base,line_vertices_top), axis=0)
    return vertices, vertices_line


if __name__ == '__main__':
    base_center = np.array([0.,0.,0.])
    top_center = np.array([6.,6.,6.])
    base_radius=2.
    top_radius=1.
    #cylinder(base_center,top_center,base_radius,top_radius)
    cylinder_with_cones(base_center,top_center,base_radius,top_radius)
