import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import numpy as np
from random import seed,randint
import cv2 as cv
import matplotlib.patches as patches
import trackCube_v3

#W_face_data = np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])

def plotCube(face_list):
    face_colors = []
    for face in face_list:
        face_colors.append((face.colors, face.rotation))
    cube_layout = plt.figure()
    cube_layout.canvas.set_window_title('Cube State')
    W_face = cube_layout.add_subplot(345, aspect='equal')
    R_face = cube_layout.add_subplot(346, aspect='equal')
    Y_face = cube_layout.add_subplot(347, aspect='equal')
    O_face = cube_layout.add_subplot(348, aspect='equal')
    B_face = cube_layout.add_subplot(341, aspect='equal')
    G_face = cube_layout.add_subplot(349, aspect='equal')
    ax_faces = [W_face,R_face,Y_face,O_face,B_face,G_face]
    for face in face_colors:
        f_colors,f_rot = face[0],face[1]
        for p in [
            patches.Rectangle(
                (0, 0), .333, .333,
                facecolor=f_colors[2,0]
            ),
            patches.Rectangle(
                (.333, 0), .333, .333,
                facecolor=f_colors[2,1]
            ),
            patches.Rectangle(
                (.666, 0), .333, .333,
                facecolor=f_colors[2,2]
            ),
            patches.Rectangle(
                (0, .333), .333, .333,
                facecolor=f_colors[1,0]
            ),
            patches.Rectangle(
                (.333, .333), .333, .333,
                facecolor=f_colors[1,1]
            ),
            patches.Rectangle(
                (.666, .333), .333, .3331,
                facecolor=f_colors[1,2]
            ),
            patches.Rectangle(
                (0, .666), .333, .333,
                facecolor=f_colors[0,0]
            ),
            patches.Rectangle(
                (.333, .666), .333, .333,
                facecolor=f_colors[0,1]
            ),
            patches.Rectangle(
                (.666, .666), .333, .333,
                facecolor=f_colors[0,2]
            ),
        ]:
            if f_colors[1,1] == 'white':
                W_face.add_patch(p)
            if f_colors[1,1] == 'red':
                R_face.add_patch(p)
            if f_colors[1,1] == 'yellow':
                Y_face.add_patch(p)
            if f_colors[1,1] == 'orange':
                O_face.add_patch(p)
            if f_colors[1,1] == 'blue':
                B_face.add_patch(p)
            if f_colors[1,1] == 'green':
                G_face.add_patch(p)
    for ax in ax_faces:
        for loc in [0, 0.33, 0.66, 1]:
            ax.axvline(x=loc, color='k')
            ax.axhline(y=loc, color='k')
        ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

    plt.show()

W_face = trackCube_v3.Face('white')
R_face = trackCube_v3.Face('red')
Y_face = trackCube_v3.Face('yellow')
O_face = trackCube_v3.Face('orange')
B_face = trackCube_v3.Face('blue')
G_face = trackCube_v3.Face('green')
print W_face.colors[1,2]
W_face.colors[1,2] = 'red'
print W_face.colors[1,2]
print R_face.colors[1,2]
R_face.colors[1,2] = 'white'
print R_face.colors[1,2]
face_list = [W_face,R_face,Y_face,O_face,B_face,G_face]
plotCube(face_list)


def plotCubeV2(face_list):
    pass




'''
cube_layout = plt.figure()
W_face = cube_layout.add_subplot(345,aspect='equal')
R_face = cube_layout.add_subplot(346,aspect='equal')
Y_face = cube_layout.add_subplot(347,aspect='equal')
O_face = cube_layout.add_subplot(348,aspect='equal')
B_face = cube_layout.add_subplot(341,aspect='equal')
G_face = cube_layout.add_subplot(349,aspect='equal')
for p in[
    patches.Rectangle(
        (0, 0), .333, .333,
        facecolor='red'
    ),
    patches.Rectangle(
        (.333, 0), .333, .333,
        facecolor='blue'
    ),
    patches.Rectangle(
        (.666, 0), .333, .333,
        facecolor='orange'
    ),
    patches.Rectangle(
        (0, .333), .333, .333,
        facecolor='green'
    ),
    patches.Rectangle(
        (.333, .333), .333, .333,
        facecolor='white'
    ),
    patches.Rectangle(
        (.666, .333), .333, .3331,
        facecolor='yellow'
    ),
    patches.Rectangle(
        (0, .666), .333, .333,
        facecolor='red'
    ),
    patches.Rectangle(
        (.333, .666), .333, .333,
        facecolor='green'
    ),
    patches.Rectangle(
        (.666, .666), .333, .333,
        facecolor='orange'
    ),
]:
    W_face.add_patch(p)
plt.show()
'''
'''
img = np.ones((512,512,3),np.uint8)
cv.rectangle(img,(0,0),(200,200),(0,255,0),3)
cv.imshow('IMG',img)
cv.waitKey()




N = 3
data = np.ones((N,N)) * np.nan
data[(0,0)] = 1
data[(0,1)] = 2
data[(0,2)] = 3
data[(1,1)] = 2
fig,ax = plt.subplots(1,1,tight_layout=True)
my_cmap = matplotlib.colors.ListedColormap(['w','r','y','k','b','g'])
my_cmap.set_bad(color='w',alpha=0)
for x in range(N+1):
    ax.axhline(x,lw=2,color='k',zorder=5)
    ax.axvline(x,lw=2,color='k',zorder=5)
ax.imshow(data,interpolation='none',cmap=my_cmap,extent=[0,N,0,N],zorder=0)
ax.axis('off')
plt.show()


ny,nx = 3,3
r,g,b = [np.random.random(ny*nx).reshape((ny,nx)) for _ in range(3)]
#r = np.array([[0,0,0],[255,255,255],[255,255,255]])
#g = np.array([[0,0,0],[0,0,0],[255,255,255]])
#b = np.array([[255,255,255],[255,255,255],[0,0,0]])
print 'r',r
print 'g',g
print 'b',b


c = np.dstack([r,g,b])
print c

plt.imshow(c)
plt.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')
plt.show()



data = np.random.rand(3,3)*20

cmap = colors.ListedColormap(['red','blue'])
bounds = [0,10,20]
norm = colors.BoundaryNorm(bounds,cmap.N)

gridsize = (3,4)
fig = plt.figure(figsize=(12,8))
plt.axis('off')
W_face = plt.subplot2grid(gridsize,(1,0))
R_face = plt.subplot2grid(gridsize,(1,1))
Y_face = plt.subplot2grid(gridsize,(1,2))
O_face = plt.subplot2grid(gridsize,(1,3))
B_face = plt.subplot2grid(gridsize,(0,0))
G_face = plt.subplot2grid(gridsize,(2,0))

W_face.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')
R_face.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')
Y_face.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')
O_face.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')
B_face.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')
G_face.tick_params(axis='both',which='both',bottom='off',top='off',right='off',left='off',labelbottom='off',labelleft='off')

W_face.imshow(data,cmap=cmap,norm=norm)
W_face.grid(which='both',linestyle='solid')
W_face.set_xlim(-.5,2.5)
W_face.set_ylim(-.5,2.5)
W_face.grid(which='major',axis='both',color='k',linewidth=2)


plt.show()
'''
'''
data = np.random.rand(3,3)*20

cmap = colors.ListedColormap(['red','blue'])
bounds = [0,10,20]
norm = colors.BoundaryNorm(bounds,cmap.N)

fig,ax = plt.subplots()
ax.imshow(data,cmap=cmap,norm=norm)

ax.grid(which='major',axis='both',linestyle='-',color='k',linewidth=2)
ax.set_xticks(np.arange(.5,3,1))
ax.set_yticks(np.arange(0,3,1))

plt.show()
'''