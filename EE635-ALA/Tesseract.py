import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- CONFIG ---
SCALE = 1.5
ZOOM_ORTHO = 3.0 
ZOOM_PERSP = 1.8
CAM_DIST_4D = 3.0  
CAM_DIST_3D = 2.8

# --- MATRIX FUNCTIONS ---
def rotation_mat(plane, theta):
    R = np.eye(4)
    i, j = plane
    c, s = np.cos(theta), np.sin(theta)
    R[i, i] = c 
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R

# --- Orthogonal (Linear) ---
def ortho_4d_3d(points):
    p = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0]])
    P = p @ points.T
    return P.T

def ortho_3d_2d(points):
    p = np.array([[1,0,0],
                [0,1,0]])
    P = p @ points.T
    return P.T

# --- Perspective (Projective) ---
def persp_4d_3d(points, d = CAM_DIST_4D):
    w = points[:, 3]
    scale = 1 / (d - w)
    scale[np.abs(d - w) < 1e-5] = 1e-5
    k = []
    for i, s in enumerate(scale):
        p = np.array([[s,0,0,0],
                    [0,s,0,0],
                    [0,0,s,0]])
        e = p @ points[i, :]
        k.append(e)
    P = np.array(k)
    return P

def persp_3d_2d(points, d = CAM_DIST_3D):
    z = points[:, 2]
    scale = 1 / (d - z)
    scale[np.abs(d - z) < 1e-5] = 1e-5
    k=[]
    for i,s in enumerate(scale):
        p = np.array([[s,0,0],
                    [0,s,0]])
        e = p @ points[i,:]
        k.append(e)
    P = np.array(k)
    return P

# --- 4D TESSERACT ---
vertices = np.array([[x, y, z, w]
                     for x in (-1, 1)
                     for y in (-1, 1)
                     for z in (-1, 1)
                     for w in (-1, 1)], dtype=float) * SCALE

edges = []
for i, v in enumerate(vertices):
    for j in range(i + 1, len(vertices)):
        if np.isclose(np.sum(np.abs(v - vertices[j])), 2 * SCALE): 
            edges.append((i, j))

# --- PLOTTING ---
fig = plt.figure(figsize=(12, 10))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.1, hspace=0.2)

# Orthogonal
ax_ortho_3d = fig.add_subplot(221, projection='3d')
ax_ortho_3d.set_title("Orthogonal (Linear)\n4D Tesseract in 3D", fontsize=10, fontweight='bold')
ax_ortho_2d = fig.add_subplot(222)
ax_ortho_2d.set_title("Orthogonal (Linear)\n3D Tesseract in 2D", fontsize=10, fontweight='bold')

# 3D Plot
ax_ortho_3d.set_xlim([-ZOOM_ORTHO, ZOOM_ORTHO])
ax_ortho_3d.set_ylim([-ZOOM_ORTHO, ZOOM_ORTHO])
ax_ortho_3d.set_zlim([-ZOOM_ORTHO, ZOOM_ORTHO])
ax_ortho_3d.set_xticks([]) 
ax_ortho_3d.set_yticks([]) 
ax_ortho_3d.set_zticks([])

# 2D Plot
ax_ortho_2d.set_xlim([-ZOOM_ORTHO, ZOOM_ORTHO])
ax_ortho_2d.set_ylim([-ZOOM_ORTHO, ZOOM_ORTHO])
ax_ortho_2d.set_aspect('equal')
ax_ortho_2d.set_xticks([])
ax_ortho_2d.set_yticks([])

# Perspective
ax_persp_3d = fig.add_subplot(223, projection='3d')
ax_persp_3d.set_title("Perspective (Projective)\n4D Tesseract in 3D", fontsize=10, fontweight='bold')
ax_persp_2d = fig.add_subplot(224)
ax_persp_2d.set_title("Perspective (Projective)\n4D Tesseract in 2D", fontsize=10, fontweight='bold')

# 3D Plot
ax_persp_3d.set_xlim([-ZOOM_PERSP, ZOOM_PERSP])
ax_persp_3d.set_ylim([-ZOOM_PERSP, ZOOM_PERSP])
ax_persp_3d.set_zlim([-ZOOM_PERSP, ZOOM_PERSP])
ax_persp_3d.set_xticks([])
ax_persp_3d.set_yticks([])
ax_persp_3d.set_zticks([])

# 2D Plot 
ax_persp_2d.set_xlim([-ZOOM_PERSP, ZOOM_PERSP])
ax_persp_2d.set_ylim([-ZOOM_PERSP, ZOOM_PERSP])
ax_persp_2d.set_aspect('equal')
ax_persp_2d.set_xticks([])
ax_persp_2d.set_yticks([])

# Color 
cmap = plt.get_cmap('plasma')
norm = plt.Normalize(vmin=-SCALE*1.5, vmax=SCALE*1.5)

# Initializes
lines_O3 = [ax_ortho_3d.plot([], [], [], linewidth=2)[0] for _ in edges]
lines_O2 = [ax_ortho_2d.plot([], [], linewidth=2)[0] for _ in edges]
lines_P3 = [ax_persp_3d.plot([], [], [], linewidth=2)[0] for _ in edges]
lines_P2 = [ax_persp_2d.plot([], [], linewidth=2)[0] for _ in edges]

def animate(frame):
    # Rotation in 4D
    theta = frame * 0.02
    M = rotation_mat((0, 3), theta) @ rotation_mat((1, 2), theta * 0.5)
    verts_rot = vertices @ M.T
    
    # Orthogonal
    v_ortho_3d = ortho_4d_3d(verts_rot)
    v_ortho_2d = ortho_3d_2d(v_ortho_3d)

    # Perspective
    v_persp_3d = persp_4d_3d(verts_rot)
    v_persp_2d = persp_3d_2d(v_persp_3d)
    
    # Plot update
    for k, (i, j) in enumerate(edges):
        midpoint = (verts_rot[i] + verts_rot[j]) / 2.0
        color = cmap(norm(midpoint[3]))

        # Orthogonal
        lines_O3[k].set_data(v_ortho_3d[[i, j], 0], v_ortho_3d[[i, j], 1])
        lines_O3[k].set_3d_properties(v_ortho_3d[[i, j], 2])
        lines_O3[k].set_color(color)
        lines_O2[k].set_data(v_ortho_2d[[i, j], 0], v_ortho_2d[[i, j], 1])
        lines_O2[k].set_color(color)

        # Perspective
        lines_P3[k].set_data(v_persp_3d[[i, j], 0], v_persp_3d[[i, j], 1])
        lines_P3[k].set_3d_properties(v_persp_3d[[i, j], 2])
        lines_P3[k].set_color(color)
        lines_P2[k].set_data(v_persp_2d[[i, j], 0], v_persp_2d[[i, j], 1])
        lines_P2[k].set_color(color)

    return lines_O3 + lines_O2 + lines_P3 + lines_P2

anim = FuncAnimation(fig, animate, frames=200, interval=30, blit=False)
anim.save('./images/tesseract_anim.gif', writer='pillow', fps=30)
plt.show()