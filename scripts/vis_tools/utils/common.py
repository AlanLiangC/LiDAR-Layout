import numpy as np

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def box2coord3d(boxes_3d):
    num_box = boxes_3d.shape[0]
    boxes_vec_points = np.zeros([num_box, 3, 8])
    l,w,h = boxes_3d[:,3], boxes_3d[:,4], boxes_3d[:,5]
    c_xyz = boxes_3d[:,:3][:,:, None]

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    boxes_vec_points[:, 0, :] = np.transpose(np.stack(x_corners))
    boxes_vec_points[:, 1, :] = np.transpose(np.stack(y_corners))
    boxes_vec_points[:, 2, :] = np.transpose(np.stack(z_corners))

    rotzs = []
    for box in boxes_3d:
        rotzs.append(rotz(box[6]))
    rotzs = np.stack(rotzs)

    corners_3d = rotzs @ boxes_vec_points # N, 3, 8
    corners_3d += c_xyz
    corners_3d = np.transpose(corners_3d, (0,2,1)).reshape(-1, 3)

    return corners_3d