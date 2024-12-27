import numpy as np
import matplotlib as mpl
import scipy.interpolate as si

def gen_arrow_head_marker(angle):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    angle=0: positive x direction
    Parameters
    ----------
    angle : float
        rotation in rad
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """
    arr = np.array([[.1, .3], [.1, -.3], [1, 0], [.1, .3]])  # arrow shape
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [mpl.path.Path.MOVETO, mpl.path.Path.LINETO,mpl.path.Path.LINETO, mpl.path.Path.CLOSEPOLY]
    arrow_head_marker = mpl.path.Path(arr, codes)
    return arrow_head_marker, scale

""" Calculate n samples on a bspline

        cv :      Array ov control vertices
        degree:   Curve degree
    """
    
def generate_bspline(cv: np.array, degree=3) -> list:
    """
    Generate a 2D B-spline by the set of control point.

    Args:
    ----
        cv (np.array): Set of control points.
        degree (int, optional): B-spline degree. Defaults to 3.

    Returns:
    -------
        list: (BSpline, BSpline) spline and derivative spline, accordingly.

    """
    control_vertices = np.asarray(cv)
    count = control_vertices.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    knot_vector = np.array([0] * degree + list(range(count - degree + 1)) + [count - degree] * degree,
                           dtype='int')

    # Normalize the knot vector to 1
    knot_vector = knot_vector / (count - degree)

    # Calculate splines
    spline = si.BSpline(knot_vector, control_vertices, degree)
    spline_derivative = spline.derivative(1)

    return (spline, spline_derivative)