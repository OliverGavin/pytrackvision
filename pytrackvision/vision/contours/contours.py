import cv2
import numpy as np


def find_contours(mask, min_area=400, max_area=8000):
    """Find the contours for a binary mask constrained by its area.

    Parameters
    ----------
    mask: numpy.ndarray
        A 2D binarized array with values of 255 (black - ignore)
        or 0 (white - considered for contours).
    min_area: int
        The minimum area of the contour
    max_area: int
        The maximum area of the contour

    Returns
    -------
    numpy.ndarray
        A 3D array of points approximating the contour.
    """
    _, contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = list(filter(lambda cnt: min_area < cv2.contourArea(cnt) < max_area, contours))
    return contours


def find_min_enclosing_circle(contour):
    """Find the minimum enclosing circle for a contour.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.

    Returns
    -------
    Tuple[Tuple[int, int], int]
        A tuple containing the center point and radius.
    """
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def find_centroid(contour):
    """Find the centroid of a contour using moments.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the center point.
    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def find_convex_hull(contour):
    """Calculate the convex hull for a contour.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.

    Returns
    -------
    numpy.ndarray
        A 2D array containing the points along the convex hull.
    """
    hull = cv2.convexHull(contour, returnPoints=False)
    return hull


def find_convexity_defects(contour, hull=None):
    """Calculate the convexity defects for a contour.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.
    hull: numpy.ndarray, optional
        A 2D array containing the points along the convex hull.

    Returns
    -------
    numpy.ndarray
        A 3D array containing the index two adjacent points
        along the convex hull, the furthest/deepest defect
        and the distance of the defect, for each defect.
    """
    hull = hull if hull else find_convex_hull(contour)
    defects = cv2.convexityDefects(contour, hull)
    return defects


def find_k_curvatures(contour, hull, k=2, theta=60):
    """Find points along the hull with a curvature no greater than theta
    using points k steps away along the contour.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.
    hull: numpy.ndarray
        A 2D array containing the indices of points along the contour forming a convex hull.
    k: int
        The number of points to travel along the contour in both directions.
    theta: int or float
        The maximum angle of a curvature in degrees.

    Returns
    -------
    numpy.ndarray
        A 2D array containing the indices of k curvature points along the contour.
    """
    # Consider an angle ABC where a, b and c are the points along the curvature.
    sef = contour[np.swapaxes(np.array([hull - k, hull, (hull + k) % contour.shape[0]]), 1, 0)][:, :, 0][:, :, 0]

    # We will calculate the angle ABC.
    ba = sef[:, 0] - sef[:, 2]  # Distance BA as a vector
    bc = sef[:, 1] - sef[:, 2]  # Distance BC as a vector

    ba_norm = np.linalg.norm(ba, axis=1)  # ||BA|| is the Euclidean (L2) norm for each point in BA
    bc_norm = np.linalg.norm(bc, axis=1)  # ||BC|| is the Euclidean (L2) norm for each point in BC

    dot_prod = np.einsum('ij,ij->i', ba, bc)  # Dot product BA.BC as a vectorised row wise operation over the ith index ONLY.

    cosine_angle = dot_prod / (ba_norm * bc_norm)
    angle = np.arccos(cosine_angle)       # The angle in radians for each ABC

    # Using defects which has the same original dimensions we can obtain the indices for defects along the contour
    # that have and angle of greater than theta degrees.
    k_curvatures = hull[np.degrees(angle) > theta]

    return k_curvatures


def find_deep_convexity_defects_points(contour, defects):
    """Filter shallow/noisy defects.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.
    defects: numpy.ndarray
        A 3D array containing the index two adjacent points
        along the convex hull, the furthest/deepest defect
        and the distance of the defect, for each defect.

    Returns
    -------
    numpy.ndarray
        A 2D array of points along the contour which have the largest defects.
    """
    # defects is a matrix with N * 1 * 4 dimensions, that is, it contains arrays of tuples.
    # Each tuple consists of a start, end and furthest point as indices of points along the
    # contour as well as the distance of the furthest point.
    # The fartest point is the convexity defect. Start and end are the points along the convex hull.
    # defects[:, 0, :3] selects an N * 3 tuple consisting of the start, end and furthest point indices.
    # contour is a matrix with N * 1 * 2 dimensions containing points.
    # contour[defects[:, 0, :3]] selects an N * 3 * 1 * 2.
    # Since an N * 3 tuple was used as an index on an N * 1 * x, this yields an element wise operation,
    # hence the addition of a dimension of size 3.
    # contour[defects[:, 0, :3]][:, :, 0] flattens the result into an N * 3 * 2 matrix which is easier to work with.
    # sef now contains an array of tuples of start, end and furthest points.
    sef = contour[defects[:, 0, :3]][:, :, 0]

    # Consider an angle ABC where a, b and c are the start, end and furthest points, respectively.
    # We will calculate the angle ABC.
    ba = sef[:, 0] - sef[:, 2]  # Distance BA as a vector
    bc = sef[:, 1] - sef[:, 2]  # Distance BC as a vector

    ba_norm = np.linalg.norm(ba, axis=1)  # ||BA|| is the Euclidean (L2) norm for each point in BA
    bc_norm = np.linalg.norm(bc, axis=1)  # ||BC|| is the Euclidean (L2) norm for each point in BC

    dot_prod = np.einsum('ij,ij->i', ba, bc)  # Dot product BA.BC as a vectorised row wise operation over the ith index ONLY.

    cosine_angle = dot_prod / (ba_norm * bc_norm)
    angle = np.arccos(cosine_angle)       # The angle in radians for each ABC

    # Using defects which has the same original dimensions we can obtain the indices for defects along the contour
    # that have and angle of less than 90 degrees.
    deep_defects_points = contour[defects[np.degrees(angle) < 90][:, 0, 2]]

    return deep_defects_points[:, 0]  # Flatten


def nearest_point_distance(point, points, return_point=False):
    """Find the distance of the closest point from a set of points.

    Parameters
    ----------
    point: Tuple[int, int]
        A tuple containing the point.
    points: numpy.ndarray
        A 2D array of points for which the distance of `point` must be calculated.
    """
    deltas = points - point
    dist = np.linalg.norm(deltas, axis=1)

    if return_point:
        min_idx = np.argmin(dist)
        return points[min_idx], dist[min_idx]

    return np.min(dist)


def voronoi_vertices(points):
    """Finds the vertices of a Vornoi diagram for a point cloud.

    A Voronoi diagram is used to partition points by maximising the
    distance of its vertices from them.

    Parameters
    ----------
    points: numpy.ndarray
        A 2D array of points.

    Returns
    -------
    numpy.ndarray
        A 2D array of points.
    """
    # min and max points are used to create only the maximum required matrix size
    min_x_p = np.min(points[:, 0])
    min_y_p = np.min(points[:, 1])
    max_x_p = np.max(points[:, 0])
    max_y_p = np.max(points[:, 1])
    rect = (min_x_p, min_y_p, max_x_p + 1, max_y_p + 1)
    div = cv2.Subdiv2D(rect)

    for p in points:
        # insert the points into the matrix
        div.insert((p[0], p[1]))

    # Merge all the vertices from all the facets
    vertices = np.concatenate(div.getVoronoiFacetList([])[0])

    # Filter out vertices outside the image
    vertices = vertices[((min_x_p <= vertices[:, 0]) & (vertices[:, 0] <= max_x_p))
                        &
                        ((min_y_p <= vertices[:, 1]) & (vertices[:, 1] <= max_y_p))]

    # Filter out duplicate vertices (because adjacent facets share vertices)
    b = vertices.ravel().view(np.dtype((np.void, vertices.dtype.itemsize*vertices.shape[1])))
    _, unique_idx = np.unique(b, return_index=True)
    vertices = vertices[unique_idx]

    return vertices


def find_max_incircle(centroid, points):
    """Finds the maximum incircle in a point cloud which contains
    a given centroid.

    A Voronoi diagram is used to partition points by maximising the
    distance of its vertices from them. One of these vertices is the
    correct incircle which is selected if the centroid lies within
    that region. The incircle with the largest radius is selected.

    Parameters
    ----------
    centroid: Tuple[int, int]
        A tuple containing the centroid point.
    points: numpy.ndarray
        A 2D array of points.

    Returns
    -------
    Tuple[Tuple[int, int], int] or None
        A tuple containing the center point and radius.
    """
    cX, cY = centroid

    if len(points) >= 4:

        vertices = voronoi_vertices(points)

        max_d = 0
        max_v = None
        for (vX, vY) in vertices:
            d = nearest_point_distance((vX, vY), points)
            # also check if the centroid is in the circle
            if d > max_d and np.linalg.norm(np.array([vX, vY]) - np.array([cX, cY])) < d:
                max_d = d
                max_v = (vX, vY)

        if max_v:
            center, radius = (int(max_v[0]), int(max_v[1])), int(max_d)
            return (center, radius)

    return None
