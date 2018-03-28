import cv2
import numpy as np
# from scipy.spatial import Voronoi


def find_contours(mask, min_area=800, max_area=8000):
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
    _, contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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


def find_centroid(contours):
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
    M = cv2.moments(contours)  # https://en.wikipedia.org/wiki/Image_moment
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def find_convexity_defects(contours):
    """Calculate the convex hull and convexity defects for a contour.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.

    Returns
    -------
    numpy.ndarray
        A 3D array containing the index two adjacent points
        along the convex hull, the farthest/deepest defect
        and the distance of the defect, for each defect.
    """
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull)
    return defects


def find_deep_convexity_defects_points(contour, defects, min_depth=200):
    """Filter shallow/noisy defects.

    Parameters
    ----------
    contour: numpy.ndarray
        A 3D array of points approximating the contour.
    defects: numpy.ndarray
        A 3D array containing the index two adjacent points
        along the convex hull, the farthest/deepest defect
        and the distance of the defect, for each defect.
    min_depth: int
        The minimum acceptable depth.
        Anything shallower is filtered out.

    Returns
    -------
    numpy.ndarray
        A 2D array of points along the contour which have the largest defects.
    """
    deep_idx = defects[defects[:, 0, 3] > min_depth][:, 0, 2]   # d > 200  (depth index=3)
                                                                # flatten with :, 0
                                                                # get the index of the farthest defect (farthest index=2)
    deep_defects_points = contour[deep_idx][:, 0]               # select contours with idx and flatten
    return deep_defects_points


def nearest_point_distance(point, points):
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
    # min_idx = np.argmin(dist)
    # return points[min_idx], dist[min_idx], deltas[min_idx][1]/deltas[min_idx][0]
    return np.min(dist)


# def find_max_incircle(contour, centroid, points):
#     cX, cY = centroid
#     points = contour[::5][:, 0]
#     points = np.concatenate((points, points), axis=0)
#
#     try:
#         if len(points) >= 4:
#
#             # vor = Voronoi(points)
#
#             rect = (0, 0, 320, 240)
#             div = cv2.Subdiv2D(rect)
#             # mat = np.zeros(shape=(240, 320, 3))
#             for p in points:
#                 div.insert((p[0], p[1]))
#                 # cv2.circle(mat, (p[0], p[1]), 4, [255, 255, 255], -1)
#
#             # points = map(list, zip(*points))
#             # div.insert([x for x in map(tuple, points)])
#
#             # for (vX, vY) in vor.vertices:
#             #     cv2.circle(mat, (int(vX), int(vY)), 2, [0, 200, 0], 1)
#
#             # test = div.getEdgeList()[:, 2:4]
#             test = np.concatenate(div.getVoronoiFacetList([])[0])
#             test = test[((0 <= test[:, 0]) & (test[:, 0] <= 320)) & ((0 <= test[:, 1]) & (test[:, 1] <= 240))]
#
#             # print(vor.vertices)
#             # print(test)
#             # print(len(vor.vertices), len(test))
#
#             b = test.ravel().view(np.dtype((np.void, test.dtype.itemsize*test.shape[1])))
#             _, unique_idx = np.unique(b, return_index=True)
#
#             test = test[np.sort(unique_idx)]
#
#             # print(test)
#             # print(len(test))
#             # print()
#             #
#             # for p in test:
#             #     cv2.circle(mat, (p[0], p[1]), 1, [0, 0, 200], 1)
#             #
#             # cv2.imshow('test', mat)
#             # import pdb; pdb.set_trace()
#
#             max_d = 0
#             max_v = None
#             for (vX, vY) in test:
#             # for (vX, vY) in test:
#                 # cv2.circle(img, (int(vX), int(vY)), 1, [200, 200, 0], 1)
#                 # _, d, _ = nearest_point_distance((vX, vY), points)
#                 d = nearest_point_distance((vX, vY), points)
#                 # also check if starting point is in the circle
#                 if d > max_d and np.linalg.norm(np.array([vX, vY]) - np.array([cX, cY])) < d:
#                     max_d = d
#                     max_v = (vX, vY)
#
#             if max_v:
#                 center, radius = (int(max_v[0]), int(max_v[1])), int(max_d)
#                 return (center, radius)
#                 # cv2.circle(img, center, radius, [0, 0, 0], 1)
#     except Exception as e:
#         print('Voronoi failed:')
#         print()
#         print(centroid)
#         print()
#         print(points)
#         print()
#         print(e)
#         print()
#
#     return None

def find_max_incircle(contour, centroid, points):
    """Finds the maximum incircle in a point cloud which contains
    a given centroid.

    A Voronoi diagram is used to partition points by maximising the
    distance of its vertices from them. One of these vertices is the
    correct incircle which is selected if the centroid lies within
    that region. The incircle with the largest radius is selected.

    Parameters
    ----------
    contour: numpy.ndarray  # todo remove/move
        A 3D array of points approximating the contour.
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
    points = contour[::5][:, 0]
    points = np.concatenate((points, points), axis=0)

    if len(points) >= 4:

        rect = (0, 0, 320, 240)
        div = cv2.Subdiv2D(rect)

        for p in points:
            div.insert((p[0], p[1]))

        facets = np.concatenate(div.getVoronoiFacetList([])[0])
        vertices = facets[((0 <= facets[:, 0]) & (facets[:, 0] <= 320)) & ((0 <= facets[:, 1]) & (facets[:, 1] <= 240))]

        b = vertices.ravel().view(np.dtype((np.void, vertices.dtype.itemsize*vertices.shape[1])))
        _, unique_idx = np.unique(b, return_index=True)

        vertices = vertices[unique_idx]
        # vertices = vertices[np.sort(unique_idx)]

        max_d = 0
        max_v = None
        for (vX, vY) in vertices:
            # cv2.circle(img, (int(vX), int(vY)), 1, [200, 200, 0], 1)
            d = nearest_point_distance((vX, vY), points)
            # also check if starting point is in the circle
            if d > max_d and np.linalg.norm(np.array([vX, vY]) - np.array([cX, cY])) < d:
                max_d = d
                max_v = (vX, vY)

        if max_v:
            center, radius = (int(max_v[0]), int(max_v[1])), int(max_d)
            return (center, radius)
            # cv2.circle(img, center, radius, [0, 0, 0], 1)

    return None
