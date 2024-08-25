# max_area_instance

class transformation:
    
    def __init__(self):
        pass
    
    def polygon_area(polygon):
        polygon = np.array(polygon)
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    
    def select_max(self, outputs):
        max_area = 0
        max_area_index = -1
        for i, instance in enumerate(outputs['instances']):
            for j, polygon_data in enumerate(instance['mask']):
                polygon = polygon_data['polygon']
                area = polygon_area(polygon)
                # print(f"Mask_{i}_{j} area: {area}")
                if area > max_area:
                    max_area = area
                    max_area_instance = {
                        "image_name": outputs["file_name"],
                        "class": instance["class"],
                        "box": instance["box"],
                        "score": instance["score"],
                        "mask": [{"polygon": polygon}]
                    }                           
        return max_area_instance
        
    def dp_simplify(polygon, epsilon_factor=0.006):
        contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

        
    def vw_simplify(polygon, threshold=0.5):
        def area_of_triangle(p1, p2, p3):
            return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        
        points = np.array(polygon, dtype=np.float32)
        while True:
            is_removed = np.zeros(points.shape[0], dtype=bool)
            if len(points) < 3:
                break
            areas = np.array([area_of_triangle(points[i - 1], points[i], points[i + 1]) for i in range(1, len(points) - 1)])
            if len(areas) == 0 or np.min(areas) >= threshold:
                break
            min_index = np.argmin(areas) + 1
            is_removed[min_index] = True
            points = points[~is_removed]
            print(f"Removed point at index {min_index}, remaining points: {len(points)}")
        return points

    def ch_simplify(polygon):
        points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        hull = cv2.convexHull(points)
        return hull

    def simplify_polygon(polygon, method="vw", threshold=0.5):
        if method == "vw":
            return vw_simplify(polygon, threshold)
        elif method == "ch":
            return ch_simplify(polygon)
        else:
            return polygon
        
    
class perspective_tran:
    def __init__(self):
        pass
    
    def preprocess_vertices(vertices):
        points = [(v[0][0], v[0][1]) for v in vertices]

        center = np.mean(points, axis=0)
        sorted_points = sorted(points, key=lambda p: (np.arctan2(p[1] - center[1], p[0] - center[0])))
        return sorted_points

    def polygon_centroid(vertices):
        vertices = np.array(vertices)
        n = len(vertices)
        if n < 3:
            raise ValueError("A polygon must have at least three vertices")

        A = 0.5 * np.sum(vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1])
        Cx = (1/(6*A)) * np.sum((vertices[:-1, 0] + vertices[1:, 0]) * (vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]))
        Cy = (1/(6*A)) * np.sum((vertices[:-1, 1] + vertices[1:, 1]) * (vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]))

        return Cx, Cy

    def distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def get_vertices_from_quad(vertices, centroid, image_shape):
        dists = [
            distance(vertices[0], vertices[1]),
            distance(vertices[1], vertices[2]),
            distance(vertices[2], vertices[3]),
            distance(vertices[3], vertices[0])
        ]

        avg_length = [dists[0] + dists[2], dists[1] + dists[3]]

        rect_width = avg_length[0] / 2
        rect_height = avg_length[1] / 2
        
        # image center
        img_center_x = image_shape[1] / 2
        img_center_y = image_shape[0] / 2

        rect = np.array([
            [img_center_x - rect_width / 2, img_center_y - rect_height / 2],
            [img_center_x + rect_width / 2, img_center_y - rect_height / 2],
            [img_center_x + rect_width / 2, img_center_y + rect_height / 2],
            [img_center_x - rect_width / 2, img_center_y + rect_height / 2]
        ], dtype="float32")
        
        # Adjust rect to fit within image bounds
        rect[:, 0] = np.clip(rect[:, 0], 0, image_shape[1] - 1)
        rect[:, 1] = np.clip(rect[:, 1], 0, image_shape[0] - 1)
        
        return rect

    def get_perspective_transform(vertices, image_shape):
        if len(vertices) == 4:
            
            scale_factor = 0.5
            image_small = cv2.resize(image_org, (0, 0), fx=scale_factor, fy=scale_factor)

            # processing vertices
            vertices = preprocess_vertices(vertices)
            centroid = polygon_centroid(vertices)

            # scale vertices and centroid
            vertices_small = [(v[0] * scale_factor, v[1] * scale_factor) for v in vertices]
            centroid_small = (centroid[0] * scale_factor, centroid[1] * scale_factor)

            # transformation matrix
            rect_small = get_vertices_from_quad(vertices_small, centroid_small, image_shape=image_small.shape)
            M_small = cv2.getPerspectiveTransform(np.array(vertices_small, dtype="float32"), rect_small)
            
            # perspective transformation
            output_size_small = (image_small.shape[1], image_small.shape[0])
            warped_small = cv2.warpPerspective(image_small, M_small, output_size_small)

            # resize back to original size
            warped = cv2.resize(warped_small, (image_org.shape[1], image_org.shape[0]))


# TODO: Test all function
