class image_os:
    
    def __init__(self):
        pass
    
    def read_image(self, path):
        return cv2.imread(path)
    
    # draw polygon on new png image with the same size as the original image
    def draw_polygon(self, img, polygon):
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], (255, 255, 255))
        return mask
    
    # save the mask as a png image
    def save_mask(self, mask, path):
        cv2.imwrite(path, mask)