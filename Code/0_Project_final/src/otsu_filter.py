class otsu_filter:
    def __init__(self):
        pass
    
    def get_otsu_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return threshold
    
    def apply_otsu_filter(self, img):
        threshold = self.get_otsu_threshold(img)
        img[threshold == 255] = 255
        return img
    
    def apply_otsu_filter_on_path(self, path):
        img = cv2.imread(path)
        return self.apply_otsu_filter(img)
    
    def apply_otsu_filter_on_image(self, img):
        return self.apply_otsu_filter(img)