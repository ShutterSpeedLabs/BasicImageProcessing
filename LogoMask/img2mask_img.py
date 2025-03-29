import cv2
import numpy as np

class RectangleSelector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not load the image")
        
        self.original = self.image.copy()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_rectangle)
        
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.image = self.original.copy()
                cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            
    def create_mask(self):
        if self.fx == -1 or self.fy == -1:
            return None
        
        # Create black mask of same size as original image
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        # Calculate top-left and bottom-right points
        x1, y1 = min(self.ix, self.fx), min(self.iy, self.fy)
        x2, y2 = max(self.ix, self.fx), max(self.iy, self.fy)
        
        # Fill rectangle with white in the mask
        mask[y1:y2, x1:x2] = 255
        return mask
    
    def run(self):
        while True:
            cv2.imshow('image', self.image)
            key = cv2.waitKey(1) & 0xFF
            
            # Press 'm' to create and show mask
            if key == ord('m'):
                mask = self.create_mask()
                if mask is not None:
                    cv2.imshow('mask', mask)
                    cv2.imwrite('mask.png', mask)
            
            # Press 'q' to quit
            elif key == ord('q'):
                break
                
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your image path
    image_path = "/media/kisna/bkp_data/DeOldify/video_data/video_out/rrtn/test_results_30_rec2/test_data_sample/video_1/00001.png"
    selector = RectangleSelector(image_path)
    selector.run()