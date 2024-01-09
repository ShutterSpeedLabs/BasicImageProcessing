import cv2
import numpy as np

# Initialize variables to store keypoints
keypoints_img1 = []
keypoints_img2 = []

def select_keypoints(event, x, y, flags, param):
    global keypoints_img1, keypoints_img2

    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point to the corresponding keypoints array
        keypoints_img1.append([x, y])
        #keypoints_img2.append([x + img1.shape[1], y])
        keypoints_img2.append([x , y])


        # Draw a circle and number at the clicked point on the combined image
        cv2.circle(combined_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(combined_image, str(len(keypoints_img1) - 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the updated image
        cv2.imshow("Select Keypoints", combined_image)

if __name__ == "__main__":
    # Load your images
    img1 = cv2.imread("ImageRegistration//FLIR_video_00075.jpeg")
    img2 = cv2.imread("ImageRegistration//FLIR_video_00083.jpeg")

    # Display images side by side for keypoint selection
    combined_image = np.hstack((img1, img2))
    cv2.imshow("Select Keypoints", combined_image)

    # Set the mouse callback function
    cv2.setMouseCallback("Select Keypoints", select_keypoints)

    # Wait for keypoint selection
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert keypoints to NumPy arrays
    keypoints_img1 = np.array(keypoints_img1)
    keypoints_img2 = np.array(keypoints_img2)
    print("Image1 keypoints are: ", keypoints_img1, "\nImage 2 keypoints are: ", keypoints_img2)

    # Use keypoints to compute the transformation matrix
    transformation_matrix, _ = cv2.findHomography(keypoints_img1, keypoints_img2, cv2.RANSAC)
    print(transformation_matrix)

    # Warp img1 to align with img2
    registered_img = cv2.warpPerspective(img1, transformation_matrix, (img2.shape[1], img2.shape[0]))

    cv2.imshow("Registered Image", registered_img)
    cv2.waitKey(0)
    # Display overlapped images with alpha blending
    result = cv2.addWeighted(img2, 0.5, registered_img, 0.5, 0)

    cv2.imshow("Image Registration Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
