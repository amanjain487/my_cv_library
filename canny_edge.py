import cv2
import numpy as np
import sobel as se

# with sobel edge detector, you get thick edges
# thin those edges by suppressing the surrounding pixels of an edge
# eg: consider an vertical edge with orientation = 90 degrees.
# note: these suppressions are done based on gradient direction
'''
0 0 1 2 1 0 ---> 0 0 0 2 0 0   
0 1 1 2 1 0 ---> 0 0 0 2 0 0
0 0 1 2 1 1 ---> 0 0 0 2 0 0
0 0 0 2 0 0 ---> 0 0 0 2 0 0
'''


# if gradient_direction is between 67.5 and 112.5 degrees, check pixels left and right
# if gradient direction is between 112.5 and 157.5 degrees, check top_right and bottom_left pixels
# and so on cover all possible all angles
def non_max_suppression(grad_mag, grad_orient, threshold):
    nms = np.zeros(grad_mag.shape)
    for i in range(0, grad_mag.shape[0] - 1):
        for j in range(0, grad_mag.shape[1] - 1):

            # to check if pixel value is less than threshold
            if grad_mag[i][j] < threshold:
                continue
            # if less, check for any strong pixel on the direction of gradient
            if grad_orient[i][j] == 0:
                if grad_mag[i][j] > grad_mag[i][j - 1] and grad_mag[i][j] >= grad_mag[i][j + 1]:
                    nms[i][j] = grad_mag[i][j]
            if grad_orient[i][j] == 135:
                if grad_mag[i][j] > grad_mag[i - 1][j + 1] and grad_mag[i][j] >= grad_mag[i + 1][j - 1]:
                    nms[i][j] = grad_mag[i][j]
            if grad_orient[i][j] == 90:
                if grad_mag[i][j] > grad_mag[i - 1][j] and grad_mag[i][j] >= grad_mag[i + 1][j]:
                    nms[i][j] = grad_mag[i][j]
            if grad_orient[i][j] == 45:
                if grad_mag[i][j] > grad_mag[i - 1][j - 1] and grad_mag[i][j] >= grad_mag[i + 1][j + 1]:
                    nms[i][j] = grad_mag[i][j]

    return nms


# pass 2 threshold values suppress the weak edges which may be noise keeps only the edges whose pixel value is
# greater than high_threshold value if a value is less than high_threshold and greater than low_threshold,
# then check if it is connected to any strong edge if less than high threshold and greater than low threshold and not
# connected to any strong edge, then supress that pixel i.e., make that pixel value as 0 if less than low_threshold,
# suppress that straight away
def double_threshold_linking(nms, low_threshold, high_threshold):
    hysteresis = np.zeros(nms.shape)

    # forward scan
    for i in range(0, nms.shape[0] - 1):  # rows
        for j in range(0, nms.shape[1] - 1):  # columns
            if nms[i, j] >= high_threshold:
                if nms[i, j + 1] >= low_threshold:  # right
                    nms[i, j + 1] = high_threshold
                if nms[i + 1, j + 1] >= low_threshold:  # bottom right
                    nms[i + 1, j + 1] = high_threshold
                if nms[i + 1, j] >= low_threshold:  # bottom
                    nms[i + 1, j] = high_threshold
                if nms[i + 1, j - 1] >= low_threshold:  # bottom left
                    nms[i + 1, j - 1] = high_threshold

    # backwards scan
    for i in range(nms.shape[0] - 2, 0, -1):  # rows
        for j in range(nms.shape[1] - 2, 0, -1):  # columns
            if nms[i, j] >= high_threshold:
                if nms[i, j - 1] > low_threshold:  # left
                    nms[i, j - 1] = high_threshold
                if nms[i - 1, j - 1]:  # top left
                    nms[i - 1, j - 1] = high_threshold
                if nms[i - 1, j] > low_threshold:  # top
                    nms[i - 1, j] = high_threshold
                if nms[i - 1, j + 1] > low_threshold:  # top right
                    nms[i - 1, j + 1] = high_threshold

    for i in range(0, nms.shape[0] - 1):  # rows
        for j in range(0, nms.shape[1] - 1):  # columns
            if nms[i][j] >= high_threshold:
                hysteresis[i][j] = 255

    return hysteresis


# canny_edge_function which performs all the above functions
def canny_edge(image, image_name, sigma, low_threshold, high_threshold):
    grad_mag, grad_orient = se.sobel_edge(image, image_name, sigma, low_threshold)

    # make thick edges as thin edges i.e., single line
    non_max = non_max_suppression(grad_mag, grad_orient, low_threshold)

    # suppress the weak edges
    final_edges = double_threshold_linking(non_max, low_threshold, high_threshold)
    # display the image with canny edges(almost similar to open_cv)
    if __name__ == "__main__":
        cv2.imshow("Canny Edges", final_edges)

    return final_edges


def param_passing():
    image_path = 'Samples/s2.jpg'  # <- insert image name here

    # the quality of edges depends on these parameters
    # try different parameters and see the changes in edges
    sigma = 0.5  # <- insert sigma value
    low_thresh = 0.0001  # <- insert low_threshold value (in the range of 0 to 1) since all the calculations are in
    # float, 0 to 1 will act as 0 t0 255 which is int8.
    high_thresh = 0.2  # <- insert high_threshold value

    image = cv2.imread(image_path)  # -> open the image

    image_path = image_path.replace("\\", "/")
    image_name = image_path.split("/")[-1]

    # run the Canny edge detector
    canny_edge(image, image_name, sigma, low_thresh, high_thresh)

if __name__ == "__main__":
    param_passing()
    cv2.waitKey(0)
