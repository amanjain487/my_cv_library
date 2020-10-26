import cv2
import numpy as np
from tabulate import  tabulate

# Step 1
def contrast_stretching(img):
    # convert to grayscale
    if len(img.shape) == 3:  # check if img is color
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # RGB to grayscale

    min_val = img.min()
    max_val = img.max()
    output = (img.astype('float') - min_val) / (max_val - min_val)

    return output


# Step 2
def get_gauss_kernel(img, sigma):
    # sobel kernel
    Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    mask_half_size = int(3 * sigma)
    mask_full_size = 2 * mask_half_size + 1

    mask_base = np.ones((mask_full_size, mask_full_size))

    mask_base = mask_base / (float)(2 * np.pi * (sigma ** 2))

    for i in range(-mask_half_size, mask_half_size + 1):
        for j in range(-mask_half_size, mask_half_size + 1):
            mask_base[i+mask_half_size, j+mask_half_size] *= np.exp(-((i**2 + j**2)/(2.0 * (sigma**2))))

    return mask_base

#applying gauss and then sobel filter is same as applying sobel to gauss and then applying the output mask to the image.
    #applying means convolving

def pad_zero(image):
    new_image = np.zeros([image.shape[0]+2, image.shape[1]+2])
    new_image[1:new_image.shape[0]-1, 1:new_image.shape[1]-1] = image
    return new_image



def convolve_gauss_to_sobel(gauss_kernel):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#convolve means flipping kernel twice, once horizontally and once vertically
    convolving_sobel_x = np.zeros_like(sobel_x)
    convolving_sobel_y = np.zeros_like(sobel_y)
    for i in range(sobel_x.shape[0]):
        for j in range(sobel_y.shape[1]):
            convolving_sobel_x[sobel_x.shape[0] - (1 + i), sobel_x.shape[1] - (1 + j)] = sobel_x[i, j]
            convolving_sobel_y[sobel_y.shape[0] - (1 + i), sobel_y.shape[1] - (1 + j)] = sobel_y[i, j]


#i assume that kernel should always be square matrix
    padded_gauss_kernel = pad_zero(gauss_kernel)

    gauss_sobel_x = np.zeros_like(padded_gauss_kernel)
    gauss_sobel_y = np.zeros_like(padded_gauss_kernel)

    for i in range(gauss_sobel_x.shape[0]):
        for j in range(gauss_sobel_y.shape[1]):
            gauss_sobel_x[i, j] = padded_gauss_kernel[i, j] * convolving_sobel_x[1,1]
            gauss_sobel_y[i, j] = padded_gauss_kernel[i, j] * convolving_sobel_y[1,1]
            if i > 0:
                if j > 0:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i - 1, j - 1] * convolving_sobel_x[0,0]
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i, j - 1] * convolving_sobel_x[1,0]

                    gauss_sobel_y[i, j] += padded_gauss_kernel[i - 1, j - 1] * convolving_sobel_y[0,0]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i, j - 1] * convolving_sobel_y[1,0]
                if j < gauss_sobel_x.shape[1]-1:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i - 1, j + 1] * convolving_sobel_x[0,2]
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i, j + 1] * convolving_sobel_x[1,2]

                    gauss_sobel_y[i, j] += padded_gauss_kernel[i - 1, j + 1] * convolving_sobel_y[0,2]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i, j + 1] * convolving_sobel_y[1,2]

                gauss_sobel_x[i, j] += padded_gauss_kernel[i - 1, j] * convolving_sobel_x[0,1]
                gauss_sobel_y[i, j] += padded_gauss_kernel[i - 1, j] * convolving_sobel_y[0,1]
            if i < gauss_sobel_x.shape[0] -1:
                if j > 0:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i + 1, j - 1] * convolving_sobel_x[2,0]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i + 1, j - 1] * convolving_sobel_y[2,0]
                if j < gauss_sobel_x.shape[1]-1:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i + 1, j + 1] * convolving_sobel_x[2,2]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i + 1, j + 1] * convolving_sobel_y[2,2]
                gauss_sobel_x[i, j] += padded_gauss_kernel[i + 1, j] * convolving_sobel_x[2,1]
                gauss_sobel_y[i, j] += padded_gauss_kernel[i + 1, j] * convolving_sobel_y[2,1]

    return gauss_sobel_x, gauss_sobel_y



def sobel_edges(image, kernel):

    sobel = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sub_mat = np.zeros_like(kernel)
            for x in range(-(kernel.shape[0]//2), kernel.shape[0]//2 + 1):
                for y in range(-(kernel.shape[1]//2), kernel.shape[1]//2+1):
                    new_i = i + x
                    new_j = j + y
                    if new_i >= image.shape[0]:
                        new_i = new_i - 2*(new_i - (image.shape[0]-1))
                    if new_j >= image.shape[1]:
                        new_j = new_j - 2*(new_j - (image.shape[1]-1))
                    sub_mat[x + kernel.shape[0]//2, y + kernel.shape[0]//2] = image[abs(new_i), abs(new_j)]
            sobel[i,j] = (sub_mat * kernel).sum()

    return sobel





# Step 3
def get_mag_and_orient(grad_x, grad_y, threshold):
    # compute magnitude
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # normalize magnitude image
    grad_mag = contrast_stretching(grad_mag)

    # compute orientation of gradient
    grad_orient = np.arctan2(grad_y, grad_x)


    for i in range(grad_orient.shape[0]):
        for j in range(grad_orient.shape[1]):
            if grad_mag[i, j] > threshold:
                # case 0
                if (grad_orient[i, j] > (- np.pi / 8) and grad_orient[i, j] <= (np.pi / 8)):
                    grad_orient[i, j] = 0
                elif (grad_orient[i, j] > (7 * np.pi / 8) and grad_orient[i, j] <= np.pi):
                    grad_orient[i, j] = 0
                elif (grad_orient[i, j] >= -np.pi and grad_orient[i, j] < (-7 * np.pi / 8)):
                    grad_orient[i, j] = 0
                # case 1
                elif (grad_orient[i, j] > (np.pi / 8) and grad_orient[i, j] <= (3 * np.pi / 8)):
                    grad_orient[i, j] = 45
                elif (grad_orient[i, j] >= (-7 * np.pi / 8) and grad_orient[i, j] < (-5 * np.pi / 8)):
                    grad_orient[i, j] = 45
                # case 2
                elif (grad_orient[i, j] > (3 * np.pi / 8) and grad_orient[i, j] <= (5 * np.pi / 8)):
                    grad_orient[i, j] = 90
                elif (grad_orient[i, j] >= (-5 * np.pi / 4) and grad_orient[i, j] < (-3 * np.pi / 8)):
                    grad_orient[i, j] = 90
                # case 3
                elif (grad_orient[i, j] > (5 * np.pi / 8) and grad_orient[i, j] <= (7 * np.pi / 8)):
                    grad_orient[i, j] = 135
                elif (grad_orient[i, j] >= (-3 * np.pi / 8) and grad_orient[i, j] < (-np.pi / 8)):
                    grad_orient[i, j] = 135

    return grad_mag, grad_orient


# Step 4
def non_max_suppression(grad_mag, grad_orient, threshold):
    nms = np.zeros(grad_mag.shape)
    for i in range(0, grad_mag.shape[0] - 1):
        for j in range(0, grad_mag.shape[1] - 1):
            if grad_mag[i][j] < threshold:
                continue
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


# Step 5
def double_threshold_linking(nms, low_threshold, high_threshold):
    hysterisis = np.zeros(nms.shape)

    # forward scan
    for i in range(0, nms.shape[0] - 1):  # rows
        for j in range(0, nms.shape[1] - 1):  # columns
            if nms[i,j] >= high_threshold:
                if nms[i,j + 1] >= low_threshold:  # right
                    nms[i,j + 1] = high_threshold
                if nms[i + 1,j + 1] >= low_threshold:  # bottom right
                    nms[i + 1,j + 1] = high_threshold
                if nms[i + 1,j] >= low_threshold:  # bottom
                    nms[i + 1,j] = high_threshold
                if nms[i + 1,j - 1] >= low_threshold:  # bottom left
                    nms[i + 1,j - 1] = high_threshold

    # backwards scan
    for i in range(nms.shape[0] - 2, 0, -1):  # rows
        for j in range(nms.shape[1] - 2, 0, -1):  # columns
            if nms[i,j] >= high_threshold:
                if nms[i,j - 1] > low_threshold:  # left
                    nms[i,j - 1] = high_threshold
                if nms[i - 1,j - 1]:  # top left
                    nms[i - 1,j - 1] = high_threshold
                if nms[i - 1,j] > low_threshold:  # top
                    nms[i - 1,j] = high_threshold
                if nms[i - 1,j + 1] > low_threshold:  # top right
                    nms[i - 1,j + 1] = high_threshold

    for i in range(0, nms.shape[0] - 1):  # rows
        for j in range(0, nms.shape[1] - 1):  # columns
            if nms[i][j] >= high_threshold:
                hysterisis[i][j] = 255

    return hysterisis


# Step 6
def canny_edge(img, image_name, sigma, low_threshold, high_threshold):

    cv2.imshow(image_name, img)
    new_image = contrast_stretching(img)
    gauss_kernel = get_gauss_kernel(new_image, sigma)

    sobel_kernel_x, sobel_kernel_y = convolve_gauss_to_sobel(gauss_kernel)

    gradient_x = sobel_edges(new_image, sobel_kernel_x)
    gradient_y = sobel_edges(new_image, sobel_kernel_y)

    grad_mag, grad_orient = get_mag_and_orient(gradient_x, gradient_y, low_threshold)

    non_max = non_max_suppression(grad_mag, grad_orient, low_threshold)

    final_edges = double_threshold_linking(non_max, low_threshold, high_threshold)
    cv2.imshow('Final_Edges_' + image_name, final_edges)


    return final_edges




image_path = 'Samples/Bottle/2.jpg'  # <- insert image name here
sigma = 0.5  # <- insert sigma value
low_threshold = 0.01  # <- insert low_threshold value (in the range of 0 to 1) since all the calculations are in float, 0 to 1 will act as 0 t0 255 which is int8.
high_threshold = 0.2  # <- insert high_threshold value

image = cv2.imread(image_path)

image_path = image_path.replace("\\","/")
image_name = image_path.split("/")[-1]


# run the Canny edge detector
edge_image = canny_edge(image, image_name, sigma, low_threshold, high_threshold)


# wait for esc to terminate
key = cv2.waitKey(0)
key = chr(key & 255)
if key == 27:
    cv2.destroyAllWindows()