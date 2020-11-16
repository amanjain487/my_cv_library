# This module contains the basic functions
# of Image Processing involved in solving Sudoku Problem

import cv2
import numpy as np
import math

# --------------------------------------------------------------------------------------------------------------------#
# 1. Contrast Stretching or Normalization
# To normalize the image or improve the contrast of the image
# i.e., stretching the range of intensity values to span the desired range specified by (min, max)

# input -> Grayscale Image
# output -> Grayscale Image

# formula
'''
output_pixel_value = ((input_pixel_value - smallest pixel value already present in image) * 
                        ((max - min)/(largest pixel value in image - smallest pixel value))) + min
'''


# where max and min are the range of pixel values of output image
def contrast_stretching(img, maxi=1, mini=0):
    # convert to grayscale
    if len(img.shape) == 3:  # check if img is color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB to grayscale

    # by default, max - min = 1
    min_val = img.min()  # <- smallest pixel value already present in image
    max_val = img.max()  # <- largest pixel value in image
    output = ((img.astype('float') - min_val) * (maxi - mini)) / (max_val - min_val) + mini

    return output


# --------------------------------------------------------------------------------------------------------------------#

# 2. Create kernel for applying Gaussian Filter
# to reduce noise.
# input -> sigma value
# output -> gauss_filter kernel

# formula
'''
kernel[x,y] = (1/(2*pi*sigma^2)) * e^(-((x^2 + y^2)/(2*sigma^2))) 
'''


# kernel size can be specified explicitly, but I have determined it based on sigma value.
def get_gauss_kernel(sigma, mask_size=3):
    mask_half_size = int(mask_size * sigma)  # <-determine half size
    mask_full_size = 2 * mask_half_size + 1  # <-double the size and add 1, so that mask dimensions are odd

    # create a kernel with all 1's.
    mask_base = np.ones((mask_full_size, mask_full_size))

    # divide by constant 2*pi*sigma^2
    mask_base = mask_base / float(2 * np.pi * (sigma ** 2))

    # multiply by the remaining term specified in the formula above.
    for i in range(-mask_half_size, mask_half_size + 1):
        for j in range(-mask_half_size, mask_half_size + 1):
            mask_base[i + mask_half_size, j + mask_half_size] *= np.exp(-((i ** 2 + j ** 2) / (2.0 * (sigma ** 2))))

    return mask_base


# --------------------------------------------------------------------------------------------------------------------#

# 3. Apply gaussian filter to smoothen the image
# to remove noise
# cross-correlation of image and kernel
# input -> image, gaussian filter
# output -> smoothened image
def apply_gaussian_filter(source, kernel):
    size = kernel.shape[0]
    # create output image
    gauss = source.copy()
    # cross correlate kernel on ech pixel
    for i in range(size // 2, source.shape[0] - (size // 2)):
        for j in range(size // 2, source.shape[1] - (size // 2)):
            value = 0
            div = 0
            for a in range(-(size // 2), (size // 2) + 1):
                for b in range(-(size // 2), (size // 2) + 1):
                    try:
                        value += (kernel[a + (size // 2), b + (size // 2)] * source[i + a, j + b])
                        div += kernel[a + (size // 2), b + (size // 2)]
                    # handle index error
                    except IndexError:
                        continue
            # assign smoothened pixel value
            gauss[i, j] = value // div
    # return smoothened image
    return gauss


# --------------------------------------------------------------------------------------------------------------------

# 4. padding extra rows and cols to image
# the above process is done, so that while convolving we don't lose dimensions,
# because convolution on nxn image with fxf filter gives n-f+1xn-f+1.
# so to avoid that we pad f-1 extra rows and cols.
# input -> image to be padded and padding size(default = 1)
# output -> padded image
def pad_zero(image, padding_size=1):
    temp_image = np.copy(image)  # <-copy and store the image
    # create image with required size with all pixel values 0
    new_image = np.zeros([temp_image.shape[0] + 2 * padding_size, temp_image.shape[1] + 2 * padding_size])
    # copy the existing pixels onto the new image with extra added pixels set as zero
    new_image[1 * padding_size:new_image.shape[0] - 1, 1:new_image.shape[1] - 1 * padding_size] = image

    return new_image


# --------------------------------------------------------------------------------------------------------------------#

# 5. find the largest object detected by edges
# input -> list of traced edges
# output -> longest traced edge
def largest_boundary(list_of_boundaries):
    max_length = 0
    max_path = []
    # count the number of edge pixel in each boundary
    for i in list_of_boundaries:
        for j in i:
            if len(j) > max_length:
                max_path = j
                max_length = len(j)
    # return the one with most number of edge pixels.
    return max_path


# ---------------------------------------------------------------------------------------------------------------------#

# 6. Object Extraction
# remove all other pixels i.e., make all other pixel values other than the specified object as 0
# the specified object must be filled, i.e., it must be complete path(starting point and ending point should be same)
# input -> image from which object has to be extracted, boundary which tells the edges of the object
# output -> image which contains only object and all other pixel values as 0
def extract_object(image, boundary):
    image = image.copy()
    # make all pixels as zero
    output_image = np.zeros_like(image)
    # make only edges pixels as 255
    for i in boundary:
        output_image[i[0], i[1]] = 255
    res1 = np.copy(output_image)
    # from all 4 sides, make all pixels as zero, until edge pixel is encountered
    l_to_r = np.copy(output_image)
    r_to_l = np.copy(output_image)
    t_to_b = np.copy(output_image)
    b_to_t = np.copy(output_image)
    # left_to_right ----->
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # if edge pixel encountered, break out of the loop
            if l_to_r[i, j] == 255:
                break
            # else, make that pixel as black
            else:
                l_to_r[i, j] = 255
    # right_to_left <-------
    for i in range(image.shape[0] - 1, -1, -1):
        for j in range(image.shape[1] - 1, -1, -1):
            if r_to_l[i, j] == 255:
                break
            else:
                r_to_l[i, j] = 255
    # top_to bottom
    '''
    |
    |
    |
    |
    v
    '''
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            if t_to_b[i, j] == 255:
                break
            else:
                t_to_b[i, j] = 255
    # bottom_to_top
    '''
    ^
    |
    |
    |
    '''
    for j in range(image.shape[1] - 1, -1, -1):
        for i in range(image.shape[0] - 1, -1, -1):
            if b_to_t[i, j] == 255:
                break
            else:
                b_to_t[i, j] = 255
    # add all the 4 images, so we get black pixels from all sides.
    res = r_to_l + l_to_r + t_to_b + b_to_t
    # initially we made edge pixel as all white, turn them to their original colour.
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if res[i, j] > 0 and res1[i, j] == 0:
                image[i, j] = 0
    # return the image with extracted object
    return image


# ------------------------------------------------------------------------------------------------------------------#

# 7. thicken the boundary by adding pixels to boundary
# look around a pixel using kernel
# assign the largest nearby value to that pixel
# hence the boundary will be thickened
# example
'''
image is :
0 0 0 0 0
0 0 4 5 0
0 3 5 0 0
0 2 3 4 0
0 0 0 0 0

kernel is:
1 1
1 1 

output will be:
0 4 5 5 0
3 5 5 5 0
3 5 5 4 0
2 3 4 4 0
0 0 0 0 0
'''


# input -> image with boundaries, kernel(which nearby pixel should be looked), no of iteration(how much to thicken)
# output -> image with thickened edges
def thicken_boundary(source, kernel, iterations=1):
    size = kernel.shape
    # create output image
    output_image = np.copy(source)
    # thicken edges for specified no of iterations
    # default is 1
    for k in range(iterations):
        output_image = np.copy(source)
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                # look for the maximum value based on kernel
                maximum = -1
                for a in range(-(size[0] // 2), size[0] // 2 + 1):
                    for b in range(-(size[1] // 2), size[1] // 2 + 1):
                        try:
                            sub = source[(i + a), (j + b)] * kernel[a + size[0] // 2, b + size[1] // 2]
                            if sub > maximum:
                                maximum = sub
                        except IndexError:
                            continue
                # assign the maximum value
                output_image[i, j] = maximum
        # make current output as source for next_iteration
        source = output_image.copy()
    # return image with thickened edges/boundary
    return output_image


# -------------------------------------------------------------------------------------------------------------------#

# 8. thinning the boundary by removing pixels to boundary
# look around a pixel using kernel
# assign the smallest nearby value to that pixel
# hence the boundary will be thinned
# example
'''
image is :
0 0 0 0 0
0 5 4 5 0
0 3 5 3 0
0 2 3 4 0
0 0 0 0 0

kernel is:
1 1
1 1 

output will be:
0 0 0 0 0
0 0 0 0 0
0 2 3 3 0
0 0 0 0 0
0 0 0 0 0
'''


# input -> image with boundaries, kernel(which nearby pixel should be looked), no of iteration(how much to thin)
# output -> image with thinner edges


def thinning_boundary(source, kernel, iterations=1):
    size = kernel.shape
    # create output image
    output_image = np.copy(source)
    # thinning edges for specified no of iterations
    # default is 1
    for k in range(iterations):
        output_image = np.copy(source)
        for i in range(source.shape[0]):
            for j in range(source.shape[1]):
                # look for the minimum value based on kernel
                minimum = 255
                for a in range(-(size[0] // 2), size[0] // 2 + 1):
                    for b in range(-(size[1] // 2), size[1] // 2 + 1):
                        try:
                            sub = source[(i + a), (j + b)] * kernel[a + size[0] // 2, b + size[1] // 2]
                            if sub < minimum:
                                minimum = sub
                        except IndexError:
                            continue
                # assign the minimum value
                output_image[i, j] = minimum
        # make current output as source for next_iteration
        source = output_image.copy()
    # return image with thinned edges/boundary
    return output_image


# ------------------------------------------------------------------------------------------------------------------#

# 9. get kernel for thickening ot thinning the edges/boundary
# input -> size of kernel
# output -> kernel as specified by input parameters
def get_structuring_kernel(size):
    kernel = []
    for i in range(size[0]):
        kernel.append([1 for _ in range(size[1])])
    kernel = np.array(kernel)
    return kernel


# ------------------------------------------------------------------------------------------------------------------#

# 10. converting image to binary image based on nearby pixel values
# input -> source image, value to assign to a pixel if certain condition is satisfied, size of kernel and bias
# output -> binary image
# if the mean of pixel values determined by kernel minus bias is less than source pixel value
# assign changing value(mostly 255)
# else assign 0
def convert_to_binary(source, size, bias=0, changing_value=255):
    # create kernel of size specified by input parameter size
    kernel = np.ones((size, size))
    output_image = np.zeros_like(source)
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            value = 0
            div = 0
            for a in range(-(size // 2), (size // 2) + 1):
                for b in range(-(size // 2), (size // 2) + 1):
                    try:
                        value += (kernel[a + (size // 2), b + (size // 2)] * source[i + a, j + b])
                        div += kernel[a + (size // 2), b + (size // 2)]
                    except IndexError:
                        continue
            # if condition met, assign changing value
            if source[i, j] >= (value / div) - bias:
                output_image[i, j] = changing_value
            # else assign 0
            else:
                output_image[i, j] = 0
    # return binary image
    return output_image


# ------------------------------------------------------------------------------------------------------------------#

# 11. Find centroid of object in image
# the image is expected to contain only one object
# input -> image with only 1 object extracted using extract object function
# output -> centroid of that object
def calculate_centroid(image):
    x = 0
    y = 0
    n = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] != 0:
                n += 1
                x += i
                y += j

    return [round(y / n), round(x / n)]

# ------------------------------------------------------------------------------------------------------------------#


# ========================================== SOBEL EDGES DETECTOR=====================================================

# If you want to apply gauss to an image and then apply Sobel filter
# Then, 1st apply Sobel to gauss filter and then apply the result on the image
# Result will be the same, but it will require less computation.
# input -> gauss_kernel(gauss_filter), sobel_x(vertical edge detector kernel), sobel_y(horizontal edge detector kernel)
# output -> 2 filters
# 1. gauss applied to vertical edge detector
# 2. gauss applied to horizontal edge detector
def convolve_gauss_to_sobel(gauss_kernel, sobel_x=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                            sobel_y=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])):
    # convolve means flipping kernel twice, once horizontally and once vertically
    convolving_sobel_x = np.zeros_like(sobel_x)
    convolving_sobel_y = np.zeros_like(sobel_y)
    for i in range(sobel_x.shape[0]):
        for j in range(sobel_y.shape[1]):
            # flipping vertical edge detector
            convolving_sobel_x[sobel_x.shape[0] - (1 + i), sobel_x.shape[1] - (1 + j)] = sobel_x[i, j]
            # flipping horizontal edge detector
            convolving_sobel_y[sobel_y.shape[0] - (1 + i), sobel_y.shape[1] - (1 + j)] = sobel_y[i, j]

    # assuming that kernel will always be square matrix
    # padding the gauss kernel, so that kernel_size isn't reduced
    padded_gauss_kernel = pad_zero(gauss_kernel)
    # final kernel with all elements as zero currently
    gauss_sobel_x = np.zeros_like(padded_gauss_kernel)
    gauss_sobel_y = np.zeros_like(padded_gauss_kernel)

    # calculate the values of the final filter
    # after applying Sobel to gauss
    # applying here means convolution.
    for i in range(gauss_sobel_x.shape[0]):
        for j in range(gauss_sobel_y.shape[1]):
            gauss_sobel_x[i, j] = padded_gauss_kernel[i, j] * convolving_sobel_x[1, 1]
            gauss_sobel_y[i, j] = padded_gauss_kernel[i, j] * convolving_sobel_y[1, 1]
            # the following conditions are to manage index errors.
            if i > 0:
                if j > 0:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i - 1, j - 1] * convolving_sobel_x[0, 0]
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i, j - 1] * convolving_sobel_x[1, 0]

                    gauss_sobel_y[i, j] += padded_gauss_kernel[i - 1, j - 1] * convolving_sobel_y[0, 0]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i, j - 1] * convolving_sobel_y[1, 0]
                if j < gauss_sobel_x.shape[1] - 1:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i - 1, j + 1] * convolving_sobel_x[0, 2]
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i, j + 1] * convolving_sobel_x[1, 2]

                    gauss_sobel_y[i, j] += padded_gauss_kernel[i - 1, j + 1] * convolving_sobel_y[0, 2]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i, j + 1] * convolving_sobel_y[1, 2]

                gauss_sobel_x[i, j] += padded_gauss_kernel[i - 1, j] * convolving_sobel_x[0, 1]
                gauss_sobel_y[i, j] += padded_gauss_kernel[i - 1, j] * convolving_sobel_y[0, 1]
            if i < gauss_sobel_x.shape[0] - 1:
                if j > 0:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i + 1, j - 1] * convolving_sobel_x[2, 0]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i + 1, j - 1] * convolving_sobel_y[2, 0]
                if j < gauss_sobel_x.shape[1] - 1:
                    gauss_sobel_x[i, j] += padded_gauss_kernel[i + 1, j + 1] * convolving_sobel_x[2, 2]
                    gauss_sobel_y[i, j] += padded_gauss_kernel[i + 1, j + 1] * convolving_sobel_y[2, 2]
                gauss_sobel_x[i, j] += padded_gauss_kernel[i + 1, j] * convolving_sobel_x[2, 1]
                gauss_sobel_y[i, j] += padded_gauss_kernel[i + 1, j] * convolving_sobel_y[2, 1]

    # filter which contains Sobel applied to gauss, which can be applied to any image
    # the result after applying this filter to any image will result in edges detected using Sobel after applying gauss
    return gauss_sobel_x, gauss_sobel_y


# Apply cross-correlation of input kernel on the input image.
# apply Sobel filter(any of the edge detector filter) to image
# finding edges using Sobel edge detector kernel on which Gaussian filter is applied
# you can skip applying a gaussian filter by directly passing the Sobel kernel
# Input -> image, edge detector kernel
# output -> image with edges(horizontal or vertical based on input kernel)

def sobel_edges(image, kernel):
    # convolve sobel_edge_detector to image
    # same as applying differentiation or finding gradients
    # i.e., partial differentiation
    # w.r.t to x when using vertical edge detector kernel
    # w.r.t to y when using horizontal edge detector kernel

    # create output image with all pixel values zero
    sobel = np.zeros_like(image)

    # apply filter to image
    # here, applying means cross-correlation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # create sub_array to apply kernel on that sub_array
            sub_mat = np.zeros_like(kernel)
            # perform cross-correlation
            for x in range(-(kernel.shape[0] // 2), kernel.shape[0] // 2 + 1):
                for y in range(-(kernel.shape[1] // 2), kernel.shape[1] // 2 + 1):
                    new_i = i + x
                    new_j = j + y
                    if new_i >= image.shape[0]:
                        new_i = new_i - 2 * (new_i - (image.shape[0] - 1))
                    if new_j >= image.shape[1]:
                        new_j = new_j - 2 * (new_j - (image.shape[1] - 1))
                    sub_mat[x + kernel.shape[0] // 2, y + kernel.shape[0] // 2] = image[abs(new_i), abs(new_j)]
            # add the sub array elements and divide by no.of elements
            sobel[i, j] = (sub_mat * kernel).sum()
    # image with edges based on input edge detector kernel
    return sobel


# finding gradient magnitude and gradient angle
# grad magnitude = square_root(square(sobel_x applied to image) + square(sobel_y applied to image))
# grad_direction = tan_inverse(sobel_y applied to image/sobel_x applied to image)
# input -> vertical edges, horizontal edges, threshold for gradient_magnitude
# output -> gradient_magnitude and gradient_direction
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
                if (- np.pi / 8) < grad_orient[i, j] <= (np.pi / 8):
                    grad_orient[i, j] = 0
                elif (7 * np.pi / 8) < grad_orient[i, j] <= np.pi:
                    grad_orient[i, j] = 0
                elif -np.pi <= grad_orient[i, j] < (-7 * np.pi / 8):
                    grad_orient[i, j] = 0
                # case 1
                elif (np.pi / 8) < grad_orient[i, j] <= (3 * np.pi / 8):
                    grad_orient[i, j] = 45
                elif (-7 * np.pi / 8) <= grad_orient[i, j] < (-5 * np.pi / 8):
                    grad_orient[i, j] = 45
                # case 2
                elif (3 * np.pi / 8) < grad_orient[i, j] <= (5 * np.pi / 8):
                    grad_orient[i, j] = 90
                elif (-5 * np.pi / 4) <= grad_orient[i, j] < (-3 * np.pi / 8):
                    grad_orient[i, j] = 90
                # case 3
                elif (5 * np.pi / 8) < grad_orient[i, j] <= (7 * np.pi / 8):
                    grad_orient[i, j] = 135
                elif (-3 * np.pi / 8) <= grad_orient[i, j] < (-np.pi / 8):
                    grad_orient[i, j] = 135

    return grad_mag, grad_orient


# sobel_edge_function which performs all the above functions
def sobel_edge(img, image_name, sigma, threshold):
    cv2.imshow(image_name, img)
    # normalizing the image
    new_image = contrast_stretching(img)
    # find gauss_kernel for smoothening the image
    gauss_kernel = get_gauss_kernel(sigma)
    # find sobel_kernels after applying gauss
    sobel_kernel_x, sobel_kernel_y = convolve_gauss_to_sobel(gauss_kernel)

    # find vertical and horizontal edges
    gradient_x = sobel_edges(new_image, sobel_kernel_x)
    gradient_y = sobel_edges(new_image, sobel_kernel_y)

    # find gradient_magnitude and gradient_orientation
    grad_mag, grad_orient = get_mag_and_orient(gradient_x, gradient_y, threshold)

    # display the image with sobel edges(almost similar to open_cv)
    if __name__ == "__main__":
        cv2.imshow('Final_Edges_' + image_name, grad_mag)

    return grad_mag, grad_orient


# ----------------------------------------------------------------------------------------------------------------------

# ======================================= CANNY EDGE DETECTOR ==========================================================

# with sobel edge detector, you get thick edges
# thin those edges by suppressing the surrounding pixels of an edge
# eg: consider an vertical edge with orientation = 90 degrees.
# note: these suppression are done based on gradient direction
'''
0 0 1 2 1 0 ---> 0 0 0 2 0 0   
0 1 1 2 1 0 ---> 0 0 0 2 0 0
0 0 1 2 1 1 ---> 0 0 0 2 0 0
0 0 0 2 0 0 ---> 0 0 0 2 0 0
'''


# if gradient_direction is between 67.5 and 112.5 degrees, check pixels left and right
# if gradient direction is between 112.5 and 157.5 degrees, check top_right and bottom_left pixels
# and so on cover all possible all angles
# input -> gradient_magnitude, gradient_orientation and threshold for gradient_magnitude
# output -> thin_edges
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
# then check if it is connected to any strong edge if less than the high threshold and greater than the low threshold and not
# connected to any strong edge, then suppress that pixel i.e., make that pixel value as 0 if less than low_threshold,
# suppress that straight away
# input -> image with thin edges, low_threshold, high_threshold
# output -> image with canny edges
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
def canny_edge(image, image_name, sigma=0.5, low_threshold=0.0001, high_threshold=0.2):
    grad_mag, grad_orient = sobel_edge(image, image_name, sigma, low_threshold)

    # make thick edges as thin edges i.e., single line
    non_max = non_max_suppression(grad_mag, grad_orient, low_threshold)

    # suppress the weak edges
    final_edges = double_threshold_linking(non_max, low_threshold, high_threshold)
    # display the image with canny edges(almost similar to open_cv)
    if __name__ == "__main__":
        cv2.imshow("Canny Edges", final_edges)

    return final_edges


# ----------------------------------------------------------------------------------------------------------------------

# ==============================================BOUNDARY TRACING=======================================================
# trace the boundaries of objects detected by edge detector
# in other words, trace the edges.


# rotate/shift list based on specified index
# input -> list, index for shift
# output -> shifted/rotated list
def shift(offset, index):
    rotated_list = []
    for i in range(len(offset)):
        rotated_list.append(offset[(i + index) % len(offset)])
    return rotated_list


# trace the boundary/edge when a pixel is part of edge
# input -> image with edges, current_path which keeps getting updated, previous_pixel_coordinate which is part of edge,
# coordinate to start looking for next edge pixel, general_offset(list of coordinates to look for edge pixel)
# output -> image, path/boundary
def trace_further_boundary(image, path, prev, off_start, general_offset):
    # start = path[0]
    # prev = last value that was added in path
    # off_start = where to go from prev to start the iteration

    while True:
        # performing shift operation to arrange the nearby pixels in clockwise order with off_start as a starting point.
        index = 0
        for k in general_offset:
            if k == off_start:
                break
            index += 1
        # perform shift operation or rotating list at specified index
        curr_offset = shift(general_offset, index)
        # now we have all nearby pixels in clockwise order and in which order they must be searched for edge pixel
        stop = 1
        # looking each coordinate for if it is edge.
        for k in range(len(curr_offset)):

            i = prev[0] + curr_offset[k][0]
            j = prev[1] + curr_offset[k][1]
            try:
                if image[i, j] != 0:
                    now = [i, j]
                    # if current_pixel is same as second pixel in path, then path is complete, return the image and path
                    if len(path) > 1 and now == path[1]:
                        return image, path
                    else:
                        path.append(now)

                    # calculate the parameters for finding next pixel
                    off_i = prev[0] + curr_offset[k - 1][0]
                    off_j = prev[1] + curr_offset[k - 1][1]

                    off_start = [off_i - i, off_j - j]
                    prev = [i, j]
                    stop = 0
                    break
            except IndexError:
                continue

        # if no nearby pixel is edge image, stop the tracing and return the path.
        # it means, it was not filled object
        if stop == 1:
            break

    return image, path


# trace the boundary
# check for nearby edge pixel in the clockwise direction,
# if found, add it to the path,
# else, end that boundary and store the path in the index of the beginning pixel of the path
# use Dynamic Programming to avoid retracing of already traced objects
# input -> image with canny/Sobel or any edges
# output -> boundary/path of the object present at each pixel, so no of path = no. of pixels in the image
# if a pixel does not contain any edge, then add an empty list as a path.
def trace_boundary(image):
    # possible nearby pixels containing edges for [0,0].
    general_offset = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    boundary = []
    # for each pixel add the coordinate of that pixel as path in boundary list
    for i in range(image.shape[0]):
        new_l = []
        for j in range(image.shape[1]):
            new_l.append([[i, j]])
        boundary.append(new_l)

    # for each pixel, check and trace the edges
    # if edge is found in any of the nearby pixels specified by points in general_offset list
    # then continue tracing, else replace existing path with empty list
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # proceed only if the pixel in consideration is part of edge
            # else proceed to next pixel
            if image[i, j] != 0 and image[i, j - 1] == 0 and image[i - 1, j - 1] == 0 and image[i - 1, j] == 0:
                prev = [i, j]
                off_start = [0, -1]
                # trace the boundary if pixel in consideration is part of edge
                image, boundary[i][j] = trace_further_boundary(image, boundary[i][j], prev, off_start, general_offset)

            # if non-edge pixel, replace existing path with empty list
            if len(boundary[i][j]) <= 1:
                boundary[i][j] = []
    # list of all paths i.e., all objects traced
    return boundary


# ----------------------------------------------------------------------------------------------------------------------

# =======================================FINDING CORNERS OF OBJECT=====================================================
# ----------------- to find perspective transformation matrix, these corners are required


# find 4 corners by approximating the object to a trapezium
# not an actual approximation,
# but if there are 2 points with more than allowed curvature, one of them is one of the 4 corners.
# input -> path/boundary of the object whose corners are to be found, allowed_curvature
# output -> 4 corners which are not sorted

# Procedure:
# find two farthest corners
# check if their curvature is less than allowed curvature,
# if yes, append one of those point as corner and that is it.
# if no, split the whole path into 2,
# suppose the two farthest points are at index i and index j in path list and i<j.
# index i to index j-1 will be one line
# index j to index i-1%path_length will be the other line
# append these two in a list
# take one of them and repeat above steps until the list is empty and all lines are validated
def approximate_the_object(path, allowed_curvature):
    lines = []
    corners = []
    is_valid_line = False
    path_length = len(path)
    allowed_curvature = allowed_curvature * allowed_curvature
    index = 0
    second_half_starting_point = 0
    starting_point = []
    # 3 iterations to be sure about the farthest points
    # because initially 1st point is 1st point in path and 2nd point is also the same
    # so perform 3 iterations so that the distance between them is max will be verified
    for i in range(3):
        max_distance = 0
        index = (index + second_half_starting_point) % path_length
        starting_point = path[index]
        index = (index + 1) % path_length

        # find second_half_starting_point w.r.t. 1st point in path
        for j in range(1, len(path)):
            current_point = path[index]
            index = (index + 1) % path_length
            # calculate difference between x and y distance
            x_distance = current_point[0] - starting_point[0]
            y_distance = current_point[1] - starting_point[1]
            # calculate x^2 + y^2
            # taking square root will not make diff, as we don't want distance, but points whose distance is maximum,
            distance = x_distance * x_distance + y_distance * y_distance

            if distance > max_distance:
                max_distance = distance
                second_half_starting_point = j
        # distance between those two points should be greater than allowed curvature,
        # else no corners will be found since the farthest point themselves are within the allowed curvature limit.
        is_valid_line = max_distance <= allowed_curvature

    # if curvature is more than allowed curvature
    # split the path into 2 lines
    # push them in lines list
    if not is_valid_line:
        second_half_ending_point = first_half_starting_point = index % path_length
        second_half_starting_point = (second_half_starting_point + first_half_starting_point) % path_length
        first_half_ending_point = second_half_starting_point
        # whole path broken into 2 lines
        lines.append([second_half_starting_point, second_half_ending_point])
        lines.append([first_half_starting_point, first_half_ending_point])

    # if curvature is less than allowed curvature, append starting_point to corners
    # and that is the only possible corner for specified allowed_curvature
    else:
        corners.append(starting_point)
    # now consider each line in lines list
    # and check for allowed curvature for each line
    # if within limits, then append starting point as corner
    # else break the line into further two and add them in lines list
    # repeats until lines list is empty
    while lines:

        line_in_consideration = lines.pop()
        end = path[line_in_consideration[1]]
        index = line_in_consideration[0]
        starting_point = path[index]
        index = (index + 1) % path_length
        first_half_starting_point = line_in_consideration[0]
        second_half_ending_point = line_in_consideration[1]
        # if the two points are not consecutive
        if index != line_in_consideration[1]:
            x_distance = end[0] - starting_point[0]
            y_distance = end[1] - starting_point[1]
            max_distance = 0
            while index != line_in_consideration[1]:
                current_point = path[index]
                index = (index + 1) % path_length

                distance = math.fabs((current_point[1] - starting_point[1]) * x_distance - (
                        current_point[0] - starting_point[0]) * y_distance)

                if distance > max_distance:
                    max_distance = distance
                    second_half_starting_point = (index + path_length - 1) % path_length
            is_valid_line = max_distance * max_distance <= allowed_curvature * (
                    x_distance * x_distance + y_distance * y_distance)

        else:
            is_valid_line = True
            starting_point = path[line_in_consideration[0]]

        # if the curvature is less than allowed_curvature, append starting point as a corner
        if is_valid_line:
            if starting_point not in corners:
                corners.append(starting_point)
        # else split the line in two and append them to lines list.
        else:
            first_half_ending_point = second_half_starting_point
            lines.append([second_half_starting_point, second_half_ending_point])
            lines.append([first_half_starting_point, first_half_ending_point])
    # corners contains the 4 corners
    return corners


# find corners of an object
# the 1st element in the path and last element will be the same if the object is filled
# input -> path/boundary of the object for which corners are to be found, an image containing the object
# output -> 4 corners after approximating the object to trapezium.
# this function is based on important parameter allowed_curvature
# which in turn is based on the perimeter of the object
# perimeter is calculated by finding the length of the path,
# allowed_curvature tells what is the maximum allowed curvature
# if the curvature between any 2 points is more than the allowed curvature, then 1 of them is considered as the essential corner
def find_corners(path, image):
    approximate_perimeter = len(path)
    allowed_curvature = approximate_perimeter * 0.15
    # find the 4 corners
    corners = approximate_the_object(path, allowed_curvature)
    # find the centroid to sort the corners
    # i.e, which is bottom right corner, which is top left and so on.
    centroid_x = 0
    centroid_y = 0
    # below calculated is approximate to actual centroid
    # used formula is
    # add all the x coordinates and divide by no.of coordinates
    # same for y
    for i in path:
        centroid_x += i[0]
        centroid_y += i[1]
    centroid_x = int(centroid_x / len(path))
    centroid_y = int(centroid_y / len(path))

    # initialize the 4 corners to 4 corners of image(Note: image, not object)
    topleft = [image.shape[0] - 1, image.shape[1] - 1]
    topright = [image.shape[0] - 1, 0]
    botleft = [0, image.shape[1] - 1]
    botright = [0, 0]

    # assign corners of object to above variables based on centroid calculated above
    for i in corners:
        if i[0] < centroid_x and i[1] < centroid_y:
            topleft = i  # <-top left corner

        elif i[0] < centroid_x and i[1] > centroid_y:
            topright = i  # <-top right corner

        elif i[0] > centroid_x and i[1] < centroid_y:
            botleft = i  # <- bottom left corner

        elif i[0] > centroid_x and i[1] > centroid_y:
            botright = i  # <- bottom right corner

    # return the 4 corners in the specified order
    return [topleft, topright, botright, botleft]


# --------------------------------------------------------------------------------------------------------------------

# =======================================PERSPECTIVE TRANSFORMATION MATRIX============================================
# Solve Ax = b by elimination method

# exchange rows when the pivot element is zero,
# input -> A(in which nth row has to be exchanged with rows below),
# b(in which nth row has to be exchanged too), n(row to be exchanged).
# output -> exchanged A and exchanged b
def exchange(a, b, n):
    exchanged = False
    # find non zero pivot element below the nth row
    for i in range(n + 1, len(b)):
        # if found, exchange them, do the same exchange on b too.
        if a[i, n] != 0:
            temp = a[n].copy()
            a[n] = a[i].copy()
            a[i] = temp.copy()

            temp = b[n].copy()
            b[n] = b[i].copy()
            b[i] = temp.copy()

            exchanged = True
            break
    # if exchange done, return them
    if exchanged:
        return a, b
    # if not, then this does not have unique solution, so return empty arrays
    else:
        return np.array([]), np.array([])


# find x in Ax = b, once A and b are row reduced
# input -> row reduced A, row reduced b, empty array for matrix co-efficients
# output -> array of matrix co-efficients
def compute(a, b, answers):
    if len(answers) == 0:
        answers = np.array([b[0] / a[-1]])
    else:
        denominator = 0
        for i in range(len(answers)):
            denominator += a[-(i + 1)] * answers[-(i + 1)]

        answers = np.concatenate([np.array([(b[0] - denominator) / a[-(len(answers) + 1)]]), answers])
    return answers


# solving Ax = b
# input -> A(matrix with known source points), b(matrix with known destination points)
# output -> x solved from Ax = b
def solve(a, b):
    for i in range(len(b)):
        # if pivot element is zero, while elimination, exchange rows
        if a[i, i] == 0:
            # exchange rows both in A and b
            returned_a, returned_b = exchange(a, b, i)
            if len(returned_a) != 0:
                # exchanged A and b
                a = returned_a
                b = returned_b
        # perform elimination method, i.e., finding pivot elements
        for j in range(i + 1, len(b)):
            var = a[j, i] / a[i, i]
            a[j] = a[j] - a[i] * var
            b[j] = b[j] - b[i] * var

    answers = np.array([])
    # find x values after row reduction
    for i in range(len(b) - 1, -1, -1):
        answers = compute(a[i], b[i], answers)
    # add 9th value of matrix
    answers = np.append(answers, 1.0)
    return answers


# find perspective matrix to perform perspective transformation
# the matrix is 3x3, so it has 9 unknown elements
# since, the 9th element will just scale the z component, which already just scales the x and y component
# so, to find 8 unknown elements, we need 4 source points and their destination
# input -> source points/corners, destination points/corners
# output -> transformation matrix
def get_perspective_matrix(source, destination):
    # no of known points
    size = len(source)
    a = np.zeros((2 * size, 2 * size))
    b = np.zeros((2 * size, 1))
    # fill A with source points, such that x will contain 8 unknowns
    # b will contain known destination points
    # once x is found, Ax will be used to find remaining b.
    for i in range(size):
        a[i, 0] = source[i][0]
        a[i + 4, 3] = source[i][0]
        a[i, 1] = source[i][1]
        a[i + 4, 4] = source[i][1]
        a[i, 2] = 1
        a[i + 4, 5] = 1
        a[i, 3] = 0
        a[i, 4] = 0
        a[i, 5] = 0
        a[i + 4, 0] = 0
        a[i + 4, 1] = 0
        a[i + 4, 2] = 0
        a[i, 6] = -source[i][0] * destination[i][0]
        a[i, 7] = -source[i][1] * destination[i][0]
        a[i + 4, 6] = -source[i][0] * destination[i][1]
        a[i + 4, 7] = -source[i][1] * destination[i][1]
        b[i] = destination[i][0]
        b[i + 4] = destination[i][1]

    # solve Ax = b
    matrix = solve(a, b)
    matrix = np.array(matrix).reshape((3, 3))

    # return perspective transformation matrix
    return matrix


# --------------------------------------------------------------------------------------------------------------------

# ====================================BI-LINEAR INTERPOLATION=========================================================

# source x and source y are probably floating points
# find the floor and ceil of those points
# than do linear interpolation along any one axis
# from 4 points you will get 2 points
# now do linear interpolation on those two points
# input -> source image, x_coordinate, y_coordinate
# output -> pixel value for destination image
def get_bilinear_pixel(src, x, y):
    # to avoid Index Error
    x = min(x, src.shape[1] - 1)
    y = min(y, src.shape[0] - 1)
    # find the coordinates to get 4 points
    floor_x = int(x)
    floor_y = int(y)
    x_diff = x - floor_x
    y_diff = y - floor_y
    x_ceil = min(floor_x + 1, src.shape[1] - 1)
    y_ceil = min(floor_y + 1, src.shape[0] - 1)
    # find the 4 points i.e., 4 pixel values
    bottom_left = src[floor_y, floor_x]
    bottom_right = src[floor_y, x_ceil]
    top_left = src[y_ceil, floor_x]
    top_right = src[y_ceil, x_ceil]

    # Calculate interpolation to find 2 pixel values
    bottom_point = x_diff * bottom_right + (1. - x_diff) * bottom_left
    top_point = x_diff * top_right + (1. - x_diff) * top_left

    # find final interpolated pixel value
    interpolated_value = y_diff * top_point + (1. - y_diff) * bottom_point

    return int(interpolated_value + 0.5)


# apply perspective transformation
# we need to find Ax = b
# where A -> Perspective Transformation Matrix,
# x -> source coordinates
# b -> destination coordinates
# we don't need to transform all input coordinates, but rather we want the pixel values for output coordinates
# so, by doing some manipulation, x = (A^-1)b
# i.ie., for every b we try to find approximate x, i.e., pixel value located at x
# input -> source image, perspective transformation matrix, size of output image
# output -> perspective transformed image
def bilinear_interpolate(image, matrix, image_size=(0, 0)):
    src = np.copy(image)
    # create image(array)
    dst = np.zeros(image_size, np.uint8)
    if image_size == (0, 0):
        dst = np.zeros(image.shape, np.uint8)

    # invert the perspective matrix
    matrix = np.linalg.inv(matrix)

    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst_coords = np.array([j, i, 1]).T
            src_coords = np.dot(matrix, dst_coords)
            # divide x and y component by z component, as it is just scaling
            src_col = src_coords[0] / src_coords[2]
            src_row = src_coords[1] / src_coords[2]
            # find pixel value by doing bilinear interpolation using the obtained source coordinates
            dst[i, j] = get_bilinear_pixel(src, src_col, src_row)
    # return destination image
    return dst

# --------------------------------------------------------------------------------------------------------------------
