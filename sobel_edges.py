# If you want to apply gauss to an image and then apply sobel filter
# Then, 1st apply sobel to gauss filter and then apply the result on image
# Result will be same, but it will require less computation.
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

    # calculate the values of final filter
    # after applying sobel to gauss
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

    # filter which contains sobel applied to gauss, which can be applied to any image
    # the result after applying this filter to any image will result in edges detected using sobel after applying gauss
    return gauss_sobel_x, gauss_sobel_y


# Apply cross-correlation of input kernel on input image.
# apply sobel filter(any of the edge detector filter) to image
# finding edges using sobel edge detector kernel on which gaussian filter is applied
# you can skip applying gaussian filter by directly passing the sobel kernel
# Input -> image, edge detector kernel
# output -> image with edges(horizontal or vertical based on input kernel)

def sobel_edges(image, kernel):
    # convolve sobel_edge_detector to image
    # same as applying differentiation or finding gradients
    # i.e., partial differentiation
    # w.r.t to x when using vertical edge detector kernel
    # wr.t to y when using horizontal edge detector kernel

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
    new_image = my_cv.contrast_stretching(img)
    # find gauss_kernel for smoothening the iamge
    gauss_kernel = my_cv.get_gauss_kernel(sigma)
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


def param_passing():
    image_path = 'Samples/s2.jpg'  # <- insert image name here

    # the quality of edges depends on these parameters
    # try different parameters and see the changes in edges
    sigma = 0.5  # <- insert sigma value
    threshold = 0.01

    image = cv2.imread(image_path)  # -> open the image

    image_path = image_path.replace("\\", "/")
    image_name = image_path.split("/")[-1]

    # run the Sobel edge detector
    sobel_edge(image, image_name, sigma, threshold)


if __name__ == "__main__":
    param_passing()
    cv2.waitKey(0)
