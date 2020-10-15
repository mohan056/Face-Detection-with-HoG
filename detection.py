import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse


def get_differential_filter():
    # To do

    # defining Sobel filters
    filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do

    # padding image by replicating its edges to avoid boundary effect
    zero_padded_im = np.zeros((2+im.shape[0], 2+im.shape[1]))
    zero_padded_im[1:-1, 1:-1] = im
    zero_padded_im[1:-1, 0] = im[:, 0]
    zero_padded_im[1:-1, -1] = im[:, -1]
    zero_padded_im[0, 1:-1] = im[0, :]
    zero_padded_im[-1, 1:-1] = im[-1, :]

    im_filtered = np.zeros(zero_padded_im.shape)

    # code to apply kernel-filter to each tile in the image
    center_k = filter.shape[0]//2
    center_l = filter.shape[1]//2
    for i in range(1, zero_padded_im.shape[0]-1):
        for j in range(1, zero_padded_im.shape[1]-1):
            im_tile = zero_padded_im[i-center_l:i+center_l+1, j-center_k:j+center_k+1]
            im_filtered[i][j] = np.sum(im_tile * filter)
    return im_filtered[1:-1, 1:-1]


def get_gradient(im_dx, im_dy):
    # To do

    # grad magnitude is calculated
    grad_mag = np.sqrt(im_dx * im_dx + im_dy * im_dy)

    # grad angle is calculated
    grad_angle = np.arctan2(im_dy, im_dx)

    # grad angle is fitted into the range of [0, pi] in case it is negative.
    grad_angle[grad_angle < 0] += np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    ori_histo = np.zeros((grad_mag.shape[0]//cell_size, grad_mag.shape[1]//cell_size, 6))

    # data structure to hold all the angle cut-offs
    angle_cutoffs = [
        tuple([165, 180, 0, 15]),
        tuple([15, 45]),
        tuple([45, 75]),
        tuple([75, 105]),
        tuple([105, 135]),
        tuple([135, 165])
    ]

    # code to assign each cell its histogram based on pixel magnitudes that belong to each bin
    for i in range(ori_histo.shape[0] * cell_size):
        for j in range(ori_histo.shape[1] * cell_size):
            for k in range(6):
                if k == 0:
                    if angle_cutoffs[k][0] <= math.degrees(grad_angle[i][j]) < angle_cutoffs[k][1] or angle_cutoffs[k][2] <= math.degrees(grad_angle[i][j]) < angle_cutoffs[k][3]:
                        ori_histo[i//cell_size][j//cell_size][k] += grad_mag[i][j]
                else:
                    if angle_cutoffs[k][0] <= math.degrees(grad_angle[i][j]) < angle_cutoffs[k][1]:
                        ori_histo[i//cell_size][j//cell_size][k] += grad_mag[i][j]

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    e = 0.001

    ori_histo_normalized = np.zeros((ori_histo.shape[0] + 1 - block_size,
                                     ori_histo.shape[1] + 1 - block_size,
                                     ori_histo.shape[2] * block_size * block_size))

    # constructing block-descriptor for each block
    for i in range(ori_histo.shape[0] + 1 - block_size):
        for j in range(ori_histo.shape[1] + 1 - block_size):
            histo_elements = ori_histo[i:i+block_size, j:j+block_size, :].flatten()
            ori_histo_normalized[i, j, :] = histo_elements

    # normalizing each block descriptor on the basis of values obtained
    ij_sum = np.sqrt(np.dstack([np.sum(np.square(ori_histo_normalized), axis=2) + e ** 2] * block_size * block_size * 6))
    ori_histo_normalized = ori_histo_normalized / ij_sum

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    im = (im - im.min())/(im.max() - im.min())

    # To do

    # getting the dI/dx and dI/dy images
    filter_x, filter_y = get_differential_filter()
    x_filtered_image = filter_image(im, filter_x)
    y_filtered_image = filter_image(im, filter_y)

    # obtaining the gradient magnitude and gradient angle values for each pixel
    grad_m, grad_a = get_gradient(x_filtered_image, y_filtered_image)

    # if the user wants, they can display all the images in order to see if the results are satisfactory or not
    # if display_images:
    #
    #     fig, ax = plt.subplots(1, 5)
    #     ax[0].imshow(im, cmap='gray', vmin=0, vmax=1)
    #     ax[1].imshow(x_filtered_image, cmap='gray', vmin=0, vmax=1)
    #     ax[2].imshow(y_filtered_image, cmap='gray', vmin=0, vmax=1)
    #     ax[3].imshow(grad_m, cmap='gray', vmin=0, vmax=1)
    #     ax[4].imshow(grad_a, cmap='gray', vmin=0, vmax=1)
    #     plt.show()

    # visualize to verify

    ori_histo = build_histogram(grad_m, grad_a, 8)
    hog = get_block_descriptor(ori_histo, 2)

    # if the user wants, they can visualize the HOG to be able to debug the code
    # if debug:
    #     visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):

    # assigning threshold values and IoU cutoff values
    threshold = 0.35
    IoU_cutoff = 0.50

    bounding_boxes = np.empty((0, 3))

    # extracting HOG for the template and mean-normalizing it
    template_hogs = extract_hog(I_template)
    template_hogs = template_hogs - np.mean(template_hogs)

    for i in range(0, I_target.shape[0] - I_template.shape[0] + 1, 5):
        for j in range(0, I_target.shape[1] - I_template.shape[1] + 1, 5):

            # exctracting HOG features for a tile that is shaped according to the template image size
            target_tile = I_target[i:i+I_template.shape[0], j:j+I_template.shape[1]]
            target_tile_hog = extract_hog(target_tile)
            target_tile_hog = target_tile_hog - np.mean(target_tile_hog)

            # calculating NCC score between the template HOG features and target image tile HOG features
            score = np.sum(target_tile_hog * template_hogs) / np.sqrt(np.sum(target_tile_hog**2) * np.sum(template_hogs**2))

            # thresholding NCC Score
            if score > threshold:
                bounding_boxes = np.append(bounding_boxes, np.array([[j, i, score]]), axis=0)

    bounding_boxes = bounding_boxes[np.argsort(bounding_boxes[:, 2])][::-1]

    new_bounding_boxes = np.empty((0, 3))
    bounding_box_area = I_template.shape[0] * I_template.shape[1]

    while bounding_boxes.size > 0:

        # getting highest NCC scored bounding box and putting it into the filtered list
        max_box = bounding_boxes[0]
        new_bounding_boxes = np.append(new_bounding_boxes, np.array([max_box]), axis=0)
        bounding_boxes = bounding_boxes[1:]

        # vectorizing the process of getting the area between two bounding boxes and successively calculating their IoU
        x_start = np.maximum(max_box[0], bounding_boxes[:, 0])
        y_start = np.maximum(max_box[1], bounding_boxes[:, 1])
        x_end = np.minimum(max_box[0]+I_template.shape[0], bounding_boxes[:, 0]+I_template.shape[0])
        y_end = np.minimum(max_box[1]+I_template.shape[1], bounding_boxes[:, 1]+I_template.shape[1])

        interesction = np.maximum(0, x_end - (x_start - 1)) * np.maximum(0, y_end - (y_start - 1))
        IoU = interesction / ((2 * bounding_box_area) - interesction)

        # reducing existing bounding boxes so that all bounding boxes with higher IoU are eliminated
        bounding_boxes = bounding_boxes[IoU < IoU_cutoff]

    return new_bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh, ww, cc = I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1 < 0:
            x1 = 0
        if x1 > ww-1:
            x1 = ww-1
        if x2 < 0:
            x2 = 0
        if x2 > ww-1:
            x2 = ww-1
        if y1 < 0:
            y1 = 0
        if y1 > hh-1:
            y1 = hh-1
        if y2 < 0:
            y2 = 0
        if y2 > hh-1:
            y2 = hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f" % bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.imwrite('bounding_boxes.png', fimg)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-mode', type=int,
                        default=1, help='set 1 to run our test demo. set 0 to run on your own template and test image')
    parser.add_argument('--template-img-path', type=str, help='set to whatever location the template is in.')
    parser.add_argument('--target-img-path', type=str, help='set to whatever location the target is in.')

    args = parser.parse_args()
    img_mode = args.img_mode

    if img_mode == 1:
        im = cv2.imread('data/target.png', 0)
        hog = extract_hog(im)
        I_target = cv2.imread('data/target.png', 0)
        #MxN image
        I_template = cv2.imread('data/template.png', 0)
        #mxn  face template
        bounding_boxes = face_recognition(I_target, I_template)
        I_target_c = cv2.imread('data/target.png')
        # MxN image (just for visualization)
        visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])

    else:
        target = cv2.imread(args.target_img_path, 0)
        template = cv2.imread(args.template_img_path, 0)

        target_c = cv2.imread(args.target_img_path, 0)
        bounding_boxes = face_recognition(target, template)

        visualize_face_detection(target_c, bounding_boxes, template.shape[0])
