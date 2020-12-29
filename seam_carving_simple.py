import numpy as np
import cv2
from scipy import ndimage
import skimage.measure


def energy_e1(image):
    grad_x = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
    grad_y = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

    # grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    # grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    grad_x = np.abs(grad_x).sum(axis=2)
    grad_y = np.abs(grad_y).sum(axis=2)
    gradient_map = (np.abs(grad_x) + np.abs(grad_y))

    return gradient_map


def energy_e2(image):
    grad_x = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
    grad_y = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

    grad_x = np.abs(grad_x).sum(axis=2)
    grad_y = np.abs(grad_y).sum(axis=2)
    gradient_map = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

    return gradient_map


def energy_entropy(image):
    gradient_map = energy_e1(image)
    height, width, _ = image.shape
    for j in range(height):
        for i in range(width):
            # crop a 9 x 9 window around the pixel (j, i)
            y_start = max(j - 4, 0)
            y_end = min(j + 4 + 1, height)
            x_start = max(i - 4, 0)
            x_end = min(i + 4 + 1, width)
            sub_image = image[y_start:y_end, x_start:x_end]

            entropy = skimage.measure.shannon_entropy(sub_image)
            gradient_map[j, i] += entropy
    return gradient_map


def make_heatmap(image):
    image = image / image.max()
    image = (image * 255).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return heatmap_img


def find_min_energy_seam_horizontal(image_energy, protect_mask=None, remove_mask=None):
    height, width = image_energy.shape
    image_energy = np.array(image_energy)  # make a copy
    traceback = np.zeros_like(image_energy, dtype=np.int)

    if protect_mask is not None:
        image_energy[protect_mask] = 1000000

    if remove_mask is not None:
        image_energy[remove_mask] = -1000000

    for x in range(1, width):
        for y in range(0, height):
            if y == 0:
                idx = np.argmin(image_energy[y:y + 2, x - 1])
                traceback[y, x] = idx + y
                min_energy = image_energy[idx + y, x - 1]
            else:
                idx = np.argmin(image_energy[y - 1:y + 2, x - 1])
                traceback[y, x] = idx + y - 1
                min_energy = image_energy[idx + y - 1, x - 1]
            image_energy[y, x] += min_energy

    # backtrack to find path
    seam_idx = []
    seam_mask = np.ones((height, width), dtype=np.bool)
    i = np.argmin(image_energy[:, -1])  # last column
    for j in range(width - 1, -1, -1):
        seam_mask[i, j] = False
        seam_idx.append(i)
        i = traceback[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), seam_mask


def find_min_energy_seam_vertical(image_energy, protect_mask=None, remove_mask=None):
    height, width = image_energy.shape
    image_energy = np.array(image_energy)  # make a copy
    traceback = np.zeros_like(image_energy, dtype=np.int)

    if protect_mask is not None:
        image_energy[protect_mask] = 1000000

    if remove_mask is not None:
        image_energy[remove_mask] = -1000000

    for y in range(1, height):
        for x in range(0, width):
            if x == 0:
                idx = np.argmin(image_energy[y - 1, x:x + 2])
                traceback[y, x] = idx + x
                min_energy = image_energy[y - 1, idx + x]
            else:
                idx = np.argmin(image_energy[y - 1, x - 1:x + 2])
                traceback[y, x] = idx + x - 1
                min_energy = image_energy[y - 1, idx + x - 1]

            image_energy[y, x] += min_energy

    # backtrack to find path
    seam_idx = []
    seam_mask = np.ones((height, width), dtype=np.bool)
    j = np.argmin(image_energy[-1])  # last row
    for i in range(height - 1, -1, -1):
        seam_mask[i, j] = False
        seam_idx.append(j)
        j = traceback[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), seam_mask


# set energy function here
energy_func = energy_e1
reduce_width = True
if __name__ == '__main__':
    # input_file = 'data/castle.jpg'
    # input_file = 'data/train.jpg'
    # input_file = 'data/shore.jpg'
    input_file = 'data/bench.png'
    # input_file = 'data/plane.jpg'
    # input_file = 'data/benz.jpg'
    input_image = cv2.imread(input_file).astype(np.float64)
    input_image = input_image / 255.0
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]

    if reduce_width:
        width_to_remove = 100
        for i in range(width_to_remove):
            energy_map = energy_func(input_image)
            heatmap = make_heatmap(energy_map)

            seam_index, seam_mask = find_min_energy_seam_vertical(energy_map, None, None)

            output_image = np.array(input_image)
            output_image[~seam_mask] = [0, 0, 1.]
            heatmap[~seam_mask] = [0, 0, 255.]
            concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=0)
            cv2.imshow("seam image + energy map", concat)
            cv2.waitKey(1)

            input_image = output_image[seam_mask].reshape((output_image.shape[0], output_image.shape[1] - 1, 3))
    else:
        height_to_remove = 150
        for i in range(height_to_remove):
            energy_map = energy_func(input_image)
            heatmap = make_heatmap(energy_map)

            seam_index, seam_mask = find_min_energy_seam_horizontal(energy_map, None, None)

            output_image = np.array(input_image)
            output_image[~seam_mask] = [0, 0, 1.]
            heatmap[~seam_mask] = [0, 0, 255]
            concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=1)
            cv2.imshow("seam image + energy map", concat)
            cv2.waitKey(1)

            output_image = np.rot90(output_image, 1, axes=(0, 1))
            seam_mask = np.rot90(seam_mask, 1, axes=(0, 1))
            input_image = output_image[seam_mask].reshape((output_image.shape[0], output_image.shape[1] - 1, 3))
            input_image = np.rot90(input_image, -1, axes=(0, 1))

    cv2.imwrite("train_{}.png".format('w' if reduce_width else 'h'), (input_image * 255).astype(np.uint8))
    cv2.imshow("final image", input_image)
    cv2.waitKey(0)
