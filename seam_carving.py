import numpy as np
import cv2
from scipy import ndimage
import skimage.measure


class SeamCarvingImage:
    """seam carving image"""

    def __init__(self, input_image, energy_function='e1'):
        self.input_image = input_image
        self.input_height = self.input_image.shape[0]
        self.input_width = self.input_image.shape[1]

        if energy_function == 'e1':
            self.energy_func = self.energy_e1
        elif energy_function == 'e2':
            self.energy_func = self.energy_e2
        elif energy_function == 'entropy':
            self.energy_func = self.energy_entropy
        elif energy_function == 'forward':
            self.energy_func = self.energy_forward
        else:
            self.energy_func = None
            print('unsupported energy function: {}'.format(energy_function))

        self.seam_id = 0
        self.result_image = np.array(input_image)
        self.result_pixel_index = np.indices((self.input_image.shape[0], self.input_image.shape[1])).transpose(1, 2, 0)
        self.seam_order = - np.ones((self.input_height, self.input_width), dtype=np.int)
        self.input_image_with_seam = np.array(input_image)
        self.faces = None
        self.protect_mask = None
        self.remove_mask = None
        self.all_seams_expand_result = None

    def reset(self):
        self.result_image = np.array(self.input_image)
        self.input_image_with_seam = np.array(self.input_image)
        self.result_pixel_index = np.indices((self.input_image.shape[0], self.input_image.shape[1])).transpose(1, 2, 0)
        self.seam_id = 0
        self.seam_order = - np.ones((self.input_height, self.input_width), dtype=np.int)
        self.faces = None
        self.protect_mask = None
        self.remove_mask = None

    def energy_e1(self, image):
        grad_x = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
        grad_y = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

        # grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        # grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        grad_x = np.abs(grad_x).sum(axis=2)
        grad_y = np.abs(grad_y).sum(axis=2)
        gradient_map = (np.abs(grad_x) + np.abs(grad_y))

        return gradient_map

    def energy_e2(self, image):
        grad_x = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=1, mode='wrap')
        grad_y = ndimage.convolve1d(image, np.array([1, 0, -1]), axis=0, mode='wrap')

        grad_x = np.abs(grad_x).sum(axis=2)
        grad_y = np.abs(grad_y).sum(axis=2)
        gradient_map = np.sqrt((grad_x * grad_x) + (grad_y * grad_y))

        return gradient_map

    def energy_forward(self, image):

        h, w = image.shape[:2]
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

        gradient_map = np.zeros((h, w))
        m = np.zeros((h, w))

        U = np.roll(image, 1, axis=0)
        L = np.roll(image, 1, axis=1)
        R = np.roll(image, -1, axis=1)

        cU = np.abs(R - L)
        cL = np.abs(U - L) + cU
        cR = np.abs(U - R) + cU

        for i in range(1, h):
            mU = m[i - 1]
            mL = np.roll(mU, 1)
            mR = np.roll(mU, -1)

            mULR = np.array([mU, mL, mR])
            cULR = np.array([cU[i], cL[i], cR[i]])
            mULR += cULR

            argmins = np.argmin(mULR, axis=0)
            m[i] = np.choose(argmins, mULR)
            gradient_map[i] = np.choose(argmins, cULR)

        return gradient_map

    def energy_entropy(self, image):
        gradient_map = self.energy_e1(image)
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

    def make_heatmap(self, image):
        image = image / image.max()
        image = (image * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return heatmap_img

    # def find_min_energy_seam_horizontal(self, image_energy, protect_mask=None, remove_mask=None):
    #     height, width = image_energy.shape
    #     image_energy = np.array(image_energy)  # make a copy
    #     traceback = np.zeros_like(image_energy, dtype=np.int)
    #
    #     if protect_mask is not None:
    #         image_energy[protect_mask] = 1000000
    #
    #     if remove_mask is not None:
    #         image_energy[remove_mask] = -1000000
    #
    #     for x in range(1, width):
    #         for y in range(0, height):
    #             if y == 0:
    #                 idx = np.argmin(image_energy[y:y + 2, x - 1])
    #                 traceback[y, x] = idx + y
    #                 min_energy = image_energy[idx + y, x - 1]
    #             else:
    #                 idx = np.argmin(image_energy[y - 1:y + 2, x - 1])
    #                 traceback[y, x] = idx + y - 1
    #                 min_energy = image_energy[idx + y - 1, x - 1]
    #             image_energy[y, x] += min_energy
    #
    #     # backtrack to find path
    #     seam_idx = []
    #     seam_mask = np.ones((height, width), dtype=np.bool)
    #     i = np.argmin(image_energy[:, -1])  # last column
    #     for j in range(width - 1, -1, -1):
    #         seam_mask[i, j] = False
    #         seam_idx.append(i)
    #         i = traceback[i, j]
    #
    #     seam_idx.reverse()
    #     return np.array(seam_idx), seam_mask

    def find_min_energy_seam_vertical(self, image_energy, protect_mask=None, remove_mask=None):
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

    # def resize(self, output_height, output_width):
    #     width_to_remove = self.input_width - output_width
    #     height_to_remove = self.input_height - output_height
    #
    #     input_image_local = np.array(self.input_image)  # make a copy
    #     input_image_tmp = np.array(self.input_image)  # make a copy
    #     pixel_index_local = np.indices((input_image_local.shape[0], input_image_local.shape[1])).transpose(1, 2, 0)
    #
    #     seam_id = 0
    #     # process width
    #     if width_to_remove < 0:
    #         print('expand width')
    #         width_to_expand = width_to_remove
    #     elif width_to_remove > 0:
    #         print('reduce width')
    #         for i in range(width_to_remove):
    #             energy_map = self.energy_func(input_image_local)
    #             heatmap = self.make_heatmap(energy_map)
    #
    #             seam_index, seam_mask = self.find_min_energy_seam_vertical(energy_map, None, None)
    #
    #             output_image = np.array(input_image_local)
    #             output_image[~seam_mask] = [0, 0, 1.]
    #             heatmap[~seam_mask] = [0, 0, 255.]
    #
    #             seam_pixel_idx = pixel_index_local[~seam_mask]
    #             idx_y = seam_pixel_idx[:, 0]
    #             idx_x = seam_pixel_idx[:, 1]
    #             self.seam_order[idx_y, idx_x] = seam_id
    #             input_image_tmp[idx_y, idx_x] = [0, 0, 1.]
    #
    #             concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=0)
    #             cv2.imshow("seam image + energy map", concat)
    #             cv2.waitKey(1)
    #
    #             # concat_tmp = (input_image_tmp * 255).astype(np.uint8)
    #             # cv2.imshow("seam image + energy map", concat_tmp)
    #             # cv2.waitKey(1)
    #
    #
    #             input_image_local = output_image[seam_mask].reshape(
    #                 (output_image.shape[0], output_image.shape[1] - 1, 3))
    #             pixel_index_local = pixel_index_local[seam_mask].reshape(
    #                 (output_image.shape[0], output_image.shape[1] - 1, 2))
    #             seam_id += 1
    #
    #
    #     # process height
    #     if height_to_remove < 0:
    #         print('expand height')
    #         height_to_expand = - height_to_remove
    #     elif height_to_remove > 0:
    #         print('reduce height')
    #
    #         for i in range(height_to_remove):
    #             energy_map = self.energy_func(input_image_local)
    #             heatmap = self.make_heatmap(energy_map)
    #
    #             seam_index, seam_mask = self.find_min_energy_seam_horizontal(energy_map, None, None)
    #
    #             output_image = np.array(input_image_local)
    #             output_image[~seam_mask] = [0, 0, 1.]
    #             heatmap[~seam_mask] = [0, 0, 255]
    #             concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=1)
    #             cv2.imshow("seam image + energy map", concat)
    #             cv2.waitKey(1)
    #
    #             output_image = np.rot90(output_image, 1, axes=(0, 1))
    #             seam_mask = np.rot90(seam_mask, 1, axes=(0, 1))
    #             input_image_local = output_image[seam_mask].reshape(
    #                 (output_image.shape[0], output_image.shape[1] - 1, 3))
    #             input_image_local = np.rot90(input_image_local, -1, axes=(0, 1))
    #
    #     self.result_image = input_image_local

    def reduce_width(self, width_to_remove):
        print('reduce width')
        input_image_local = np.array(self.result_image)  # make a copy

        protect_mask = None
        remove_mask = None
        if self.protect_mask is not None:
            protect_mask = np.array(self.protect_mask)
        if self.remove_mask is not None:
            remove_mask = np.array(self.remove_mask)

        for i in range(width_to_remove):
            energy_map = self.energy_func(input_image_local)
            heatmap = self.make_heatmap(energy_map)

            seam_index, seam_mask = self.find_min_energy_seam_vertical(energy_map, protect_mask, remove_mask)

            output_image = np.array(input_image_local)
            output_image[~seam_mask] = [0, 0, 1.]
            heatmap[~seam_mask] = [0, 0, 255.]

            seam_pixel_idx = self.result_pixel_index[~seam_mask]
            idx_y = seam_pixel_idx[:, 0]
            idx_x = seam_pixel_idx[:, 1]
            self.seam_order[idx_y, idx_x] = self.seam_id
            self.input_image_with_seam[idx_y, idx_x] = [0, 0, 1.]

            concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=0)
            cv2.imshow("seam image + energy map", concat)
            show_2 = (self.input_image_with_seam * 255).astype(np.uint8)
            cv2.imshow("original image with seam", show_2)
            cv2.waitKey(1)

            input_image_local = output_image[seam_mask].reshape(
                (output_image.shape[0], output_image.shape[1] - 1, 3))
            self.result_pixel_index = self.result_pixel_index[seam_mask].reshape(
                (output_image.shape[0], output_image.shape[1] - 1, 2))

            if protect_mask is not None:
                protect_mask = protect_mask[seam_mask].reshape(
                    (output_image.shape[0], output_image.shape[1] - 1))
            if remove_mask is not None:
                remove_mask = remove_mask[seam_mask].reshape(
                    (output_image.shape[0], output_image.shape[1] - 1))

            self.seam_id += 1

        self.result_image = input_image_local

    def reduce_height(self, height_to_remove):
        print('reduce height')
        input_image_local = np.array(self.result_image)  # make a copy
        input_image_local = np.rot90(input_image_local, -1, axes=(0, 1))

        result_pixel_index_rot = np.rot90(self.result_pixel_index, -1, axes=(0, 1))
        protect_mask = None
        remove_mask = None
        if self.protect_mask is not None:
            protect_mask = np.array(self.protect_mask)
            protect_mask = np.rot90(protect_mask, -1, axes=(0, 1))
        if self.remove_mask is not None:
            remove_mask = np.array(self.protect_mask)
            remove_mask = np.rot90(remove_mask, -1, axes=(0, 1))

        for i in range(height_to_remove):
            energy_map = self.energy_func(input_image_local)
            heatmap = self.make_heatmap(energy_map)

            seam_index_rot, seam_mask_rot = self.find_min_energy_seam_vertical(energy_map, protect_mask, remove_mask)

            output_image = np.array(input_image_local)
            output_image[~seam_mask_rot] = [0, 0, 1.]
            heatmap[~seam_mask_rot] = [0, 0, 255.]

            seam_mask = np.rot90(seam_mask_rot, 1, axes=(0, 1))
            seam_pixel_idx = self.result_pixel_index[~seam_mask]
            idx_y = seam_pixel_idx[:, 0]
            idx_x = seam_pixel_idx[:, 1]
            self.seam_order[idx_y, idx_x] = self.seam_id
            self.input_image_with_seam[idx_y, idx_x] = [0, 0, 1.]

            concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=0)
            concat = np.rot90(concat, 1, axes=(0, 1))
            cv2.imshow("seam image + energy map", concat)
            show_2 = (self.input_image_with_seam * 255).astype(np.uint8)
            cv2.imshow("original image with seam", show_2)
            cv2.waitKey(1)

            input_image_local = output_image[seam_mask_rot].reshape(
                (output_image.shape[0], output_image.shape[1] - 1, 3))
            result_pixel_index_rot = result_pixel_index_rot[seam_mask_rot].reshape(
                (output_image.shape[0], output_image.shape[1] - 1, 2))
            if protect_mask is not None:
                protect_mask = protect_mask[seam_mask_rot].reshape(
                    (output_image.shape[0], output_image.shape[1] - 1))
            if remove_mask is not None:
                remove_mask = remove_mask[seam_mask_rot].reshape(
                    (output_image.shape[0], output_image.shape[1] - 1))
            self.result_pixel_index = np.rot90(result_pixel_index_rot, 1, axes=(0, 1))
            self.seam_id += 1

        input_image_local = np.rot90(input_image_local, 1, axes=(0, 1))
        self.result_image = input_image_local

    def expand_width(self, width_to_expand, min_seam_count):
        print('expand width')
        all_seams_stage = np.array(self.result_image)

        while width_to_expand > 0:
            find_seams_count = min(min_seam_count, width_to_expand)

            input_image_local = np.array(self.result_image)  # make a copy
            k_seams_image = np.array(self.result_image)
            seams = []
            result_pixel_index_expand = np.indices((input_image_local.shape[0], input_image_local.shape[1])).transpose(
                1, 2, 0)
            for i in range(find_seams_count):
                energy_map = self.energy_func(input_image_local)
                # heatmap = self.make_heatmap(energy_map)

                seam_index, seam_mask = self.find_min_energy_seam_vertical(energy_map, None, None)

                output_image = np.array(input_image_local)
                output_image[~seam_mask] = [0, 0, 1.]
                # heatmap[~seam_mask] = [0, 0, 255.]

                seam_pixel_idx = result_pixel_index_expand[~seam_mask]

                idx_y = seam_pixel_idx[:, 0]
                idx_x = seam_pixel_idx[:, 1]
                # self.seam_order[idx_y, idx_x] = self.seam_id
                k_seams_image[idx_y, idx_x] = [0, 0, 1.]

                # concat = np.concatenate([(output_image * 255).astype(np.uint8), heatmap], axis=0)
                # cv2.imshow("seam image + energy map", concat)
                show_2 = (k_seams_image * 255).astype(np.uint8)
                cv2.imshow("stage image with seam", show_2)
                cv2.waitKey(1)

                input_image_local = output_image[seam_mask].reshape(
                    (output_image.shape[0], output_image.shape[1] - 1, 3))
                result_pixel_index_expand = result_pixel_index_expand[seam_mask].reshape(
                    (output_image.shape[0], output_image.shape[1] - 1, 2))
                seams.append(seam_pixel_idx)

            # the first k seams with lowest energy found
            # print(seams)
            img_expand_stage = np.array(self.result_image)

            seams_shifted = np.array(seams)  # [seam_count, height, 2]
            for seam_id in range(seams_shifted.shape[0]):
                insert_x = seams_shifted[seam_id, :, 1]
                should_shift = seams_shifted[seam_id + 1:, :, 1] >= insert_x
                seams_shifted[seam_id + 1:, :, 1][should_shift] += 1
            for idx, s in enumerate(seams):
                h = img_expand_stage.shape[0]
                w = img_expand_stage.shape[1]
                old_remain_mask = np.ones((h, w), dtype=bool)

                img_expand = np.zeros((h, w + 1, 3), dtype=img_expand_stage.dtype)
                img_expand_remain = np.ones((h, w + 1), dtype=bool)
                all_seams = np.zeros((h, w + 1, 3), dtype=img_expand_stage.dtype)

                seam_left_idx = seams_shifted[idx]
                seam_right_idx = seam_left_idx + np.array([0, 1])

                idx_y_left = seam_left_idx[:, 0]
                idx_x_left = seam_left_idx[:, 1]
                idx_y_right = seam_right_idx[:, 0]
                idx_x_right = seam_right_idx[:, 1]

                idx_y_old = s[:, 0]
                idx_x_old = s[:, 1]
                old_seam_pixels = self.result_image[idx_y_old, idx_x_old]

                idx_y_old_left = idx_y_old
                idx_x_old_left = idx_x_old - 1
                idx_x_old_left[idx_x_old_left < 0] = 0
                old_seam_left_pixels = self.result_image[idx_y_old_left, idx_x_old_left]

                idx_y_old_right = idx_y_old
                idx_x_old_right = idx_x_old + 1
                idx_x_old_right[idx_x_old_right >= self.result_image.shape[1]] = self.result_image.shape[1] - 1
                old_seam_right_pixels = self.result_image[idx_y_old_right, idx_x_old_right]

                old_remain_mask[idx_y_old, idx_x_old] = False

                # assign seam left
                img_expand[idx_y_left, idx_x_left] = (old_seam_pixels + old_seam_left_pixels) / 2
                img_expand[idx_y_right, idx_x_right] = (old_seam_pixels + old_seam_right_pixels) / 2

                img_expand_remain[idx_y_left, idx_x_left] = False
                img_expand_remain[idx_y_right, idx_x_right] = False

                img_expand[img_expand_remain] = img_expand_stage[old_remain_mask]
                img_expand_stage = img_expand

                all_seams[idx_y_left, idx_x_left] = [0, 0, 1.]
                all_seams[idx_y_right, idx_x_right] = [0, 0, 1.]
                all_seams[img_expand_remain] = all_seams_stage[old_remain_mask]
                all_seams_stage = all_seams

                cv2.imshow("insert 1", (img_expand * 255).astype(np.uint8))
                # cv2.waitKey(1)
                cv2.imshow("all seams", (all_seams * 255).astype(np.uint8))
                cv2.waitKey(1)
            self.result_image = img_expand_stage
            # cv2.imshow("result with seams", (self.result_image_wi * 255).astype(np.uint8))
            # cv2.imshow("expanded result", (self.result_image * 255).astype(np.uint8))
            # cv2.waitKey(0)
            width_to_expand -= find_seams_count

        self.all_seams_expand_result = all_seams_stage

    def expand_height(self, height_to_expand, min_seam_count):
        print('expand height')

    def show(self):
        cv2.imshow("result image", self.result_image)
        cv2.waitKey(0)

    def save_result(self, name):
        cv2.imwrite(name, (self.result_image * 255).astype(np.uint8))

    def save_result_with_expand_seams(self, name):
        cv2.imwrite(name, (self.all_seams_expand_result * 255).astype(np.uint8))

    def save_input_image_with_seam(self, name):
        cv2.imwrite(name, (self.input_image_with_seam * 255).astype(np.uint8))

    def save_energy_map(self, name):
        energy_map = self.energy_func(self.result_image)
        heatmap = self.make_heatmap(energy_map)
        cv2.imwrite(name, heatmap)

    def preprocess(self):
        print('Preprocess')

    def amplify(self, factor=1.2):
        print('amplify')
        assert factor > 1.0
        up_scale_height = int(self.result_image.shape[0] * factor)
        up_scale_width = int(self.result_image.shape[1] * factor)
        remove_h = up_scale_height - self.result_image.shape[0]
        remove_w = up_scale_width - self.result_image.shape[1]

        self.result_image = cv2.resize(self.input_image, (up_scale_width, up_scale_height))
        self.seam_id = 0
        self.result_pixel_index = np.indices((self.result_image.shape[0], self.result_image.shape[1])).transpose(1, 2,
                                                                                                                 0)
        self.seam_order = - np.ones((self.result_image.shape[0], self.result_image.shape[1]), dtype=np.int)
        self.input_image_with_seam = np.array(self.result_image)

        self.reduce_width(remove_w)
        self.reduce_height(remove_h)

    def find_faces(self):
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor((self.result_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        self.faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        print('face count:', len(self.faces))

    def make_protect_face_mask(self):
        if self.protect_mask is None:
            self.protect_mask = np.zeros((self.result_image.shape[0], self.result_image.shape[1]), dtype=bool)
        for (x, y, w, h) in self.faces:
            self.protect_mask[y:y + h, x:x + w] = True

    def show_faces(self):
        im = (np.array(self.result_image) * 255).astype(np.uint8)
        for (x, y, w, h) in self.faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("Faces", im)
        cv2.waitKey(0)

    def save_faces_image(self, name):
        im = (np.array(self.result_image) * 255).astype(np.uint8)
        for (x, y, w, h) in self.faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(name, im)


def demo1():
    """content amplification"""
    input_file = 'data/bench.png'

    input_image = cv2.imread(input_file).astype(np.float64)
    input_image = input_image / 255.0
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]

    sc_image = SeamCarvingImage(input_image, energy_function='e1')

    sc_image.amplify(1.2)
    sc_image.show()
    sc_image.save_result('output/bench_amplification.png')

def demo1_2():
    """content amplification"""
    input_file = 'data/balloons.jpg'
    input_image = cv2.imread(input_file).astype(np.float64)
    input_image = input_image / 255.0
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]

    sc_image = SeamCarvingImage(input_image, energy_function='forward')

    sc_image.amplify(1.5)
    sc_image.show()
    sc_image.save_result('output/ballons_amplification.png')

def demo2():
    """remove the woman"""
    input_file = 'data/couple.png'
    input_image = cv2.imread(input_file).astype(np.float64)
    input_image = input_image / 255.0
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]

    pro_mask = cv2.imread('data/protect.mask.png').sum(axis=2).astype(np.bool)
    rm_mask = cv2.imread('data/remove.mask.png').sum(axis=2).astype(np.bool)

    sc_image = SeamCarvingImage(input_image, energy_function='forward')
    sc_image.protect_mask = pro_mask
    sc_image.remove_mask = rm_mask
    sc_image.reduce_width(80)
    sc_image.show()
    sc_image.save_result('output/remove_woman.png')


def demo3():
    """remove the man"""
    input_file = 'data/couple.png'
    input_image = cv2.imread(input_file).astype(np.float64)
    input_image = input_image / 255.0
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]

    pro_mask = cv2.imread('data/remove.mask.png').sum(axis=2).astype(np.bool)
    rm_mask = cv2.imread('data/protect.mask.png').sum(axis=2).astype(np.bool)

    sc_image = SeamCarvingImage(input_image, energy_function='forward')
    sc_image.protect_mask = pro_mask
    sc_image.remove_mask = rm_mask
    sc_image.reduce_width(80)
    sc_image.show()
    sc_image.save_result('output/remove_man.png')

def demo4():
    """forward energy"""
    input_file = 'data/bench.png'
    input_image = cv2.imread(input_file).astype(np.float64)
    input_image = input_image / 255.0
    if input_image.shape[2] > 3:
        input_image = input_image[:, :, :3]

    sc_image = SeamCarvingImage(input_image, energy_function='forward')
    sc_image.reduce_width(150)
    sc_image.show()
    sc_image.save_result('output/forward_bench_result.png')
    sc_image.save_input_image_with_seam('output/forward_bench_result_seams.png')

    sc_image = SeamCarvingImage(input_image, energy_function='e1')
    sc_image.reduce_width(150)
    sc_image.show()
    sc_image.save_result('output/e1_bench_result.png')
    sc_image.save_input_image_with_seam('output/e1_bench_result_seams.png')

if __name__ == '__main__':
    # demo1()
    # demo1_2()
    # demo2()
    # demo3()
    demo4()
    # input_file = 'data/bench.png'
    # input_file = 'data/art3.png'
    # input_file = 'data/art2.jpg'
    # input_file = 'data/art1.png'
    # input_file = 'data/trump.jpg'
    # input_file = 'data/trump_and_baby.jpg'
    # input_file = 'data/king.jpg'
    # input_file = 'data/ma2.jpg'
    # input_file = 'data/balloons.jpg'
    # input_file = 'data/plane.jpg'
    # input_file = 'data/benz.jpg'
    # input_image = cv2.imread(input_file).astype(np.float64)
    # input_image = input_image / 255.0
    # if input_image.shape[2] > 3:
    #     input_image = input_image[:, :, :3]
    #
    # pro_mask =  cv2.imread('data/protect.mask.png').sum(axis=2).astype(np.bool)
    # rm_mask =  cv2.imread('data/remove.mask.png').sum(axis=2).astype(np.bool)
    #
    # sc_image = SeamCarvingImage(input_image, energy_function='forward')
    # sc_image.expand_width(100, 40)
    # sc_image.show()
    # sc_image.save_result('art1_result_40.png')
    # sc_image.save_result_with_expand_seams('art1_result_seams_40.png')
    # sc_image.find_faces()
    # sc_image.make_protect_face_mask()
    # sc_image.save_faces_image('ma2_faces_detected.png')
    #
    # sc_image.save_energy_map('ma2_energy_face.png')
    # sc_image.reduce_width(200)
    # sc_image.show()
    # sc_image.save_result('ma2_result_face.png')
    # sc_image.save_input_image_with_seam('ma2_result_seam_face.png')

    # sc_image.find_faces()
    # sc_image.show_faces()

    # sc_image = SeamCarvingImage(input_image, energy_function='forward')
    # sc_image.save_energy_map('output/bench_forward.png')
    # sc_image = SeamCarvingImage(input_image, energy_function='e1')
    # sc_image.save_energy_map('output/bench_e1.png')
    # sc_image = SeamCarvingImage(input_image, energy_function='e2')
    # sc_image.save_energy_map('output/bench_e2.png')
    # sc_image = SeamCarvingImage(input_image, energy_function='entropy')
    # sc_image.save_energy_map('output/bench_entropy.png')
    # sc_image = SeamCarvingImage(np.rot90(input_image,1,axes=(0, 1)), energy_function='forward')
    # sc_image.save_energy_map('output/bench_forward_rot.png')
    # sc_image.reduce_height(10)
    # sc_image.reduce_width(180)
    # sc_image.reduce_height(10)


