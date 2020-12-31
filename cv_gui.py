import numpy as np
import cv2

drawing = False
mode = 0
ix, iy = -1, -1

color_protect = (0, 255, 0)
color_remove = (0, 0, 255)
color_empty = (0, 0, 0)

protect_mode = 0
remove_mode = 1


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == protect_mode:
                cv2.circle(canvas, (x, y), 10, color_protect, -1)
                cv2.circle(protect_mask, (x, y), 10, 255, -1)
                cv2.circle(remove_mask, (x, y), 10, 0, -1)

            else:
                cv2.circle(canvas, (x, y), 10, color_remove, -1)
                cv2.circle(remove_mask, (x, y), 10, 255, -1)
                cv2.circle(protect_mask, (x, y), 10, 0, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == protect_mode:
            cv2.circle(canvas, (x, y), 10, color_protect, -1)
        else:
            cv2.circle(canvas, (x, y), 10, color_remove, -1)


cv2.namedWindow('Mask Creator')
cv2.setMouseCallback('Mask Creator', draw_circle)


def save():
    cv2.imwrite('data/protect.mask.png', protect_mask)
    cv2.imwrite('data/remove.mask.png', remove_mask)
    print('mask saved')

if __name__ == '__main__':
    input_file = 'data/couple.png'
    input_image = cv2.imread(input_file)
    # h, w, _ = input_image.shape
    canvas = np.array(input_image)
    protect_mask = np.zeros(input_image.shape[:2], dtype=canvas.dtype)
    remove_mask = np.zeros(input_image.shape[:2], dtype=canvas.dtype)

    while (1):
        cv2.imshow('Mask Creator', canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            mode = remove_mode
        elif k == ord('p'):
            mode = protect_mode
        elif k == ord('s'):
            save()
        elif k == 27:
            break

    cv2.destroyAllWindows()
