import glob
import cv2 as cv


def getfiles(path):
    print('searching', path)
    files = sorted(glob.glob(path+'/*.png'))

    if not files:
        raise FileNotFoundError(f'no files found under {path}')

    return files


def visualize_image_on_colab(image_path):
    from google.colab.patches import cv2_imshow

    files = getfiles(image_path)

    if not files:
        raise FileNotFoundError(f'No files found under {image_path}')
    
    for fn in files:
        image = cv.imread(fn)
        cv2_imshow(image) # Note cv2_imshow, not cv2.imshow


def visualize_video_on_colab(video_path):
    from google.colab.patches import cv2_imshow

    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        cv2_imshow(image) # Note cv2_imshow, not cv2.imshow
        
        cv.waitKey(1) & 0xff

    cv.destroyAllWindows()
    cap.release()
