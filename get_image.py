import cv2 as cv
from pathlib import Path

def get_image():
    class_name = 'iloveu'
    save_dir = Path('DATASET') / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_count = 0
    max_frames = 500

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_count += 1

            if frame_count % 5 == 0:
                image_filename = save_dir / f'{frame_count}.png'
                cv.imwrite(str(image_filename), frame)

            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
   get_image()
