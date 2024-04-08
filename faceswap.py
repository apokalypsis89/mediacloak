import cv2
import dlib
import numpy as np

def face_swap(source_img, target_img):
    # Load face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Load source and target images
    source = cv2.imread(source_img)
    target = cv2.imread(target_img)

    # Convert images to grayscale
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Detect faces in the images
    source_faces = detector(source_gray)
    target_faces = detector(target_gray)

    if len(source_faces) == 0 or len(target_faces) == 0:
        print("No faces detected in one or both images.")
        return None

    # Assume only one face in each image for simplicity
    source_face = source_faces[0]
    target_face = target_faces[0]

    # Detect facial landmarks for source and target faces
    source_landmarks = predictor(source_gray, source_face)
    target_landmarks = predictor(target_gray, target_face)

    # Convert landmarks to numpy arrays
    source_points = np.array([[p.x, p.y] for p in source_landmarks.parts()])
    target_points = np.array([[p.x, p.y] for p in target_landmarks.parts()])

    # Estimate affine transformation between source and target landmarks
    transform_matrix = cv2.estimateAffinePartial2D(source_points, target_points)[0]

    # Warp source face onto target face using the transformation
    warped_source = cv2.warpAffine(source, transform_matrix, (target.shape[1], target.shape[0]))

    # Create a mask for the warped source face
    mask = np.zeros_like(target)
    mask[target_face.top():target_face.bottom(), target_face.left():target_face.right()] = 255

    # Use Poisson blending to blend the warped source face into the target image
    result = cv2.seamlessClone(warped_source, target, mask, (target_face.left() + target_face.width() // 2, target_face.top() + target_face.height() // 2), cv2.NORMAL_CLONE)

    # Apply gamma correction to the result
    source_face_regions = [(p.x, p.y, 1, 1) for p in source_landmarks.parts()]
    target_face_regions = [(p.x, p.y, 1, 1) for p in target_landmarks.parts()]
    result = apply_gamma_correction(result, source_face_regions, gamma=0.5)
    result = apply_gamma_correction(result, target_face_regions, gamma=0.5)

    # Draw landmarks on the result image
    for point in source_landmarks.parts():
        cv2.circle(result, (point.x, point.y), 2, (0, 255, 0), -1)
    for point in target_landmarks.parts():
        cv2.circle(result, (point.x, point.y), 2, (255, 0, 0), -1)

    return result

def apply_gamma_correction(image, face_regions, gamma=0.5):
    corrected_image = image.copy()
    for region in face_regions:
        x, y, w, h = region
        face = corrected_image[y:y+h, x:x+w]
        face = np.power(face / 255.0, gamma) * 255.0
        corrected_image[y:y+h, x:x+w] = np.uint8(face)
    return corrected_image


# Example usage:
source_image_path = "deepfake_15.png"
target_image_path = "frame_15.png"
result = face_swap(source_image_path, target_image_path)

if result is not None:
    cv2.imshow("Face Swapped", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Face swapping failed.")
