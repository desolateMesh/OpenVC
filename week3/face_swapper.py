import cv2
import numpy as np
import dlib
import sys
import os

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    This finds INPUT files inside the bundled executable.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

# (All other functions like get_landmarks, etc., remain the same)
def get_landmarks(image, gray_image, detector, predictor, face_index=0):
    faces = detector(gray_image)
    if not faces: return None
    faces = sorted(faces, key=lambda rect: rect.left())
    if face_index >= len(faces): return None
    return np.array([[p.x, p.y] for p in predictor(image, faces[face_index]).parts()])
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]: index = num; break
    return index
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def main():
    # --- FIX 1: Determine the correct path for the OUTPUT file ---
    # This checks if the script is running as a bundled executable
    if getattr(sys, 'frozen', False):
        # If so, the output path is the directory of the executable
        output_dir = os.path.dirname(sys.executable)
    else:
        # Otherwise, it's the directory of the script file
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # Use resource_path for INPUT files
    predictor_path = resource_path("shape_predictor_68_face_landmarks.dat")
    source_image_path = resource_path("my_picture.jpg")
    dest_image_path = resource_path("destination.jpg")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img_source = cv2.imread(source_image_path)
    img_dest = cv2.imread(dest_image_path)
    
    # ... (All the image processing code remains exactly the same)
    gray_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    gray_dest = cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)
    print("Detecting facial landmarks...")
    landmarks_source = get_landmarks(img_source, gray_source, detector, predictor, 0)
    landmarks_dest = get_landmarks(img_dest, gray_dest, detector, predictor, 0)
    if landmarks_source is None or landmarks_dest is None: return
    print("Creating face mask and triangulation...")
    dest_hull = cv2.convexHull(landmarks_dest)
    rect = cv2.boundingRect(np.float32(landmarks_dest))
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_dest.tolist())
    triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
        index_pt1 = extract_index_nparray(np.where((landmarks_dest == pt1).all(axis=1)))
        index_pt2 = extract_index_nparray(np.where((landmarks_dest == pt2).all(axis=1)))
        index_pt3 = extract_index_nparray(np.where((landmarks_dest == pt3).all(axis=1)))
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            indexes_triangles.append([index_pt1, index_pt2, index_pt3])
    print("Warping source face to destination...")
    img_source_warped = np.zeros(img_dest.shape, dtype=img_source.dtype)
    for ti in indexes_triangles:
        tr1_pt1, tr1_pt2, tr1_pt3 = landmarks_source[ti[0]], landmarks_source[ti[1]], landmarks_source[ti[2]]
        tr2_pt1, tr2_pt2, tr2_pt3 = landmarks_dest[ti[0]], landmarks_dest[ti[1]], landmarks_dest[ti[2]]
        triangle1, triangle2 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32), np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
        rect1, rect2 = cv2.boundingRect(np.float32([triangle1])), cv2.boundingRect(np.float32([triangle2]))
        tr1_rect = [((triangle1[i][0] - rect1[0]), (triangle1[i][1] - rect1[1])) for i in range(3)]
        tr2_rect = [((triangle2[i][0] - rect2[0]), (triangle2[i][1] - rect2[1])) for i in range(3)]
        mask = np.zeros((rect2[3], rect2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tr2_rect), (1.0, 1.0, 1.0), 16, 0)
        img1_rect = img_source[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
        warp_img2 = apply_affine_transform(img1_rect, tr1_rect, tr2_rect, (rect2[2], rect2[3]))
        img_source_warped[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] = img_source_warped[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]] * (1 - mask) + warp_img2 * mask
    print("Blending images for a seamless result...")
    mask = np.zeros(img_dest.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dest_hull), (255, 255, 255))
    r = cv2.boundingRect(np.float32(dest_hull))
    center = ((r[0] + int(r[2] / 2)), (r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(img_source_warped, img_dest, mask, center, cv2.NORMAL_CLONE)


    # --- FINAL SAVE AND DISPLAY SECTION ---
    print("Process complete. Saving result...")
    
    # Use the output_dir we defined earlier to save the file
    output_filename = "deepfake_output.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output)
    print(f"Success! Result saved to: {output_path}")

    print("\nAttempting to display images...")
    try:
        cv2.imshow("Face Swap Result", output)
        print("Display window opened. Press any key in the window to exit.")
        cv2.waitKey(0)
        # --- FIX 2: Move destroyAllWindows() INSIDE the try block ---
        # It only runs if the imshow calls were successful.
        cv2.destroyAllWindows()
    except cv2.error:
        print("-----------------------------------------------------------------")
        print("Could not open display windows (this is expected on a server/VM).")
        print("Your output file has been saved successfully.")
        print("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()