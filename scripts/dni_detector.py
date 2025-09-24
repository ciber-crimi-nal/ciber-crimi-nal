from typing import Literal
from ultralytics import YOLO
from scripts.dni import *
import scripts.utils as utils
from scripts.utils import *
import numpy as np
from numpy import ndarray



def adapt_image(dni_image: cv2.Mat | ndarray, element: str) -> cv2.Mat | ndarray:
    if element not in POSICIONES.keys():
        return None

    posiciones = POSICIONES[element]
    x1 = int(posiciones[0] * dni_image.shape[1])
    y1 = int(posiciones[1] * dni_image.shape[0])
    x2 = int(posiciones[2] * dni_image.shape[1])
    y2 = int(posiciones[3] * dni_image.shape[0])

    cv2.rectangle(dni_image, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    return dni_image

def censor_dni_fields(dni_image: cv2.Mat | ndarray, side: str, debug=False) -> cv2.Mat | ndarray:
    elements = ELEMENTOS_FRONTALES if side == FRONTAL else ELEMENTOS_TRASEROS

    for element in elements:
        dni_image = adapt_image(dni_image, element)
        show_debug(debug, dni_image)

    return dni_image

def pass_yolo(img: cv2.Mat, debug=False) -> cv2.Mat:
    model = YOLO("scripts/model_prov.pt")
    result = model(img)

    if len(result) > 0:
        points = result[0].obb.xyxyxyxy.cpu().numpy().astype(np.int32)

        if debug:
            result[0].show()

        if len(points) > 0:
            points = points[0]
            x1, y1, x2, y2 = utils.positive_coords(result[0].obb.xyxy.cpu().numpy().astype(np.int32)[0])

            mat = np.array([[x1, y1] for _ in range(4)])
            points = np.subtract(points, mat)

            points = np.array(sorted(points, key=lambda p: np.linalg.norm(p)))
            origen = points[0]
            points = np.array(sorted(points, key=lambda p: np.linalg.norm(p - origen)))

            dni_src = points[0:3].astype(np.float32)
            dni_dst = np.array([[0, 0], [0, y2-y1], [x2-x1, 0]]).astype(np.float32)

            mat = cv2.getAffineTransform(dni_src, dni_dst)
            img = cv2.warpAffine(img[y1:y2, x1:x2], mat, (x2-x1, y2-y1))
            show_debug(debug, img)
            cv2.imwrite("memoria/primer_recorte.jpg", img)

    return img

def process_homography(dni_image: cv2.Mat | ndarray, side: str, debug=False, other_template=False) -> cv2.Mat | ndarray:

    if other_template:
        template = cv2.imread(f"templates/{side}.png")
    else:
        template = cv2.imread(f"templates/{side}.png")

    template = cv2.resize(template, (dni_image.shape[1], dni_image.shape[0]))
    cv2.imwrite("memoria/plantilla.jpg", template)

    n_points = 10500

    orb_points = cv2.ORB.create(n_points)
    dni_keypoints, dni_descriptors = orb_points.detectAndCompute(dni_image, np.array([]))
    template_keypoints, template_descriptors = orb_points.detectAndCompute(template, np.array([]))

    matcher = cv2.DescriptorMatcher.create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
    matches = matcher.match(dni_descriptors, template_descriptors, None)

    matches = sorted(matches, key=lambda match: match.distance)
    matches = matches[0:int(len(matches)*0.2)]

    if debug:
        match_img = cv2.drawMatches(dni_image, dni_keypoints, template, template_keypoints, matches, cv2.Mat(np.array([])))
        show_debug(debug, match_img)
        cv2.imwrite(f"pruebas/{n_points}_{side}.jpg", match_img)

    dni_points = np.zeros((len(matches), 2), dtype='float')
    template_points = np.zeros((len(matches), 2), dtype='float')

    for i, m in enumerate(matches):
        dni_points[i] = dni_keypoints[m.queryIdx].pt
        template_points[i] = template_keypoints[m.trainIdx].pt

    homography, _ = cv2.findHomography(dni_points, template_points, method=cv2.RANSAC)
    dni_homography = cv2.warpPerspective(dni_image, homography, (dni_image.shape[1], dni_image.shape[0]))

    print_debug(debug, f"{dni_homography.shape}")
    show_debug(debug, dni_homography)
    # cv2.imwrite(f"pruebas/{n_points}_res.jpg", dni_homography)
    cv2.imwrite("memoria/segundo_recorte.jpg", dni_homography)

    return dni_homography

def read_picture(path_file: str, side: Literal["front", "back"], debug=False) -> cv2.Mat | np.ndarray | None:
    try:
        dni_image = cv2.imread(path_file)
        content_image = pass_yolo(dni_image, debug) 
        # content_image = find_content(dni_image, debug)

        if content_image is not None:
            homography_image = process_homography(content_image, side, debug)
            utils.show_debug(debug, homography_image)

            censored_image = censor_dni_fields(homography_image, side, debug)
            return censored_image.copy()

        return None

    except Exception as e:
        print(e)
        return None


def read_dni(path_files: [str, str], sides: [str, str], debug=False) -> dict[str, str] | bool:
    dni_data = get_dni()

    for path_file, side in zip(path_files, sides):
        dni_data = read_picture(path_file, side, debug)

    return dni_data