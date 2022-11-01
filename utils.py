import numpy as np
import cv2
from math import dist


def draw_outer_pts(
    image: np.ndarray,
    points: np.ndarray,
    points_size=7,
    win_size=(800, 800),
    win_name="CV Window",
):
    """
    image: 최외곽 점들을 그릴 이미지
    points: 최외곽 점들
    """
    cv2.circle(image, tuple(map(int, points[0].tolist())), points_size, (0, 0, 255), -1)
    cv2.circle(image, tuple(map(int, points[1].tolist())), points_size, (0, 0, 255), -1)
    cv2.circle(image, tuple(map(int, points[2].tolist())), points_size, (0, 0, 255), -1)
    cv2.circle(image, tuple(map(int, points[3].tolist())), points_size, (0, 0, 255), -1)

    image = cv2.resize(image, win_size)
    cv2.imshow(win_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def euclidean_distance(point1: list, point2: list) -> float:
    """
    유클리드 거리 구하기
    """
    return dist(point1, point2)


def transform_coordinate(trans_coor: np.array, point: list) -> list:
    """
    좌표를 바꾸고자 하는 변환 행렬을 통과시켜주는 함수
    trans_coor : 변환 행렬 (3X3) - np.array
    point : 변환하고자 하는 좌표 - ex) [300, 500]
    result : 변환된 좌표
    """
    # 2 col -> 3 col -> 3 row 1 col
    re_point = point.copy()
    re_point.append(1)
    # re_point = np.array(re_point).reshape(3, -1).tolist()

    after = trans_coor @ re_point

    # 3 row -> 3 col
    after = after.reshape(-1)  # wx`, wy`, w
    # w로 나눠준다 -> x`, y`
    result = [(after[0] / after[2]), (after[1] / after[2])]

    return result


def trans_checker_stand_coor(point: list, stand_corr: tuple, checker_size: tuple) -> list:
    """
    ** 수정 필요 **
    이미지상의 4개의 좌표를 일정한 간격으로 펴서 4개의 좌표로 만들어주는 함수
    point : ex) np.float32([[931, 1411], [1101, 2033], [1667, 1189], [2045, 1706]])
    stand_coor : 새로 만드는 좌표의 좌측 상단의 기준 좌표
    result : point와 같은 형식 
    """

    # x, y 비율과 똑같이 ar 이미지에 투시한다.
    # 첫번째 좌표를 기준으로 오른쪽에서 x, 아래쪽 좌표에서 y 간격(비율)을 구해준다.
    # 1칸당 거리 구하기
    one_step = abs(point[0][0] - point[2][0]) / (checker_size[0] - 1)

    # y_ucl = abs(point[0][1] - point[1][1])

    w, h = stand_corr
    result = np.float32(
        [[w, h], 
        [w, h + one_step * (checker_size[1] - 1)], 
        [w + one_step * ((checker_size[0] - 1)), h], 
        [w + one_step * ((checker_size[0] - 1)), h + one_step * (checker_size[1] - 1)],]
    )

    return result

def make_cube_axis(checker_num: int, checker_size: tuple, cube_size: tuple) -> list:
    """
    체커 번호를 시작점으로 큐브와 xyz축을 그리기 위한 실제 세계의 좌표 구하기
    checker_num : 시작점 체커 번호 (int)
    checker_size : 내부 체커교차점 개수 (xline, yline) - ex) (7, 6)
    cube_size : (x, y, z) 형태 - ex) (3, 4, 5)
    return values : axis (xyz 축방향 3개 좌표), axisCube (8개 좌표) - np.float32
    """

    xline_size = checker_size[0]
    yline = checker_num // xline_size
    xline_number = xline_size * yline
    x, y, z = cube_size

    axis = np.float32(
        [
            [x + checker_num - xline_number, yline, 0],  # 파란색
            [checker_num - xline_number, y + yline, 0],  # 초록색
            [checker_num - xline_number, yline, -z],  # 빨강색
        ]
    ).reshape(-1, 3)
    axisCube = np.float32(
        [
            [checker_num - xline_number, yline, 0],  # [0, 0, 0],
            [checker_num - xline_number, y + yline, 0],  # [0, 3, 0],
            [x + checker_num - xline_number, y + yline, 0],  # [3, 3, 0],
            [x + checker_num - xline_number, yline, 0],  # [3, 0, 0],
            [checker_num - xline_number, yline, -z],  # [0, 0, -3],
            [checker_num - xline_number, y + yline, -z],  # [0, 3, -3],
            [x + checker_num - xline_number, y + yline, -z],  # [3, 3, -3],
            [x + checker_num - xline_number, yline, -z],  # [3, 0, -3],
        ]
    )
    return axis, axisCube

def pixel_coordinates(
    camera_mtx: np.ndarray, rvecs: np.ndarray, tvecs: np.ndarray, real_coor: tuple
) -> np.ndarray:
    """
    camera_mtx: npz에 있는 mtx
    rvecs: rotation 변환 행렬
    tvecs: translation 변환 행렬
    real_coor: 현실 좌표계의 좌표
    반환값인 pixel_coor: 이미지상에서의 좌표
    """

    # Rodgigues notation으로 된 회전변환 행렬을 3x3 회전변환 행렬로 변환
    rotation_mtx = cv2.Rodrigues(rvecs)

    translation_mtx = tvecs

    # np.hstack으로 회전변환과 병진변환 행렬을 합쳐 [R|t]행렬 생성
    # Rodrigues()를 쓰면 0번째로 회전변환 행렬, 1번째로 변환에 사용한 Jacobian행렬이 나오므로 0번째만 사용
    R_t = np.hstack((rotation_mtx[0], translation_mtx))

    # 실제 좌표를 3x1행렬로 변환
    real_coor = np.array(real_coor).reshape(-1, 1)

    # 실제 좌표 마지막 행에 1을 추가
    real_coor = np.vstack((real_coor, np.array([1])))

    # 이미지 좌표계에서의 픽셀 좌표 연산
    pixel_coor = camera_mtx @ R_t @ real_coor

    # 마지막 행을 1로 맞추기 위해 마지막 요소값으로 각 요소를 나눔
    pixel_coor /= pixel_coor[-1]
    return pixel_coor[:2]

def read_npz(npz_file):
    with np.load(npz_file) as X:
        camera_matrix, dist, rvecs, tvecs, outer_points, checker_size = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs', 'outer_points', 'checker_size')]
    return camera_matrix, dist, rvecs, tvecs, outer_points, checker_size
