import numpy as np
import glob, cv2
import utils
import timeit
from rembg import remove

class volumetric:
    
    def __init__(self, image_address: str, npz_file: str):

        self.image_address = image_address
        self.origin_image = cv2.imread(image_address)
        self.npz_file = npz_file

        self.h = self.origin_image.shape[0]
        self.w = self.origin_image.shape[1]

    def set_init(self):
        npz = self.npz_file.split("/")[-1] # "./30'_19cm + 40cm/Calibration/cs_(8, 5)_rd_3_te_0.06_rs_4.npz
        npz = npz.split("_") # cs_(8, 5)_rd_3_te_0.06_rs_4.npz
        self.checker_sizes = (int(npz[1][1]), int(npz[1][4])) # 8, 5
        self.check_real_dist = int(npz[3])
        self.resize = int(npz[-1].split(".")[0])

        self.img = cv2.resize(self.origin_image, (self.w //self.resize, self.h //self.resize))
        self.h = self.img.shape[0]
        self.w = self.img.shape[1]

        if "hexahedron" in self.image_address:
            self.object_type = "hexahedron"
        elif "cylinder" in self.image_address:
            self.object_type = "cylinder"
        else:
            self.object_type = "circle"

    def set_image(self, image_address: str):
        self.img = cv2.imread(image_address)
    
    # 배경 제거
    def remove_background(self):
        self.remove_bg_image = remove(self.img)

    # 코너 사이즈, 보정 코너들, 회전 벡터, 변환 벡터
    def set_npz_values(self):
        self.camera_matrix, self.dist, self.rvecs, self.tvecs, self.outer_points1, self.checker_sizes = utils.read_npz(self.npz_file)

    def find_vertex(self, draw=False):
        '''
        물체 꼭지점 6좌표 추출하는 함수
        draw : 그리기
        '''
        gray_img = cv2.cvtColor(self.remove_bg_image, cv2.COLOR_BGR2GRAY)

        kernel_custom = np.array([[0,0,1,0,0],
                                  [0,1,1,1,0],
                                  [1,1,1,1,1],
                                  [0,1,1,1,0],
                                  [0,0,1,0,0]],dtype=np.uint8)
        opening_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel_custom)

        kernel = np.ones((5, 5), np.uint8)
        opening_img = cv2.erode(opening_img, kernel, iterations = 1)

        kernel_clear = np.array([[0, -1, 0],
                                 [-1, 9, -1],
                                 [0, -1, 0]])
        #선명한 커널 적용 
        self.object_detection_image = cv2.filter2D(opening_img, -1, kernel_clear)

        # Find contours
        contours, _ = cv2.findContours(self.object_detection_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # contours는 튜플로 묶인 3차원 array로 출력

        image = self.img.copy()

        # vertex 출력값 그려보기
        vertex_list = list()
        check = 0
        for cnt in contours:
            for eps in np.arange(0.001, 0.2, 0.001):
                length = cv2.arcLength(cnt, True)
                epsilon = eps * length
                vertex = cv2.approxPolyDP(cnt, epsilon, True)

                if self.object_type == "hexahedron" and len(vertex) == 6 and length > (1000//(self.resize**2)):      # vertex가 6 -> 꼭짓점의 갯수
                    vertex_list.append(vertex)
                    self.object_vertexes = np.reshape(vertex_list, (-1, 2))
                    check = 1
                    break
                elif self.object_type == "cylinder" and 10 < len(vertex) < 50 and length > (1000//(self.resize)) and cv2.contourArea(vertex) >(5000 //self.resize):    
                    vertex_list.append(vertex)
                    self.object_vertexes = np.array(vertex_list).reshape(-1, 2) 
                    self.cylinder_mask_img = np.zeros(self.img.shape).astype(self.img.dtype)
                    cv2.drawContours(self.cylinder_mask_img,[vertex],0,(0,0,255),2, cv2.LINE_AA)
                    check = 1
                    break
        #########  수정 필요 #############
                elif self.object_type == "circle" and len(vertex) == 8 and length > (1000//(self.resize**2)):  
                    self.vertexes = np.reshape(vertex_list, (-1, 8, 2))
                    check = 1
                    break
            if check ==1:
                break
        
        self.vertexes_image = self.img.copy()
        cv2.drawContours(self.vertexes_image,[vertex],0,(0,0,255),2, cv2.LINE_AA)
        
        if draw:
            vertexes_image = self.img.copy()
            cv2.drawContours(vertexes_image,[vertex],0,(0,0,255),1)
            vertexes_image = cv2.resize(vertexes_image, (self.w // 6 * self.resize, self.h // 6 * self.resize))
            cv2.imshow(f"vertexes_image", vertexes_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
   
        if len(self.object_vertexes) == 0:
            print("object vertexes are not detected....")
            quit() 

    def fix_vertex(self):
        '''
        꼭지점 좌표들을 원하는 순서대로 정렬해주는 함수
        '''
        if self.object_type == "hexahedron":
            # 최소 y좌표
            y_coors = np.min(self.object_vertexes, axis=0)[1]

            # 좌상단 좌표가 index 0 번 -> 반시계 방향으로 좌표가 돌아간다.
            contours = self.object_vertexes.tolist()
            while y_coors != contours[-1][1]:
                temp = contours.pop(0)
                contours.append(temp)
            self.object_vertexes = np.array(contours)

        elif self.object_type == "cylinder":
            # 특징점 알고리즘 객체 생성 (ORB)
            feature = cv2.ORB_create(scaleFactor=1.2, edgeThreshold=100,)   

            kp1 = feature.detect(self.cylinder_mask_img)
            _, desc1 = feature.compute(self.cylinder_mask_img, kp1)

            corners = [list(map(int, kp1[i].pt)) for i in range(len(kp1))]

            center, r, d = cv2.fitEllipse(self.object_vertexes)
            center = np.array(center)

            quadrant1 = list()
            quadrant2 = list()
            quadrant3 = list()
            quadrant4 = list()

            # 중심점보다 작은지로 4사분면으로 코너 점들을 나눈다
            for idx, cor in enumerate(corners):
                
                sort_quadrant  = (cor < center).tolist()

                if sort_quadrant == [False, True]: # 1사분면
                    quadrant1.append(corners[idx])
                elif sort_quadrant  == [True, True]: # 2사분면
                    quadrant2.append(corners[idx])
                elif sort_quadrant  == [True, False]: # 3사분면
                    quadrant3.append(corners[idx])
                elif sort_quadrant  == [False, False]: # 4사분면
                    quadrant4.append(corners[idx])


            # 사분면의 값들을 리스트로 만들어준다 -> [2, 3, 4, 1] 사분면 순서로 정리해준다
            # 나중에 체적 정보를 측정할때 순서를 위한 것
            quadrant_list = [quadrant2, quadrant3, quadrant4, quadrant1]

            # 각 분면마다 center와 가장 멀리 떨어져 있는값을 그려준다 (맨하탄 거리 - 파란색)
            center = center.tolist()
            
            self.object_vertexes = []
            for qua in quadrant_list:
                longest_dist = [0, 0]
                for idx, q in enumerate(qua):
                    # dist = utils.euclidean_distance(center, q) # 유클리드

                    # 맨하탄 거리 : x 축과 y축에 가중치를 준다.
                    x, y = abs(np.array(center) - np.array(q))   
                    dist = (x * 1.5) + (y * 0.8)

                    if longest_dist[0] < dist:
                        longest_dist = [dist, idx]
                
                # 중심점에서 각 사분면 마다 맨하탄 거리가 가장 먼 좌표 넣는다
                self.object_vertexes.append(qua[longest_dist[1]])

            self.object_vertexes = np.array(self.object_vertexes)

    def trans_checker_stand_coor(self):# point: list, stand_corr: tuple, checker_size: tuple) -> list:
        """
        이미지상의 4개의 좌표를 일정한 간격으로 펴서 4개의 좌표로 만들어주는 함수
        """
        # x, y 비율과 똑같이 ar 이미지에 투시한다.
        # 첫번째 좌표를 기준으로 오른쪽에서 x, 아래쪽 좌표에서 y 간격(비율)을 구해준다.
        # 1칸당 거리 구하기
        one_step = abs(self.outer_points1[0][0] - self.outer_points1[2][0]) / (self.checker_sizes[0] - 1)

        # y_ucl = abs(point[0][1] - point[1][1])

        w, h = (self.w, self.h * 2)
        self.outer_points2 = np.float32(
            [[w, h], 
            [w, h + one_step * (self.checker_sizes[1] - 1)], 
            [w + one_step * ((self.checker_sizes[0] - 1)), h], 
            [w + one_step * ((self.checker_sizes[0] - 1)), h + one_step * (self.checker_sizes[1] - 1)],]
        )

    # 투시 행렬 구하기
    def set_transform_matrix(self):
        self.transform_matrix = cv2.getPerspectiveTransform(self.outer_points1, self.outer_points2)
     
    def measure_width_vertical(self):
        """
       printer : 가로, 세로 길이 출력문 실행 여부 - bool
        """
        re_point = list()
        checker_points = self.outer_points1
        checker_points = checker_points.tolist()
        # 체커보드가 정방향으로 투시되었을때 각 좌표들을 다시 구해준다.
        for point in checker_points:
            re_point.append(utils.transform_coordinate(self.transform_matrix, point))

        re_object_points = list()
        re_checker_points = self.object_vertexes.tolist()

        for point in re_checker_points:
            re_object_points.append(utils.transform_coordinate(self.transform_matrix, point))

        # pt2[0]의 x축과 pt2[2]의 x축의 픽셀 거리 // 코너 사이즈 - 1 (칸) = 1칸당 떨어진 픽셀거리
        one_checker_per_pix_dis = abs(re_point[0][0] - re_point[2][0]) / (
            self.checker_sizes[0] - 1
        )

        # 픽셀당 실제 거리 - check_real_dist(cm) / 1칸당 떨어진 픽셀 거리
        self.pix_per_real_dist = self.check_real_dist / one_checker_per_pix_dis

        if self.object_type == "hexahedron":
        # 두 점 사이의 픽셀거리 * 1픽셀당 실제 거리 = 두 점의 실제 거리
            self.width = (
                utils.euclidean_distance(re_object_points[1], re_object_points[2]) * self.pix_per_real_dist
            )
            self.vertical = (
                utils.euclidean_distance(re_object_points[2], re_object_points[3]) * self.pix_per_real_dist
            )
        elif self.object_type == "cylinder":
            self.width = (
                utils.euclidean_distance(re_object_points[1], re_object_points[2]) * self.pix_per_real_dist
            )
        else:
            print("구 입니다")

    def measure_height(self, draw=True):
        """
        높이 측정 함수
        """
        pts1 = self.outer_points1.tolist()
        ar_start = utils.transform_coordinate(self.transform_matrix, pts1[0])
        ar_second = utils.transform_coordinate(self.transform_matrix, pts1[2])
        
        vertexes_list = self.object_vertexes[1].tolist()
        ar_object_standard_z = utils.transform_coordinate(self.transform_matrix, vertexes_list)

        # 두 점을 1으로 나눈 거리를 1칸 기준 (ckecker 사이즈에서 1 빼면 칸수)
        standard_ar_dist = abs(ar_start[0] - ar_second[0]) / (self.checker_sizes[0] - 1)  

        # 실제 세계의 기준 좌표를 기준으로 물체의 z축을 구할 바닥 좌표의 실제 세계의 좌표를 구한다
        # x, y, z 값을 갖는다
        ar_object_real_coor = [
            (ar_object_standard_z[0] - ar_start[0]) / standard_ar_dist,
            (ar_object_standard_z[1] - ar_start[1]) / standard_ar_dist,
            0,
        ]

        # pixel_coordinates 
        height_pixel = utils.pixel_coordinates(self.camera_matrix, self.rvecs, self.tvecs, ar_object_real_coor)
        # y축으로 비교해서 z 수치가 증가하다가 물체 높이보다 높아지면 break
        for i in np.arange(0, 10, 0.01):
            if (height_pixel[1] - self.object_vertexes[0][1]) < 0:
                break

            height_pixel = utils.pixel_coordinates(
                self.camera_matrix, self.rvecs, self.tvecs, (ar_object_real_coor[0], ar_object_real_coor[1], -i)
            )
            self.height = i
            if draw:
                self.img = cv2.circle(self.img, tuple(list(map(int, height_pixel[:2]))), 1, (0, 0, 255), -1, cv2.LINE_AA)

    def draw_image(self, printer=False):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 가로, 세로, 높이 출력
        if self.object_type == "hexahedron":
            if printer:
                print("육면체")
                print("가로길이 :",self.width)
                print("세로길이 :",self.vertical)
                print("높이길이 :",self.height * self.check_real_dist)
                print(f"{self.width: .2f} x {self.vertical: .2f} x {(self.height * self.check_real_dist): .2f}")
                
            # 가로세로 그리기
            cv2.putText(self.img, f"{self.width: .2f}cm" , (self.object_vertexes[1][0]- (self.object_vertexes[1][0]//3), self.object_vertexes[1][1]+((self.h-self.object_vertexes[1][1])//3)), font, (3/self.resize) , (0, 255, 0), (10//self.resize ))
            cv2.putText(self.img, f"{self.vertical: .2f}cm" , (self.object_vertexes[3][0], self.object_vertexes[1][1]+((self.h-self.object_vertexes[3][1])//3)), font, (3/self.resize) , (255, 0, 0), (10//self.resize ))
            cv2.putText(self.img, f"{(self.height*self.check_real_dist): .2f}cm" , (self.object_vertexes[0][0] - (self.object_vertexes[0][0]//2), (self.object_vertexes[0][1] + self.object_vertexes[1][1])//2), font, (3/self.resize) , (0, 0, 255), (10//self.resize ))
            
            cv2.line(self.img,(self.object_vertexes[1]), (self.object_vertexes[2]), (0, 255, 0), (10//self.resize), cv2.LINE_AA)
            cv2.line(self.img,(self.object_vertexes[2]), (self.object_vertexes[3]), (255, 0, 0), (10//self.resize), cv2.LINE_AA)
        
        elif self.object_type == "cylinder":
            cylinder_real_volume = self.image_address.split("_")[1]
            real_width = float(cylinder_real_volume.split("x")[0])
            real_height = float(cylinder_real_volume.split("x")[1])
            
            if printer:
                print("원기둥")
                print("--- 측정 ----")
                print(f"지름 : {self.width: .2f}")
                print(f"높이 : {(self.height * self.check_real_dist): .2f}")
                print("--- 실제 ----")
                print(f"지름 : {real_width: .2f}")
                print(f"높이 : {real_height: .2f}")
                print("--- 오차 ----")
                print(f"지름 : {abs(self.width - real_width): .2f}")
                print(f"높이 : {abs((self.height * self.check_real_dist) - real_height): .2f}")
                print(f"산술오차율: {((abs(self.width - real_width)/real_width) + (abs((self.height * self.check_real_dist) - real_height)/real_height))/2 * 100:.2f}%")
                print(f"조화오차율: {1 / ((real_width/abs(self.width - real_width)) + (real_height / abs((self.height * self.check_real_dist) - real_height))/2 )* 100:.2f}%")
                
            cv2.putText(self.img, f"{self.width: .2f}cm" , ((self.object_vertexes[1][0]), self.object_vertexes[1][1]+ 50), font, (3/self.resize) , (0, 255, 0), (10//self.resize ))
            cv2.putText(self.img, f"{(self.height*self.check_real_dist): .2f}cm" , (self.object_vertexes[0][0] - (self.object_vertexes[0][0]//3)-30, (self.object_vertexes[0][1] + self.object_vertexes[1][1])//2), font, (3/self.resize) , (0, 0, 255), (10//self.resize ))
            cv2.line(self.img,(self.object_vertexes[1]), (self.object_vertexes[2]), (0, 255, 0), (10//self.resize), cv2.LINE_AA)
          
    def show_image(self, image: np.array, image_name:str):
        window_size_x, window_size_y = 600, 800
        cv2.namedWindow(f"{image_name}", cv2.WINDOW_NORMAL)
  
        # Using resizeWindow()
        cv2.resizeWindow(f"{image_name}", window_size_x, window_size_y)
        cv2.moveWindow(f"{image_name}", 400, 0)
        
        # Displaying the image
        cv2.imshow(f"{image_name}", image)
        # cv2.waitKey(0)

        # cv2.destroyAllWindows()

    def save_image(self, image_address: str, image: np.array):
        # image_address = "G:/download/image.jpg"
        cv2.imwrite(image_address, image)
    
    def time_check(self, time_check_number=10):
        t1 = timeit.timeit(stmt=self.remove_background,number=time_check_number, setup="pass" )
        t2_1 = timeit.timeit(stmt=self.find_vertex,number=time_check_number, setup="pass" )
        t2_2 = timeit.timeit(stmt=self.fix_vertex,number=time_check_number, setup="pass" )
        t3 = timeit.timeit(stmt=self.measure_width_vertical,number=time_check_number, setup="pass" )
        t4 = timeit.timeit(stmt=self.measure_height,number=time_check_number, setup="pass" )

        print(f"1.remove_background : {t1 / time_check_number} ")
        print(f"2.find_vertex : {(t2_1 + t2_2) / time_check_number}")
        print(f"3.measure_width_vertical : {t3 / time_check_number}")
        print(f"4.measure_height : {t4 / time_check_number}")
        print(f"total time : {(t1 + t2_1 + t2_2 + t3 + t4) / time_check_number}")

# if __name__ == '__main__':
#7. 전체
def main(fname, npz):
    a = volumetric(fname, npz)
    a.set_init()
    a.set_npz_values()

    # 1. 배경제거
    a.remove_background()

    # 2. 물체 꼭지점 찾기
    a.find_vertex(draw=False)
    a.fix_vertex()
    
    a.trans_checker_stand_coor()
    a.set_transform_matrix()
    
    # 3. 가로세로 구하기
    a.measure_width_vertical()
    
    # 4. 높이 구하기
    a.measure_height(draw=True)
    
    a.draw_image()

    a.show_image(a.origin_image, "Origin Image")
    cv2.waitKey()

    a.show_image(a.remove_bg_image, "Remove Background")
    cv2.waitKey(1500)

    a.show_image(a.object_detection_image, "Object Detection Image")
    cv2.waitKey(1500)

    a.show_image(a.vertexes_image, "vertexes Image")
    cv2.waitKey(1500)

    a.show_image(a.img, "Result Image")
    cv2.waitKey()

    cv2.destroyAllWindows()

    # a.save_image(image_address="G:/download/image.jpg", image=a.img)
    # a.time_check()
