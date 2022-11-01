# Volume Measure

- **물체의 체적 정보 측정 ( 제한사항 최소화, 오차범위 5mm 이내 )**
- 마트나 편의점의 물건 적제에 필요한 체적 정보 측정
![image](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/10ba33dd-c53a-46d3-a444-a24aae27e276/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221101%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221101T103242Z&X-Amz-Expires=86400&X-Amz-Signature=a3aee6a3a6b505f4c1d22ab545adefa729ab8c1f79c16664cfa04bd067fedb1a&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)


# 📝 Summary

마트나 편의점에서는 많은 상품들을 효과적으로 진열하는 것이 중요합니다. 따라서 해당 프로젝트가 유통 받은 물체들의 체적 정보들을 즉시 측정하고 해당 정보들로 최적의 진열 방법을 찾을 수 있는 바탕이 될 것입니다. 더불어 택배 상하차 과정에서 택배 차량에 가장 많은 상품을 효과적으로 적재하는 방법 등 다양한 유통 산업에 이용 가능할 것으로 보입니다. 다른 센서나 여러 대의 카메라들이 필요없이 단안 카메라로 측정하기 때문에 설치 단가에서도 유리할 것으로 예측 됩니다.

# ⭐️ Key Function

- **단안 카메라**
    - 다른 하드웨어적인 도움없이 하나의 카메라 만으로 체적 정보를 측정
- **물체 형태의 종류**
    - 직육면체와 원기둥 형태만 측정가능
    - 상하차 시스템에서는 대부분 직육면체의 박스를 사용하므로 문제가 없을 것으로 예상
    - 편의점과 마트의 경우 높이문제보다는 가로 세로의 중요성이 더 크므로 상이할 것으로 예상

# **Calibration.npz 파일 생성 방법**

[final_make_calibraton.py]이 정상적으로 실행되기 위해선 아래와 같이 과정이 필요합니다.

추가적인 정보는 아래 링크를 참조하시기 바랍니다.

[[OpenCV] 07-1. Camera Calibration — 참신러닝 (Fresh-Learning) (tistory.com)](https://leechamin.tistory.com/345)

### 체커보드를 들고 있는 사진

[Camera Calibration Pattern Generator – calib.io](https://calib.io/pages/camera-calibration-pattern-generator)

# 사용 방법

```python
import glob
from volume_measure import volumetric

main_path = "."
calibration_path = main_path + "/calibration" + "/cs_(8, 5)_rd_3_te_0.06_rs_4.npz"

hexahedron = glob.glob(main_path + "/hexahedron/*.jpg")
cylinder = glob.glob(main_path + "/cylinder/*.jpg")

images = hexahedron + cylinder

for fname in images:
    try:
        test = volumetric(fname, calibration_path)
        test.set_init()
        test.set_npz_values()

        # 1. 배경제거
        test.remove_background()

        # 2. 물체 꼭지점 찾기
        test.find_vertex(draw=False)
        test.fix_vertex()

        test.trans_checker_stand_coor()
        test.set_transform_matrix()

        # 3. 가로세로 구하기
        test.measure_width_vertical()

        # 4. 높이 구하기
        test.measure_height(draw=True)

        # 이미지에 최적정보들을 그리기
        test.draw_image()

        test.show_image(test.img, "Result Image")
        cv2.waitKey()

        cv2.destroyAllWindows()
    except:
        continue
```


# 📷 Screenshot(result) 
- 단위 : cm
- hexahedron
![image](https://drive.google.com/uc?export=view&id=16XEimDh3hfWV0f0Ds8dpFusU5i7LtNC8)
- cylinder
![image](https://drive.google.com/uc?export=view&id=1bW5UwwkYER18Mismg6gOChcaaZMl_qUW)

## ****References****

- **Rembg**:[https://github.com/danielgatis/rembg](https://github.com/danielgatis/rembg)
- **Opencv_Calibration**:[https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- **ORB**:[https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html](https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html)

### **License**

Copyright (c) 2022 Yeardream-Power-4-team

Please contact me gomugomutree@gmail.com if you need help
