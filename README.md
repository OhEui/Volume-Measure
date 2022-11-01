# volume measure

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
    

# 📷 Screenshot(result) 
- hexahedron
![image](https://drive.google.com/uc?export=view&id=16XEimDh3hfWV0f0Ds8dpFusU5i7LtNC8)
- cylynder
![image](https://drive.google.com/uc?export=view&id=1bW5UwwkYER18Mismg6gOChcaaZMl_qUW)
