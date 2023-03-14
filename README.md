## ISIC-2018-Challenge-Task-3-Disease-Classification 🏥

### Task : Image Classification - 피부 병변 이미지를 7개의 라벨로 분류. <br/>

[Challenge Link](https://challenge.isic-archive.com/landing/2018/)


---

## Description

### 1. 데이터 : Kaggle Dataset 사용 - [Skin cancer: HAM10000](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification?select=masks)
   
   - 데이터 분석
     - 10015장의 이미지 및 라벨값이 담긴 메타데이터 파일(csv) (+Segmentation을 위한 마스킹이미지도 같이 들어있음)
     - 테스트셋은 대회 주최측이 보유하고 있으나 해당 챌린지가 끝나서 얻을 수 없으므로 라벨당 10%씩 분할하여 998장 테스트셋 생성함
     - 이미지 크기 : 450 x 600 x 3
     - 7개의 병변 
     <br/>
     
      <details>
      <summary>라벨의 약칭과 풀네임 </summary>
      <div markdown="1">

      label | lesion |
     |-------|--------|     
     | nv  | Melanocytic nevi |     
     | mel | Melanoma |     
     | bkl | Benign keratosis-like lesions|     
     | bcc | Basal cell carcinoma|
     | akiec | Actinic keratoses|     
     | vasc  | Vascular lesions|     
     | df | Dermatofibroma|

      </div>
      </details>
     
     
<br/>
     
     
     
   - 라벨당 이미지 개수
     <img src="https://user-images.githubusercontent.com/103362361/188297176-34f9c64e-ca4f-4f0b-bb6f-057ad3c0844e.png"  width="400" height="300"/>
     
     
     | label | count |
     |-------|-------|
     |NV|6705|
     |MEL|1113|
     |BKL|1099|
     |BCC|514|
     |AKIEC|327|
     |VASC|142|
     |DF|115|
     
  - Data Imbalance 존재.
  - NV가 뭐길래 이렇게 많은걸까??
    - 멜라닌 세포성 모반(Melanocytic nevus)
    - [Reference](https://velog.io/@jj770206/ISIC-dataset) 참고
        
        
---   
        
        
### 2. 진행 내용

   -  Rank1 ~ 10 페이퍼 참고 및 정리, 사용해볼 기법들 정리 - [10개 페이퍼 리뷰 및 정리한 노션 페이지](https://www.notion.so/Rank1-10-5aa47146a64d45a7a548dc4291e7993d?pvs=4)
      - 논문에서 사용한 전처리(Augmentation, resize등), 사용한 모델, 클래스 불균형에 대한 접근방법, 성능지표, 앙상블 여부 등
   - 베이스 모델 생성 후 성능 확인
   - 논문에서 사용한 전처리 기법들, Class Imbalance에 대한 접근법  사용한 해결방법
   - 모델 앙상블

#### 베이스 모델 생성 후 성능 확인
- 이미지 리사이즈와 정규화같은 기본적인 전처리를 한 후에 간단한 모델을 쌓아서 성능 확인을 함. 
   
   => accuracy = 0.71. 하지만 극단적인 클래스 불균형을 생각하면 NV라벨을 찍기만 해도 60%이상의 정확성이 나오므로 여기서 accuracy는 신뢰도가 떨어져 성능 지표로 사용할 수 없음.
   
   따라서 Precision / Recall / F1-score를 측정, 0.4712 / 0.2746 / 0.347 .
   
     <details>
      <summary>베이스라인 코드 </summary>
      <div markdown="1">

      model = Sequential()
      model.add(layers.Conv2D(128,3,padding='same', input_shape=[256,256,3]))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.MaxPooling2D(2))
      model.add(layers.Conv2D(128,3,padding='same'))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.MaxPooling2D(2))
      model.add(layers.Conv2D(256,3,padding='same'))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.MaxPooling2D(2))
      model.add(layers.Conv2D(512,3,padding='same'))
      model.add(layers.BatchNormalization())
      model.add(layers.Activation('relu'))
      model.add(layers.GlobalAveragePooling2D())
      model.add(layers.Dense(7, activation='softmax'))

      opt = keras.optimizers.SGD()
      loss = keras.losses.SparseCategoricalCrossentropy()


      es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

      model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
      model.fit(train_dataset, validation_data=val_dataset, epochs=40, verbose=1, callbacks=[es])

     </div>
     </details>
   

<br/>


#### 논문에서 사용한 전처리 기법들, Class Imbalance에 대한 접근법  사용한 해결방법
- 위의 데이터 분석 부분에서도 언급했듯이 가장 많은 라벨 NV는 6700장, 가장 적은 라벨 DF는 115장으로 대략 60배의 차이를 보인다. 
  
  즉, NV로 찍기만 해도 60%의 정확도가 나오므로 이대로 사용한다면 편향된 모델이 학습될 것이다. 
  
  이 클래스 불균형에 대해 논문에서는 여러 방법들을 제시하고 사용했는데 나는 Augmentation, Class weights 를 사용하였다. 
  
  데이터 양이 적은 특정 라벨 몇개 혹은 NV를 제외한 라벨 전체에 대해서 Augmentation을 사용해 어느 정도의 라벨에 어느 정도의 이미지 증강을 해줬을 때 성능이 좋았는지, 원본 데이터셋과 증강된 데이터셋에 Class weights를 적용시 어느 것이 성능이 좋은지 각각의 조건에 대해 성능을 확인 및 비교를 하였다.

   




<br/>
<br/>



<details>
<summary>10개 논문 리뷰 후 진행방향</summary>
<div markdown="1">
1. 전처리 기법(Augmentation, resize 등)
2. 사용한 모델
3. 클래스 불균형에 대한 해결방법(over/undersampling, class weight 등)
4. 사용한 성능지표 및 모델 검증 방법(대회에서 지정한 성능지표)
5. 앙상블 방법


</div>
</details>

   






