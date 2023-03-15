## ISIC-2018-Challenge-Task-3-Disease-Classification 🏥

### Task : Image Classification - 피부 병변 이미지를 7개의 라벨로 분류. <br/>

[Challenge Link](https://challenge.isic-archive.com/landing/2018/)


---

## Description

### I. Dataset 
   - Kaggle Dataset 사용 - [Skin cancer: HAM10000](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification?select=masks)
   
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
        
        
### II. Progress
   -  Rank1 ~ 10 페이퍼 참고 및 정리, 사용해볼 기법들 정리 - [10개 페이퍼 리뷰 및 정리한 노션 페이지](https://www.notion.so/Rank1-10-5aa47146a64d45a7a548dc4291e7993d?pvs=4)
      - 논문에서 사용한 전처리(Augmentation, resize등), 사용한 모델, 클래스 불균형에 대한 접근방법, 성능지표, 앙상블 여부 등
   - 베이스 모델 생성 후 성능 확인
   - 논문에서 사용한 전처리 기법들, Class Imbalance에 대한 접근법  사용한 해결방법
   - 모델 앙상블

#### 2.1 베이스 모델 생성 후 성능 확인
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


#### 2.2 논문에서 사용한 전처리 기법들, Class Imbalance에 대한 접근법  사용한 해결방법
- 위의 데이터 분석 부분에서도 언급했듯이 가장 많은 라벨 NV는 6700장, 가장 적은 라벨 DF는 115장으로 대략 60배의 차이를 보인다. 
  
  즉, NV로 찍기만 해도 60%의 정확도가 나오므로 이대로 사용한다면 편향된 모델이 학습될 것이다. 
  
  이 클래스 불균형에 대해 논문에서는 여러 방법들을 제시하고 사용했는데 나는 Augmentation, Class weights 를 사용하였다. 
  
  데이터 양이 적은 특정 라벨 몇개 혹은 NV를 제외한 라벨 전체에 대해서 Augmentation을 사용해 어느 정도의 라벨에 어느 정도의 이미지 증강을 해줬을 때 성능이 좋았는지, 원본 데이터셋과 증강된 데이터셋에 Class weights를 적용시 어느 것이 성능이 좋은지 각각의 조건에 대해 성능을 확인 및 비교를 하였다.
  
  <br/>

- 사용한 전처리와 Augmentation에는 tf.Data API를 이용하여 파이프라인을 만들었다. 
   
  처음에는 이미지 리사이즈와 정규화를 사용하였고 후에 다양한 이미지 변형 방법들을 추가하여 처리하도록 하였다.


   <details>
   <summary>간단한 전처리 및 데이터셋 생성 코드</summary>
   <div markdown="1">
      
   ```Python
   main_path = '/content/images/'

   train_paths, test_paths = [], []

   for filename in train_df.image:
     train_paths.append(main_path + filename + '.jpg')

   for filename in test_df.image:
     test_paths.append(main_path + filename + '.jpg')

   train_paths = np.array(train_paths)
   test_paths = np.array(test_paths)

   train_labels = train_df.label.values
   test_labels = test_df.label.values

   len(train_paths), len(train_labels), len(test_paths), len(test_labels)
   
   s = np.arange(len(train_paths))
   np.random.shuffle(s)

   train_paths = train_paths[s]
   train_labels = train_labels[s]
   X_train, X_val, y_train, y_val = train_test_split(train_paths, train_labels, test_size=0.2, random_state=42)
         

   ss = s = np.arange(len(test_paths))
   np.random.shuffle(ss)

   test_paths = test_paths[ss]
   test_labels = test_labels[ss]

   def preprocessing(path, label):
     img = tf.io.read_file(path)
     img = tf.io.decode_jpeg(img)
     img = tf.image.resize(img, (256,256))
     img = img/255
     return img, label

   batch_size = 16

   train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
   val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
   test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

   train_dataset = train_dataset.shuffle(len(train_paths))
   train_dataset = train_dataset.map(preprocessing).batch(batch_size).prefetch(1)
   val_dataset = val_dataset.map(preprocessing).batch(batch_size).prefetch(1)
   test_dataset = test_dataset.map(preprocessing).batch(batch_size).prefetch(1)
   ```
      
   </div>
   </details>
   
   <br/>

- 내가 살펴본 페이퍼들에서는 다양한 이미지 변형을 일정 범위를 줘서 그 안에서 랜덤으로 변형하는 방식을 많이 사용하였다. 

  offline Augmenation에 사용된 이미지 처리 방법
   - random square corp
   - random horizontal/vertical flip 
   - random rotation 
   - random brightness 
   - random contrast
  
   <details>
   <summary>Offline Image Augmentation Image Processing </summary>
   <div markdown="1">

      ```
      def rotation(img, angle=90):
          angle = int(np.random.uniform(-angle, angle))
          h, w = img.shape[:2]
          M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
          img = cv2.warpAffine(img, M, (w, h))
          return img

      def brightness(img, low, high):
          value = np.random.uniform(low, high)
          hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
          hsv = np.array(hsv, dtype = np.float64)
          hsv[:,:,1] = hsv[:,:,1]*value
          hsv[:,:,1][hsv[:,:,1]>255]  = 255
          hsv[:,:,2] = hsv[:,:,2]*value 
          hsv[:,:,2][hsv[:,:,2]>255]  = 255
          hsv = np.array(hsv, dtype = np.uint8)
          img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
          return img

      def contrast(gray, min_val, max_val):
          #gray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
          alpha = int(np.random.uniform(min_val, max_val)) # Contrast control
          adjusted = cv2.convertScaleAbs(gray, alpha=alpha)
          return adjusted

      def random_crop(image):
        cropped_image = tf.image.random_crop(
            image, size=[400,400, 3])

        return cropped_image


      def random_image(img): # 좌우, 위아래 반전 / 반시계 90도 회전 3가지
        random = np.random.randint(0,5)
        if random == 0:
          image = cv2.flip(img,1) # 1은 좌우반전, horizontal_flip

        elif random == 1:
          image = cv2.flip(img,0) # 0은 위아래 반전, vertical_flip

        elif random == 2:
          image = rotation(img) # -90도에서 90도 범위 내 랜덤으로 rotation.

        elif random == 3:
          image = brightness(img, 0.5, 1.5)

        elif random == 4:
          #   image = contrast(img, 0.8, 1.5)
          image = np.array(random_crop(img))

        return image  
      
         
         ```
      
   </div>
   </details>

 
 <br/>

- Class weight. 클래스 불균형이 심할 경우 클래스당 데이터의 개수를 계산하여 각각의 클래스에 가중치를 부여할 수 있다. 사이킷런의 class_weight.compute_class_weight를 통해 한번에 계산할 수 있다
- 데이터가 적을수록 가중치가 커지고 많을수록 가중치는 작아진다. 

   <details>
   <summary>클래스별 가중치 계산 코드 및 결과값</summary>
   <div markdown="1">

   ```python
   weights = class_weight.compute_class_weight(class_weight = "balanced" , 
                                     classes=np.unique(test_labels), 
                                     y = test_labels)

   weights = {i : weights[i] for i in range(7)}
   weights
   ```
      
      
    라벨 |	이미지 개수	| 클래스별 가중치 |
    -----|---------------|----------------|
    0	|295|	4.455357142857143
    1	|463|	2.795518207282913
    2	|990|	1.307994757536042
    3	|104|	12.96103896103896
    4	|1002|	1.2844272844272844
    5	|6035|	0.21279317697228145
    6	|128|	10.183673469387756

   </div>
   </details>




```
<br/>
<br/>



<details>
<summary>10개 논문 리뷰 후 진행방향</summary>
<div markdown="1">

   

</div>
</details>
```
   






