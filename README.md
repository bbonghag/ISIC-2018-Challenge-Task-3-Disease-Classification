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
   - 베이스 모델 생성 후 성능 확인, 다른 모델 사용. 
   - 논문에서 사용한 전처리 기법들, Class Imbalance에 대한 해결방법
   - 모델 앙상블

#### 2.1 베이스 모델 생성 후 성능 확인
- 이미지 리사이즈와 정규화같은 기본적인 전처리를 한 후에 간단한 모델을 쌓아서 성능 확인. 그리고 ResNet152에 전이학습을 진행하였다.
   
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


#### 2.2 논문에서 사용한 전처리 기법들, Class Imbalance에 대한 해결방법
- 위의 데이터 분석 부분에서도 언급했듯이 가장 많은 라벨 NV는 6700장, 가장 적은 라벨 DF는 115장으로 대략 60배의 차이를 보인다. 
  
  즉, NV로 찍기만 해도 60%의 정확도가 나오므로 이대로 사용한다면 편향된 모델이 학습될 것이다. 
  
  이 클래스 불균형에 대해 논문에서는 여러 방법들을 제시하고 사용했는데 나는 Augmentation, Class weights 를 사용하였다. 
  
  데이터 양이 적은 특정 라벨 몇개 혹은 NV를 제외한 라벨 전체에 대해서 Augmentation을 사용해 어느 정도의 라벨에 어느 정도의 이미지 증강을 해줬을 때 성능이 좋았는지, 원본 데이터셋과 증강된 데이터셋에 Class weights를 적용시 어느 것이 성능이 좋은지 각각의 조건에 대해 성능을 확인 및 비교를 하였다.
  
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



  
  <br/>

- 사용한 전처리와 Augmentation에는 tf.Data API를 이용하여 파이프라인을 만들었다. 
   
  처음에는 이미지 리사이즈와 정규화를 사용하였고 후에 다양한 이미지 변형 방법들을 추가하여 처리하도록 하였다.


   <details>
   <summary>간단한 전처리 및 데이터셋 생성 코드</summary>
   <div markdown="1">
      
   ```Python
   <pre><code>   
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
   </code></pre>
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
   
   <br/>
  
   <details>
   <summary>Offline Image Augmentation Image Processing </summary>
   <div markdown="1">


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

      
   </div>
   </details>
   
   <br/>

- 위에서 사용한 베이스라인에서 데이터수가 극단적으로 부족한 3개의 클래스에 대해 어느정도의 Augmentation을 해줬을 때 성능이 얼마나 올라가는지 3번의 학습과 검증을 진행하였다.
   - 원본 데이터셋(SGD, 학습률은 디폴트값)
   - 0,3,6 클래스 2배 증강 (위와 동일 조건)
   - 0,3,6 클래스 1500장 증강 (SGD -> Adam, 학습률 디폴트값)
   - Accuracy는 신뢰도가 부족함. Precision, Recall, F1-score를 성능 지표로 사용하여 모델 검증
   - Augmentation은 좌우, 위아래 반전 / 반시계 90도 회전 3가지 처리만 간단하게 하였다.
   
   <br/>
   
   <details>
   <summary>해당 실험에서 사용한 Augmentation 코드 </summary>
   <div markdown="1">

   ```Python
   def random_image(img): # 좌우, 위아래 반전 / 반시계 90도 회전 3가지
     random = np.random.randint(0,2)
     if random == 0:
       image = cv2.flip(img,1) # 1은 좌우반전
     elif random == 1:
       image = cv2.flip(img,0) # 0은 위아래 반전
     elif random == 2:
       mat = cv2.getRotationMatrix2D(tuple(np.array(img.shape[:2]) /2), 90, 1.0) # 반시계방향으로 90도 회전
       image = cv2.warpAffine(img, mat, img.shape[:2])
     # elif random == 3:

     return image
     ```

   </div>
   </details>
   
   
   <br/>
   
   지표\데이터셋 | 원본 데이터셋 | 0,3,6 클래스 2배 증강 | 0,3,6 클래스 1500장 증강 |
   -------------|--------------|-----------------------|-------------------------|
   Precision | 0.4712 | 0.4198 | 0.5198 | 
   Recall | 0.2746 | 0.3067 | 0.5305 | 
   F1-Score | 0.3470 | 0.3545 | 0.5251 | 
   
   => 클래스 불균형 해소에 데이터수가 적은 클래스 일부에 대한 이미지 증강은 효과적. NV가 6000장임을 생각하면 나머지 라벨들도 3,4000장에서 많게는 5,6000장까지도 증강을 해서 성능을 확인해볼 필요가 있다. 그리고 단순회전, 뒤집기만 사용하면 같은 이미지가 반복되어 학습효과가 떨어질 것이므로 다양한 방법들을 추가해서 전처리 할 것. 

---
- 위의 실험의 연장선으로 0,3,6번 클래스 3000장씩 증강, 5번 NV를 제외한 나머지 라벨 3000장, 약 5000장씩 증강 후 학습, 성능 확인을 진행함.
- ISIC 2018 챌린지에서 요구하는 성능지표, Balanced_accuracy_score를 추가
- 성능이 가장 좋았던 0,3,6번 클래스 3000장씩 증가시킨 데이터셋에 class_weight를 적용시켜보았다

   지표\데이터셋 | 원본 데이터셋 | 0,3,6 클래스 2배 증강 | 0,3,6 클래스 1500장 증강 | 0,3,6 클래스 3000장씩 증강 | 5번 클래스 제외 3000장씩 증강| 5번 NV제외 모든 클래스 약 5000장까지 증강 | 0,3,6번 클래스 3000장씩 증강 | 
   -------------|--------------|-----------------------|-------------------------|--------------------|---------------|------------------|----------|
   Model | base-line | base-line | base-line | ResNet152 | ResNet152 | ResNet152 | ResNet152 | 
   Class_weights | x | x | x | x | x | x | y_train class_weights |
   Precision | 0.4712 | 0.4198 | 0.5198 | 0.6518 | 0.6428 | 0.5938 | 0.5678 |
   Recall | 0.2746 | 0.3067 | 0.5305 | 0.6566 | 0.6377 | 0.6458 | 0.6597 |  
   F1-Score | 0.3470 | 0.3545 | 0.5251 | 0.6542 | 0.6402 | 0.6187 | 0.6103 |
   balanced_accuracy_score | x | x | x | 0.6566 | 0.6377 | 0.6458 | 0.6597 | 
   
   => class_weights 적용 전까지는 0,3,6번 클래스 3000장씩 증강한 데이터셋의 성능이 가장 좋았음.
      
   => 가장 많은 라벨인 NV의 개수에 맞춰서 다른 클래스들의 이미지를 증강시켜주면 성능이 더 좋아질 것이라 생각했으나 오히려 떨어지는 것을 보니 과한 Augmentation은 중복된 이미지가 많아지므로 학습의 효과가 더 떨어지는 것으로 보인다. 
   
   => 0,3,6번 클래스를 3000장씩 증강시키고 class_weights를 계산해서인지 원본데이터셋 기준 class_weights만큼 극단적으로 가중치가 나오진 않았따.
   
   => 증강에 따른 성능 향상을 제대로 비교하고자 한다면 모든 조건을 통일했어야 했는데 이 점이 아쉽다. 
      0,3,6번 클래스 3000장씩 증강한 데이터셋이 성능이 가장 좋았는데 이 성능향상의 원인에는 ResNet152라는 더 좋은 모델을 사용한 것도 비중이 크기 때문에 꼭 Augmentation만이 성능향상에 기여를 했다고 할 수는 없다. 
      따라서 다음 프로젝트에서는 이렇게 성능을 비교 실험시 꼭 조건을 통일하도록 하자.



---
 




<br/>


#### 2.3 모델 앙상블 및 모델 검증
- 상위권 페이퍼 중 사용된 모델 ResNet152를 전이학습을 통해 학습을 시키고 다양한 Augmentation을 적용하여 증강, 학습시킨 ResNet152들을 앙상블하였다. 
- 앙상블은 모델별 예측한 클래스 확률값들을 클래스별로 합친 후에 가장 값이 높은 것들을 뽑는 방법을 사용하였다. 
- 여러 성능지표와 Confusion_matrix, classification_report를 통해 좀 더 자세한 모델의 검증을 

   <details>
   <summary>사용한 앙상블 코드 </summary>
   <div markdown="1">
   
   ```Python
   
   resnet152 = keras.models.load_model('/content/drive/MyDrive/skincancer_weighted_resnet152.h5')
   resnet152_2 = keras.models.load_model('/content/drive/MyDrive/skincancer_resnet152.h5')
   resnet152_3 = keras.models.load_model('/content/drive/MyDrive/skincancer_resnet152_ver2.h5')
   resnet152_4 = keras.models.load_model('/content/drive/MyDrive/skincancer_resnet152_ver3.h5')

   def pred_preprocessing(path, label):
     img = tf.io.read_file(path)
     img = tf.io.decode_jpeg(img)
     img = tf.image.resize(img, (224,224))
     img = tf.keras.applications.resnet.preprocess_input(img)
   #   img = img/255 # ResNet152는 -1~1 제로센터링을 입력받는걸로 기억한다. 
     return img

   batch_size = 16

   pred_test_dataset = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
   pred_test_dataset = pred_test_dataset.map(pred_preprocessing).batch(batch_size).prefetch(1)
   ######################################################################
   models = [resnet152, resnet152_2, resnet152_3, resnet152_4]
   y_pred = [model.predict(pred_test_dataset) for model in models]
   pred_sum = np.sum(y_pred, axis=0)
   label_pred = np.argmax(pred_sum, axis=1)
   ######################################################################
   precision = precision_score(test_labels, label_pred, average='macro')
   recall = recall_score(test_labels, label_pred, average= "macro")
   F1 = 2 * (precision * recall) / (precision + recall)
   balanced_acc_score = balanced_accuracy_score(test_labels, label_pred)
   ######################################################################
   print(f'정밀도 : {precision:.4f}')
   print(f'재현율 : {recall:.4f}')
   print(f'f1_score : {F1:.4f}')
   print(f'balanced_accuracy_score : {balanced_acc_score: .4f}')
   정밀도 : 0.6237
   재현율 : 0.6552
   f1_score : 0.6390
   balanced_accuracy_score :  0.6552
   ######################################################################
   cm = confusion_matrix(test_labels, label_pred)
   array([[ 17,   5,   5,   0,   5,   0,   0],
          [  6,  31,   6,   2,   0,   6,   0],
          [  6,   2,  64,   3,  17,  17,   0],
          [  0,   0,   0,   9,   1,   1,   0],
          [  2,   2,  20,   0,  45,  39,   3],
          [  1,   6,  10,   2,  31, 618,   2],
          [  0,   0,   0,   1,   1,   2,  10]])

   class_names = [0,1,2,3,4,5,6]
   report_df = pd.DataFrame(classification_report(test_labels, label_pred, target_names=class_names, output_dict=True)).T
   report_df
   ```
    <img src="https://user-images.githubusercontent.com/103362361/226217372-d3ef0ebd-a0a3-4398-8b93-a0404317dd1e.png"  width="400" height="300"/>

   </div>
   </details>


- 클래스별 RoC Curve 그래프
   
   <img src="https://user-images.githubusercontent.com/103362361/226217432-7642802f-9d8c-46c8-84c8-946e47b35d94.png"  width="400" height="300"/>


<br/>

---

## Review

- 0.65라는 아쉬운 점수로 마무리하였다. 
- 상위권의 페이퍼들이 전부 원서라 읽는데도 꽤 시간이 걸렸지만 잘하는 사람들은 어떻게 접근하여 어떤 방법들을 썼는지 등을 배울 수 있었다. 하지만 자세하게 방법을 설명해주는 페이퍼도 있고 사용한 방법만 언급하고 끝낸 간단한 페이퍼들도 있어서 그 점이 아쉬웠다
- 피부암 분류의 주 포인트는 적은 데이터와 극단적인 클래스 불균형에 대한 접근 및 해결방법. 따라서 대부분의 페이퍼들은 이에 대해 증강과 이미지 전처리 부분에 많은 지면을 할애하였다. 물론 앙상블은 기본, 모델 스태킹과 교차검증을 이용한 엄청난 양의 연산 방법을 사용하거나 분할과 연계하여 분류하는 방법 등 신박하고 좋은 아이디어들을 볼 수 있었다. 절반이상의 이미지를 차지하는 클래스인 NV와 나머지 클래스에 대해 이진분류 모델을 만들고, 6개 클래스를 분류하는 다중분류 모델 두 가지를 연계해 이용한다던가 분할을 통해 병변의 마스크를 얻은다음 백그라운드를 날려서 병변에 대한 학습만 하여 정확도를 높인다던가 등등 리소스가 부족하여 시도를 못해본 방법들이 많았으나 나중에 기회된다면 한번 사용해보고 싶다
- 나는 이번 피부암 분류에서 증강과 전처리 위주로 집중하였다. 클래스의 불균형이므로 이미지의 개수들을 균등하게 맞춰주면 성능이 좋아질 것이란 생각과 달리 부족한 일부 라벨을 증강한 경우보다 성능이 안나오는 걸 보며 반복적인 성능확인을 통해 적잘한 증강범위를 찾아야함을 배웠다. 이미지 전처리도 직접 구현보단 검색해서 가져온 코드를 사용했지만 직접 이미지를 찍어보면서 적정 범위도 찾아보았다. 하지만 이런 기능들을 직접 구현할 줄 알아야 진짜 이미지를 다룬다고 할 수 있기에, opencv관련된 책이나 강의등을 통해 공부를 진행할 예정이다. 




   






