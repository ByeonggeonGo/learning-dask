import dask.dataframe as dd
import dask
import numpy as np
from dask import delayed
import dask_image.imread
import dask_image.ndfilters
import dask_image.ndmeasure
import cv2
from glob import glob
import json


from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster)
# print(cluster,client)
# client.close()


@delayed
def padding(data, array_len, col_len):
    pad = np.zeros((array_len, col_len))
    length = min(array_len, len(data))
    pad[:length] = data[:length]
    return pad

@delayed
def img_resize(img):
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)/255
    return img


csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']


#making_csv_set
row_csv = dd.read_csv('sampled_data/*/*.csv',dtype={'외부 누적일사 평균': 'object','내부 이슬점 최고': 'object',
       '내부 이슬점 최저': 'object',
       '내부 이슬점 평균': 'object'})
partitions = row_csv.to_delayed()
# datas = [padding(part.drop_duplicates(subset=['측정시각'])[csv_features].values,290,9) for part in partitions]
datas = [padding(part.drop_duplicates(subset=['측정시각'])[csv_features][part != "-"].dropna().values,290,9) for part in partitions]
processed_csv = np.array(dask.compute(*datas))

print(processed_csv.shape)


#making_img_set
row_img = dask_image.imread.imread('sampled_data/*/*.jpg')
datas = [img_resize(part) for part in row_img]
processed_img = np.array(dask.compute(*datas))
print(processed_img.shape)

#making_label_set
# 
crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
risk = {'1':'초기','2':'중기','3':'말기'}

label_description = {}
for key, value in disease.items():
    label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    for disease_code in value:
        for risk_code in risk:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'

global label_encoder
label_encoder = {key:idx for idx, key in enumerate(label_description)}
label_decoder = {val:key for key, val in label_encoder.items()}




@delayed
def label_encoding(label):
    global label_encoder
    encoded_label = label_encoder[label]
    return encoded_label

def getlable(jsonpath):
    with open(jsonpath, 'r') as f:
        json_file = json.load(f)

    crop = json_file['annotations']['crop']
    disease = json_file['annotations']['disease']
    risk = json_file['annotations']['risk']
    label = f'{crop}_{disease}_{risk}'
    return label
    
def get_label_list(labelpath_list):
    labelarr = np.array([])
    for ind,json_path in enumerate(labelpath_list):
        label = label_encoding(getlable(json_path))
        labelarr = np.append(labelarr,label)
    return labelarr

label_list = sorted(glob('sampled_data/*/*.json'))
y_train = get_label_list(label_list)
results = dask.compute(*y_train)
label = np.array(results)
print(label.shape)


client.close()
