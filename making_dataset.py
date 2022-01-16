import dask.dataframe as dd
import dask
import numpy as np
from dask import delayed
import dask_image.imread
import dask_image.ndfilters
import dask_image.ndmeasure
import cv2

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

print(1)
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
