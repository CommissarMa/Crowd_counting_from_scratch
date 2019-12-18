# Fudan-ShanghaiTech Dataset (FDST)

## Introduce 
**FDST** collected 100 videos captured from 13 different scenes, totally contains 150,000 frames, with a total of 394,081 annotated heads. In particular,the training set of FDST dataset consists of 60 videos, 9000 frames and the testing set contains the remaining 40 videos, 6000 frames.  

## Download
+ Original download link in BaiduDisk: [[password:sgt1]](https://pan.baidu.com/s/1NNaJ1vtsxCPJUjDNhZ1sHA)  
+ Original download link in GoogleDrive: [[link]](https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w?usp=sharing)  
+ Since the original author didn't compress the files in the dataset, we compress the dataset and provide a new download link in BaiduDisk: [[password:5qxs]](https://pan.baidu.com/s/10lnJYGnHVEk-u-lJNy5b-Q)

## Density map prepare code
+ We first use the fixed gaussian kernel to generate ground truth density map. [[code]](fdst_densitymap_prepare.py)

## Dataloader code
+ Code for single image crowd counting: [[code]]()
+ Code for video crowd counting: [[code]]()

## Reference
More information can be viewed in original author webpage: [[link]](https://github.com/sweetyy83/Lstn_fdst_dataset)