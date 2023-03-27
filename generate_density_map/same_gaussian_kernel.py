import numpy as np
from scipy.io import loadmat


def generate_density_map_with_fixed_kernel(img,points,kernel_size=15,sigma=4.0):
    '''
    img: input image.
    points: annotated pedestrian's position like [row,col]
    kernel_size: the fixed size of gaussian kernel, must be odd number.
    sigma: the sigma of gaussian kernel.

    return:
    d_map: density-map we want
    '''
    def guassian_kernel(size,sigma):
        rows=size[0] # mind that size must be odd number.
        cols=size[1]
        mean_x=int((rows-1)/2)
        mean_y=int((cols-1)/2)

        f=np.zeros(size)
        for x in range(0,rows):
            for y in range(0,cols):
                mean_x2=(x-mean_x)*(x-mean_x)
                mean_y2=(y-mean_y)*(y-mean_y)
                f[x,y]=(1.0/(2.0*np.pi*sigma*sigma))*np.exp((mean_x2+mean_y2)/(-2.0*sigma*sigma))
        return f

    [rows,cols]=[img.shape[0],img.shape[1]]
    d_map=np.zeros([rows,cols])
    f=guassian_kernel([kernel_size,kernel_size],sigma) # generate gaussian kernel with fixed size.
    normed_f=(1.0/f.sum())*f # normalization for each head.

    if len(points)==0:
        return d_map
    else:
        for p in points:
            r,c=int(p[0]),int(p[1])
            if r>=rows or c>=cols:
                continue
            for x in range(0,f.shape[0]):
                for y in range(0,f.shape[1]):
                    if x+((r+1)-int((f.shape[0]-1)/2))<0 or x+((r+1)-int((f.shape[0]-1)/2))>rows-1 \
                    or y+((c+1)-int((f.shape[1]-1)/2))<0 or y+((c+1)-int((f.shape[1]-1)/2))>cols-1:
                        continue
                    else:
                        d_map[x+((r+1)-int((f.shape[0]-1)/2)),y+((c+1)-int((f.shape[1]-1)/2))]+=normed_f[x,y]
    return d_map


# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    import matplotlib.pyplot as plt 
    img_path="D:\workspace\ShanghaiTech_dataset\part_A_final/test_data\images\IMG_8.jpg"
    img=plt.imread(img_path)
    plt.imshow(img)
    mat = loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    pts = mat["image_info"][0,0][0,0][0] #1546person*2(col,row)
    points=[]
    for p in pts:
        points.append([p[1],p[0]]) #convert (col,row) to (row,col)
    density_map=generate_density_map_with_fixed_kernel(img,points)
    plt.figure()
    from matplotlib import cm as CM
    plt.imshow(density_map,cmap=CM.jet)








