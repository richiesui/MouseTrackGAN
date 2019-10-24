import numpy as np

with open('E:\Workspace\GANproject\data_origin.txt', 'r') as f:
    data = f.readlines()

fdata0 = np.zeros([3000, 300, 3], dtype=float)
fdata1 = np.zeros([3000, 4], dtype=float)
fdata_mark = np.zeros([3000, ], dtype=float)

# 提取特征
for i in range(len(data)):
    data[i] = data[i].split()
    data0 = data[i][1].split(';')[0:-1]
    for j in range(len(data0)):
        data0[j] = data0[j].split(',')
    data0 = np.array(data0).astype(int)
    data0 = np.pad(data0, [[0, 300-len(data0)], [0, 0]], 'constant')
    fdata0[i]=data0
    fdata1[i][0:2]=data0[0,0:2]
    fdata1[i][2:4] = data[i][2].split(',')
    fdata_mark[i] = data[i][3]

#数据扩充
times=12
angle=2*np.pi/times
edata0=np.zeros([3000*times,300,3])
edata1=np.zeros([3000*times,4])
for i in range(times):
    edata0[3000*i:3000*(i+1),:,0]=np.cos(angle*i)*fdata0[:,:,0]-np.sin(angle*i)*fdata0[:,:,1]
    edata0[3000*i:3000*(i+1),:,1]=np.cos(angle*i)*fdata0[:,:,1]+np.sin(angle*i)*fdata0[:,:,0]
    edata0[3000*i:3000*(i+1),:,2]=fdata0[:,:,2]
    edata1[3000*i:3000*(i+1),[0,2]]=np.cos(angle*i)*fdata1[:,[0,2]]-np.sin(angle*i)*fdata1[:,[1,3]]
    edata1[3000*i:3000*(i+1),[1,3]]=np.cos(angle*i)*fdata1[:,[1,3]]+np.sin(angle*i)*fdata1[:,[0,2]]
edata_mark=np.tile(fdata_mark,times)

# 用随机数生成负样本
negNumber = 2000
rand_data0 = np.cumsum(np.random.normal(10, 50, [negNumber, 300,3]),axis=1)
rand_data1 = np.random.uniform(-2000, 2000, [negNumber, 4])
edata0=np.concatenate((edata0,rand_data0),axis=0)
edata1=np.concatenate((edata1,rand_data1),axis=0)
edata_mark=np.concatenate((edata_mark,np.zeros([negNumber,])))

# 保存特征
x_fdata=np.concatenate((fdata0.reshape([3000,300*3],order='F'),fdata1),axis=1)
x_edata=np.concatenate((edata0.reshape([3000*times+negNumber,300*3],order='F'),edata1),axis=1)

#np.save('x_fdata.npy',x_fdata)
#np.save('y_fdata.npy',fdata_mark)
np.save('x_edata.npy',x_edata)
np.save('y_edata.npy',edata_mark)