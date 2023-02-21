#coding=utf-8
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable

seed_value = 38
import torch
torch.manual_seed(seed_value)
import numpy as np
np.random.seed(seed_value)

batch_size=128
num_epoch=1000
z_dimension=100


INUNITS = 16     # coverage data input  change !!!!!!!!!!!!
TESTNUM_TOTAL = 6    # test case number  change!!!!!!!!!!!!!

f1 = open('Coverage_Info/covMatrix.txt','r')
f2 = open('Coverage_Info/error.txt','r')
f3 = open('Coverage_Info/covMatrix_new.txt','w')

first_ele = True
for data in f1.readlines():
    data = data.strip('\n')
    nums = data.split()
    if first_ele:
        nums = [float(x) for x in nums]
        matrix_x = np.array(nums)
        first_ele = False
    else:
        nums = [float(x) for x in nums]
        matrix_x = np.c_[matrix_x,nums]
f1.close()

first_ele = True
for data in f2.readlines():
    data = data.strip('\n')
    nums = data.split()
    if first_ele:
        nums = [float(x) for x in nums]
        matrix_y = np.array(nums)
        first_ele = False
    else:
        nums = [float(x) for x in nums]
        matrix_y = np.c_[matrix_y,nums]
f2.close()


matrix_x = matrix_x.transpose()
matrix_y = matrix_y.transpose()

inputs_pre = []
testcase_fail_num = 0
for testcase_num in range(len(matrix_y)):
    if matrix_y[testcase_num][0] == 1:
        inputs_pre.append(matrix_x[testcase_num])
        testcase_fail_num = testcase_fail_num + 1
TESTNUM = testcase_fail_num
inputs = torch.FloatTensor(inputs_pre)
labels = torch.FloatTensor(matrix_y)

minimum_suspicious_set = Variable(torch.zeros(INUNITS))
for element_num in range(len(inputs[0])):
    flag = 0
    for item in inputs:
        if item[element_num] == 1:
            flag = flag +1
    if flag == testcase_fail_num:
        minimum_suspicious_set[element_num] = 1
#定义判别器  #####Discriminator######使用多层网络来作为判别器
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(INUNITS,256),#输入特征数为784，输出为256
            nn.LeakyReLU(0.2),#进行非线性映射
            nn.Linear(256,256),#进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()#也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )
    def forward(self, x):
        x=self.dis(x)
        return x
####### 定义生成器 Generator #####
class generator(nn.Module):
    def __init__(self):
        super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(100,256),#用线性变换将输入映射到256维
            nn.ReLU(True),#relu激活
            nn.Linear(256,256),#线性变换
            nn.ReLU(True),#relu激活
            nn.Linear(256,INUNITS),#线性变换
            nn.Tanh()#Tanh激活使得生成数据分布在【-1,1】之间
        )
    def forward(self, x):
        x=self.gen(x)
        return x
#创建对象
D=discriminator()
G=generator()
if torch.cuda.is_available():
    D=D.cuda()
    G=G.cuda()
#########判别器训练train#####################
criterion = nn.BCELoss() #是单目标二分类交叉熵函数
d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(G.parameters(),lr=0.0003)
###########################进入训练##判别器的判断过程#####################
for epoch in range(num_epoch): #进行多个epoch的训练
#    for i in range(len(inputs)):
        real_testcase = inputs.cuda()
      #  real_label = labels.cuda()
        real_label = Variable(torch.ones(TESTNUM)).cuda()
        fake_label = Variable(torch.zeros(TESTNUM)).cuda()
      #  fake_label = [0.,1.,1.,1.,1.,0.]
        real_label_new = []
        for item in real_label:
            temp=[]
            temp.append(item)
            real_label_new.append(temp)
        real_label_new = torch.tensor(real_label_new).cuda()
        real_label = real_label_new
        fake_label_new = []
        for item in fake_label:
            temp=[]
            temp.append(item)
            fake_label_new.append(temp)
        fake_label_new = torch.tensor(fake_label_new).cuda()
        
        real_out = D(real_testcase)  # 将真实测试用例放入判别器中
        d_loss_real = criterion(real_out, real_label)# 得到真实测试用例的loss
        real_scores = real_out  # 得到真实测试用例的判别值，输出的值越接近1越好
        
        z = Variable(torch.randn(TESTNUM, z_dimension)).cuda()  # 随机生成一些噪声
        fake_testcase = G(z)  # 随机噪声放入生成网络中，生成一张假的测试用例
        fake_out = D(fake_testcase)  # 判别器判断假的测试用例
        d_loss_fake = criterion(fake_out, fake_label_new)  # 得到假的测试用例的loss
        fake_scores = fake_out  # 得到假测试用例的判别值，对于判别器来说，假测试用例的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake #损失包括判真损失和判假损失
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # ==================训练生成器============================
        ################################生成网络的训练###############################
        # 原理：目的是希望生成的假的测试用例被判别器判断为真的图片，
        # 在此过程中，将判别器固定，将假的测试用例传入判别器的结果与真实的label对应，
        # 反向传播更新的参数是生成网络里面的参数，
        # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的测试用例让判别器以为是真的
        # 这样就达到了对抗的目的
        # 计算假的测试用例的损失
        z = Variable(torch.randn(TESTNUM, z_dimension)).cuda()  # 得到随机噪声
        fake_testcase = G(z) #随机噪声输入到生成器中，得到一副假的测试用例
        output = D(fake_testcase)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的测试用例片与真实的测试用例的label的loss
        # bp and optimize
        g_optimizer.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f}'.format(
                epoch,num_epoch,d_loss.data.item(),g_loss.data.item(),
                real_scores.data.mean(),fake_scores.data.mean()  #打印的是真实测试用例的损失均值
            ))

z = Variable(torch.randn((TESTNUM_TOTAL-TESTNUM*2), z_dimension)).cuda()  # 得到随机噪声
fake_testcase = G(z) #随机噪声输入到生成器中，得到一副假的测试用例

fake_testcase = fake_testcase.cpu().detach()
fake_testcase_numpy = fake_testcase.numpy()

for item in fake_testcase_numpy:
    for element_num in range(len(item)):
        if item[element_num] <= 0.5 and minimum_suspicious_set[element_num] == 0:
            f3.write('0')
        else:
#            f3.write('1')
            f3.write(str(round(item[element_num],2)))
        f3.write(' ')
    f3.write('\n')
print (fake_testcase_numpy)
f3.close()
#保存模型
torch.save(G.state_dict(),'./generator.pth')
torch.save(D.state_dict(),'./discriminator.pth')
