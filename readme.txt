bsd.txt文件是乳腺癌的原始数据

MDS.py是用官网给的数据来进行的从3维到2维的维数约简代码

MDS-breast.py是用乳腺癌的原始数据来进行维数约简，中间注释部分是计算降到8维之后原始数据和MDS、ISOmap的欧氏距离的比较（只有前10个）
然后剩下的部分就是显示presentation中约简后的图像，用涛神说的方式，加号表示良性，三角形表示恶性

official_round_data.py是官方的圆形三维数据降到二维的代码

official_S_data.py是官方的S形三维数据降到二维的代码