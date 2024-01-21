# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch
import torch.nn as nn

#######一.选择sklearn数据集###############
s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
s_curve = s_curve[:, [0, 2]] / 10.0

print("shape of moons:", np.shape(s_curve))  # (10000,2)shape一万个点二维
# print(s_curve)
data = s_curve.T  # 转为(2,10000)维
# print(data)
fig, ax = plt.subplots()
ax.scatter(*data, color='red', edgecolor='white')  # *data为解包，将元组或列表解包

ax.axis('off')

dataset = torch.Tensor(s_curve).float()  # 列表写成一个张量，作为dataset

plt.show()

################二.确定超参数的值#####################
num_steps = 100  # 对于步骤数，由beta、分布的均值和标准差共同确定

# 制定每一步的beta
betas = torch.linspace(-6, 6, num_steps)  # beta(1,100)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5  # beta从小到大变化 sigmoid缩放到（0，1）然后继续缩放到较小值

# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1 - betas  # alpha(1,100)
# print(alphas)
alphas_prod = torch.cumprod(alphas, 0)  # 连乘 alphas_prod(1,100)
print(alphas_prod)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)  # alpha_prod t-1
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
       alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
       == one_minus_alphas_bar_sqrt.shape
print("all the same shapr:", betas.shape)


################三.确定扩散过程任意时刻的采样值###################
# 计算任意时刻的x的采样值， 基于x_0和参数重整化技巧
def q_x(x_0, t):
    '''扩散过程仅与x_0与t有关'''

    noise = torch.randn_like(x_0)  # noise是由正态分布中生成的随机噪声，维度与clean picture x_0一致
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    '''方括号表示索引'''
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声 这里求解出的是t时刻的图片x_t


#################四.演示原始数据分布加噪100步后的效果####################
num_shows = 20
fig, axs = plt.subplots(2, 10, figsize=(28, 3))
plt.rc('text', color='blue')
# 共有10000个点， 每个点包含两个坐标
# 生成100步以内每隔5步加噪声后的图像
for i in range(num_shows):
    j = i // 10
    k = i % 10
    q_i = q_x(dataset, torch.tensor([i * num_steps // num_shows]))  # 生成t时刻的采样数据
    axs[j, k].scatter(q_i[:, 0], q_i[:, 1], color='red', edgecolor='white')

    axs[j, k].set_axis_off()
    axs[j, k].set_title('$q(\mathbf{x}_{' + str(i * num_steps // num_shows) + '})$', color='blue')
plt.show()


##############编写拟合逆扩散过程高斯分布地模型##############
##采用MLP网络#### 多层感知机，可用于分类回归等，是复杂网络的基础。
class MLPDiffusion(nn.Module):

    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 2)
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):

        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x


######六.编写训练时的误差函数################
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    '''对任一时刻t进行采样计算loss'''
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机的时刻t，覆盖到更多不同的t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch_size]
    t = t.unsqueeze(-1)  # [batch_size, 1]

    # x0的系数
    a = alphas_bar_sqrt[t]

    # eps的系数
    am1 = one_minus_alphas_bar_sqrt[t]

    # 生成随机噪音eps
    e = torch.randn_like(x_0)

    # 构造模型的输入
    x = x_0 * a + e * am1

    # 送入模型， 得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))
    '''这里传入model不太懂'''
    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()


###########编写逆扩散采样函数（inference过程）##########
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    '''从x[T]恢复x[T-1]、x[T-2]、。。。、x[0]'''

    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    # '''从x[T]采样t时刻的重构值'''

    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)

    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z
    return (sample)


##############开始训练模型，并打印loss及中间的重构效果#
seed = 1234


class EMA():
    '''构建一个参数平滑器'''

    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


print('Training model...')
batch_size = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 4000
plt.rc('text', color='blue')

model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(num_epoch):
    for idx, batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    if (t % 100 == 0):
        print(loss)
        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)  # 共有100个元素

        fig, axs = plt.subplots(1, 10, figsize=(28, 3))
        for i in range(1, 11):
            cur_x = x_seq[i * 10].detach()
            axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white');
            axs[i - 1].set_axis_off()
            axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')
        plt.show()
