# role

你是一个顶级程序员

# context

目录 /home/chenxu/codes/ichibanFATE/client/FMNIST/datasets 下面是用来做联邦学习实验的FMNIST数据集。

# action

请先压缩 /home/chenxu/codes/ichibanFATE/client/FMNIST/datasets/fashionmnist.zip 文件，获取所有数据集文件。如果该压缩包里还有压缩包就一直解压。

下面开始只关注训练集。

生成三个python代码在刚才的目录。python文件你自己取名，但名字里需要保留30、60、90三组值。然后这三个代码是把原始 **训练集** 拆分为30、60、90三组不同数目的客户端数据（就是分别拆分30 60 90份）。

python文件运行前要先检查是否有拆分过的文件（比如拆分60份后跑第二次实验，拆分90份的代码需要发现有拆过60份，需要先全删掉）

你可以选择拆分成30、60、90份文件。也可以选择保留为一个文件，但要明确划分好每个客户端需要取的数据区间。但如果你想拆分成多组文件。请注意，只能是datasets/node_0 - node_60 这种。（node_0等可以是一层目录，下面直接是裸的数据文件，node_0等也可以直接是裸的数据文件）




# role


你是一个顶级程序员

# context

现在你的论文需要增加一些普适性数据集。比如 FMNIST（在/home/chenxu/codes/ichibanFATE/client/FMNIST目录下）。

请你参照同级目录下的MHAD USC HatefulMemes CrisisMMD 四组数据集的处理方法。这几个是合力完成实验的代码。

此外，/home/chenxu/codes/ichibanFATE/client/FMNIST/fminst.ipynb 是一个已经保证可用的训练代码。

# action

请在 /home/chenxu/codes/ichibanFATE/client/FMNIST 目录下完成数据加载、模型训练、保存等完整的实验功能。具体功能在上面的几个参考实例里。

然后 FMNIST的模型、参数等一切都参照fmnist.ipynb就行。保证这个是能用的。

FMNIST 下面你要生成的所有代码都是模拟的一个客户端的模型训练操作。也就是client.py给 FMNIST 发送一个client_id，FMNIST 代码需要读取的数据集编号为 client_id 的编号。

此外，FMNIST 目录下面需要暴露给外部组件的信息请参考 /home/chenxu/codes/ichibanFATE/client/client.py 中其他目录在client.py中的使用方法。

对于 FMNIST 数据的加载方法，请查看你上面生成的拆分方法决定。

# requirements

1. FMNIST 数据集或者任务的名字就叫 FMNIST

2. 你的更改权限仅限于 FMNIST 目录；以及在 /home/chenxu/codes/ichibanFATE/client/client.json 中完善新的任务信息：即增加一条        {
            "dataset_name": "USC",
            "modalities_num": 2,
            "modalities_name" : [
                "acc",
                "gyr"
            ],
            "model_size" : 1201624
        } 目录


3. 所有路径均只能使用相对路径，方便我在多机间进行迁移。

4. 你所有的权限均同刚才生成 MNIST 实验目录一样，可以不用再问了。








# role


你是一个顶级程序员

# context

现在你的论文需要增加一些普适性数据集。比如 FMNIST（在/home/chenxu/codes/ichibanFATE/client/FMNIST目录下）。

请你参照同级目录下的MHAD USC HatefulMemes CrisisMMD 四组数据集的处理方法。这几个是合力完成实验的代码。

此外，/home/chenxu/codes/ichibanFATE/client/FMNIST/fminst.ipynb 是一个已经保证可用的训练代码。

# action

请在 /home/chenxu/codes/ichibanFATE/client/FMNIST 目录下完成数据加载、模型训练、保存等完整的实验功能。具体功能在上面的几个参考实例里。

然后 FMNIST的模型、参数等一切都参照fmnist.ipynb就行。保证这个是能用的。

FMNIST 下面你要生成的所有代码都是模拟的一个客户端的模型训练操作。也就是client.py给 FMNIST 发送一个client_id，FMNIST 代码需要读取的数据集编号为 client_id 的编号。

此外，FMNIST 目录下面需要暴露给外部组件的信息请参考 /home/chenxu/codes/ichibanFATE/client/client.py 中其他目录在client.py中的使用方法。

对于 FMNIST 数据的加载方法，请查看你上面生成的拆分方法决定。

# requirements

1. FMNIST 数据集或者任务的名字就叫 FMNIST

2. 你的更改权限仅限于 FMNIST 目录；以及在 /home/chenxu/codes/ichibanFATE/client/client.json 中完善新的任务信息：即增加一条        {
            "dataset_name": "USC",
            "modalities_num": 2,
            "modalities_name" : [
                "acc",
                "gyr"
            ],
            "model_size" : 1201624
        } 目录


3. 所有路径均只能使用相对路径，方便我在多机间进行迁移。

4. 你所有的权限均同刚才生成 MNIST 实验目录一样，可以不用再问了。









# role


你是一个顶级程序员

# context

现在你的论文需要增加一些普适性数据集。比如 FMNIST（在/home/chenxu/codes/ichibanFATE/client/FMNIST目录下）。

请你参照同级目录下的MHAD USC HatefulMemes CrisisMMD 四组数据集的处理方法。这几个是合力完成实验的代码。

此外，/home/chenxu/codes/ichibanFATE/client/FMNIST/fminst.ipynb 是一个已经保证可用的训练代码。

# action

请在 /home/chenxu/codes/ichibanFATE/client/FMNIST 目录下完成数据加载、模型训练、保存等完整的实验功能。具体功能在上面的几个参考实例里。

然后 FMNIST的模型、参数等一切都参照fmnist.ipynb就行。保证这个是能用的。

FMNIST 下面你要生成的所有代码都是模拟的一个客户端的模型训练操作。也就是client.py给 FMNIST 发送一个client_id，FMNIST 代码需要读取的数据集编号为 client_id 的编号。

此外，FMNIST 目录下面需要暴露给外部组件的信息请参考 /home/chenxu/codes/ichibanFATE/client/client.py 中其他目录在client.py中的使用方法。

对于 FMNIST 数据的加载方法，请查看你上面生成的拆分方法决定。

# requirements

1. FMNIST 数据集或者任务的名字就叫 FMNIST

2. 你的更改权限仅限于 FMNIST 目录；以及在 /home/chenxu/codes/ichibanFATE/client/client.json 中完善新的任务信息：即增加一条        {
            "dataset_name": "USC",
            "modalities_num": 2,
            "modalities_name" : [
                "acc",
                "gyr"
            ],
            "model_size" : 1201624
        } 目录


3. 所有路径均只能使用相对路径，方便我在多机间进行迁移。

4. 你所有的权限均同刚才生成 MNIST 实验目录一样，可以不用再问了。










# role

你是一个顶级程序员

# context

目录 /home/chenxu/codes/ichibanFATE/client/CIFAR/datasets 下面是用来做联邦学习实验的CIFAR 数据集。

# action

请先压缩 /home/chenxu/codes/ichibanFATE/client/CIFAR/cifar10-preprocessed.zip 文件，获取所有数据集文件。如果该压缩包里还有压缩包就一直解压。

下面开始只关注训练集。

生成三个python代码在刚才的目录。python文件你自己取名，但名字里需要保留30、60、90三组值。然后这三个代码是把原始 **训练集** 拆分为30、60、90三组不同数目的客户端数据（就是分别拆分30 60 90份）。

python文件运行前要先检查是否有拆分过的文件（比如拆分60份后跑第二次实验，拆分90份的代码需要发现有拆过60份，需要先全删掉）

你可以选择拆分成30、60、90份文件。也可以选择保留为一个文件，但要明确划分好每个客户端需要取的数据区间。但如果你想拆分成多组文件。请注意，只能是datasets/node_0 - node_60 这种。（node_0等可以是一层目录，下面直接是裸的数据文件，node_0等也可以直接是裸的数据文件）

# role

你是一个顶级程序员

# context

现在你的论文需要增加一些普适性数据集。比如 CIARF（在/home/chenxu/codes/ichibanFATE/client/CIARF目录下）。

请你参照同级目录下的MHAD USC HatefulMemes CrisisMMD 四组数据集的处理方法。这几个是合力完成实验的代码。

此外，/home/chenxu/codes/ichibanFATE/client/CIFAR/resnet-implementation-from-scratch-on-cifar10.ipynb 是一个已经保证可用的训练代码。

# action

请在 /home/chenxu/codes/ichibanFATE/client/CIFAR 目录下完成数据加载、模型训练、保存等完整的实验功能。具体功能在上面的几个参考实例里。

然后 CIFAR 的模型、参数等一切都参照resnet-implementation-from-scratch-on-cifar10.ipynb就行。保证这个是能用的。

CIFAR 下面你要生成的所有代码都是模拟的一个客户端的模型训练操作。也就是client.py给 CIFAR 发送一个client_id，CIFAR 代码需要读取的数据集编号为 client_id 的编号。

此外，CIFAR 目录下面需要暴露给外部组件的信息请参考 /home/chenxu/codes/ichibanFATE/client/client.py 中其他目录在client.py中的使用方法。

对于 CIFAR 数据的加载方法，请查看你上面生成的拆分方法决定。

# requirements

1. CIFAR 数据集或者任务的名字就叫 CIFAR

2. 你的更改权限仅限于 CIFAR 目录；以及在 /home/chenxu/codes/ichibanFATE/client/client.json 中完善新的任务信息：即增加一条        {
            "dataset_name": "USC",
            "modalities_num": 2,
            "modalities_name" : [
                "acc",
                "gyr"
            ],
            "model_size" : 1201624
        } 目录


3. 所有路径均只能使用相对路径，方便我在多机间进行迁移。

4. 你所有的权限均同刚才生成 MNIST FMNIST 实验目录一样，可以不用再问了。








# role

你是一个顶级程序员

# context

现在目录 /home/chenxu/codes/ichibanFATE/client/CIFAR /home/chenxu/codes/ichibanFATE/client/MNIST 和 /home/chenxu/codes/ichibanFATE/client/FMNIST 下面划分客户端数据的代码都是iid的分布。我希望划分成狄利克雷分布的Non-iid标签分布。

# action

请再给出 3 个 用于划分non-iid客户端分布数据的代码，其中狄利克雷分布的参数需要便于更改。可以先设置为0.5

# requirement

所有路径均只能使用相对路径，方便我在多机间进行迁移。





# role

你是一个顶级程序员

# context

现在我的 /home/chenxu/codes/ichibanFATE/client/client.py 和 /home/chenxu/codes/ichibanFATE/client/communication.py 还都是CrisisMMD等 4 个任务的实验代码。
同理，server目录也是CrisisMMD等 4 个任务的实验代码。

而我需要在 MNIST FMNIST 和 CIFAR 三个任务环境下完成不同客户端数量的收敛测试。客户端数量分别包括 30 60 90。

# action

请帮我修改 /home/chenxu/codes/ichibanFATE/client/ 与 /home/chenxu/codes/ichibanFATE/server/ 两个目录下的代码。然后我需要用来在 MNIST 等3个任务上完成实验。

# requirements

1. 现在的代码保证是可以用于当前的4任务实验环境的，你需要更改的仅仅是不同的任务搭配。

2. 现在的代码可能有写死30个客户端的地方。请改成柔性方式，便于随时更改客户端数量进行实验（这个便于指的是运行原先的命令就行）

3. 原先的实验环境下，server程序的运行命令是: cd/home/chenxu/codes/ichibanFATE/server; python server.py。client的运行命令在 /home/chenxu/codes/ichibanFATE/client/start_0_29.sh里


4. 所有路径均只能使用相对路径，方便我在多机间进行迁移。

5. 在这组实验里，client.json 中的 "ability" 字段也是需要以 1200000000 为均质进行异质性的，就随机生成就行。有多少客户端就生成多少个。最好是一次性生成90个，然后假如说是30个客户端就取前三十个，60个就取前60个这样子。其他字段也同样处理，但我实在是忘了均值和方差是多少了。但论文选段在这里：
      The distance between each client $i$ and the BS follows a uniform distribution $d_i \sim \mathsf{Uniform}(6, 10)$ meters. The wireless transmission power of the clients is set between $22$ and $26$ dBm. The communication bandwidth shared between the BS and the clients is $40$ MHz, with a Gaussian noise power of approximately $-101$ dBm. We adopt a long-distance path loss model to characterize the channel gain for each client. Specifically, the path loss is given by $\mathrm{PL}(d_i) = 40+30\log_{10} d_i + \varrho$ (in dB), where $\varrho \sim \mathcal{N}(0, 1^2)$, and the channel gain is calculated as $g_i = 10^{-\mathrm{PL}(d_i)/10}$.
      
      
      We assume the MAC rate of each client $i \in \mathcal{N}$, denoted as $\kappa_i$, is in the range $[800, 1400]$ MMAC/s. The energy consumption for client $i$ to perform a single MAC operation, $\rho_i$, ranges from $100$ to $1000$ pJ/MAC. Additionally, the total energy budget for each client is assumed to be in the range $[18000, 20000]$ J, encompassing both communication and computation requirements.

      如果这里面没提到的，你根据现有的离散数据自行决定该怎么生成吧。

6. 公共代码中冗余的部分可以直接删除，但仅限于公共代码（而非每个实验任务自己内部的代码）。至于 /home/chenxu/codes/ichibanFATE/client/CrisisMMD 等目录下的实验数据和代码可千万别给我删了。

7. 如果需要对任务代码进行处理与改造 参考 /home/chenxu/codes/ichibanFATE/client/CrisisMMD 等 4 个原始实验目录

8. 如果我上面的说法有出现歧义的地方，请及时告诉我，我好更改。但请用中文回答与告知。
