# coding:utf8
import warnings


class DefaultConfig(object):
    model = 'AlexNet'#模型的引入
    env = "'default"
    # load_model_path = None#当load_model_path为None的时候就是重新训练模型。
    load_model_path = 'save/1220Save.pth'
    use_gpu = False   #使用GPU
    num_workers = 2  # 加载数据时使用的线程数目
    print_freq = 10  # 打印频率

    train_image_path = "data/train/train-images.gz"
    train_label_path = "data/train/train-labels.gz"
    test_image_path = "data/test/test.gz"
    test_label_path = "data/test/test-lab.gz"
    predict_image_path="data/predict"


    image_size = 28
    num_channels = 1
    pixel_depth = 255
    train_image_nums = 60000
    test_image_nums = 5000

    seed = 42
    batch_size = 64
    max_epoch = 2
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    debug_file = 'tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # self.device = t.device('cuda') if self.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))


opt = DefaultConfig()
