"""
整合所有参数
"""
import os, sys
sys.path.append(os.path.dirname(__file__))
import torch, yaml

class OPT:
    def __init__(self):
        self.epochs = 50
        self.batch_size = 8
        self.num_classes = 20
        self.data_name = "yolo_data_VOC2012"
        self.save_name = "yolov4-pacsp-s"

        self.img_size = 512
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.file_dir = os.path.dirname(__file__)
        self.cfg = os.path.join(self.file_dir, "model_cfg", self.save_name + ".cfg")
        self.muliti_scale = True
        self.freeze_layer = False

        self.single_cls = False        #! 单类训练
        self.amp = True
        self.rect = True
        self.cache_images = False
        self.no_test = False
        self.save_best = False

        self.data_root = os.path.join(self.file_dir, "..", "data", self.data_name)
        self.save_path = os.path.join(self.file_dir, "save_weights")
        self.img_save_path = os.path.join(self.data_root, "test")
        self.resume = ""
        self.start_epoch = 0
        self.weights = ""

        self.hyp = self.read_yaml(os.path.join(self.file_dir, "model_cfg", "hyp.yaml"))
        self.weight_update()

    def read_yaml(self, yaml_file):
        with open(yaml_file) as f:
            hyp = yaml.load(f, Loader=yaml.FullLoader)
        return hyp

    def weight_update(self):
        nums = []
        #! 遍历模型文件夹
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            nums = []
        else:
            files = os.listdir(self.save_path)
            for file in files:
                if file.endswith(".pth") & file.startswith(self.save_name):
                    nums.append(int(file.split('-')[1].split('.')[0]))

        #! 如果没有训练过，resume和weights都为空
        #! 这里默认resume和weights都使用save_path下最新的模型
        if (len(nums) == 0):
            self.resume = self.weights = ""
            file_name = os.path.join(self.save_path, self.save_name + "--1.pt")
            if os.path.exists(file_name):
                self.resume = file_name
        else:
            #! 取模型编号最大的模型用于继续训练或者预测
            num = sorted(nums, reverse=True)[0]
            #! 启动一次至少训练20个epoch
            if (self.epochs < num+1+20):
                self.epochs = num+1+20
            self.resume = os.path.join(self.save_path, self.save_name + f"-{num}.pth")
            #! 如果resume为空，那么weights就等于resume
            if self.weights == "":
                self.weights = self.resume

opt = OPT()

if __name__ == "__main__":
    print(opt.resume)
