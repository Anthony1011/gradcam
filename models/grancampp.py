# Implment the GRAN-CAMPP model base on the YOLOv7 model
import time
import torch
import torch.nn.functional as F

def find_yolo_layer(model, layer_name):
    """Find model layer to calculate GradCAM and GradCAM++
    要找到指定層的activation layer(function)後的結果,就是找到輸出activate map
    對activation map做反向傳播,就可以得到該層的梯度
    """
    # hierarchy 把傳入的layer_name用'_'分割，並且把分割後的list存入hierarchy=[0,1]
    # hierarchy = [102,act]
    hierarchy = layer_name.split('_')
    # target_layer使用hierarchy[0]作為索引，找到model.model.model._modules[hierarchy[0]]這一層
    # model.model.model._modules[102] 為在model中第102層
    target_layer = model.model.model._modules[hierarchy[0]]
    # h=從hierarchy[1]開始，hierarchy[0]元素逐個輸入,單因為每次只傳入一層所以也只會有一個元素
    for h in hierarchy[1:]:
        # target_layer找到子模組，並且把子模組存入target_layer
        # target_layer._modules[act]=nn.ReLU(inplace=True)
        target_layer = target_layer._modules[h]
        #target_layer代表的是ReLU activation layer
    return target_layer

class YOLOV7GradCAM:
    '''
    - 对目标层的梯度的每个通道求平均(GAP，全局平均池化操作)
    - 将GAP后的梯度值与目标层的输出值逐点相乘
    - 由于热力图关心的是对分类有正面影响的特征，所以加上`ReLU`函数以移除负值
    '''

    # 初始化所有參數
    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict() # 用于存储每层的梯度
        self.activations = dict() # 用于存储每层的输出
        '''
        backward_hook 函數：在梯度反向傳播過程中被調用，用於記錄目標層的梯度值。
        grad_output 是目標層梯度的值，self.gradients['value'] 用於存儲該梯度值。
        
        forward_hook 函數：在前向傳播過程中被調用，用於記錄目標層的輸出值。
        output 是目標層的輸出值，self.activations['value'] 用於存儲該輸出值。
        '''
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        '''
        target_layer.register_forward_hook(forward_hook)：
        這行代表在 target_layer 的前向傳播時，會自動調用 forward_hook 函數。
        target_layer.register_full_backward_hook(backward_hook)：
        這行代表在 target_layer 的梯度反向傳播時，會自動調用 backward_hook 函數。
        '''
        # 得到指定位置layer(這一層式ReLU activation layer)
        target_layer = find_yolo_layer(self.model, layer_name)
        # .register_forward_hook代表向前傳遞時會呼叫forward_hook函數當作方法
        target_layer.register_forward_hook(forward_hook)
        # .register_full_backward_hook代表向後傳遞時會呼叫backward_hook函數當作方法
        target_layer.register_full_backward_hook(backward_hook)

        # next(self.model.model.parameters())：從模型的參數迭代器中獲取第一個參數
        # .is_cuda：給出Ture of False,如果成立device='cuda'，否則device='cpu'
        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        # torch.zeros(2, 3, 3, 3)創建一個都為零,有兩張每一張3個通道,每個通道3*3的張量
        self.model(torch.zeros(1, 3, *img_size, device=device))
        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
            preds: The object predictions
        """
        saliency_map = []
        b, c, h, w = input_img.size()
        # 得到當前時間 start_time = time.time()
        tic = time.time()
        # model.eval()：將模型設置為評估模式
        # self.model(input_img)：將input_img傳入模型中，並且得到模型的輸出
        preds, logist= self.model(input_img)
        print("[INFO] model-forward took: ", round(time.time() - tic, 4), 'seconds')
        for 




















