import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import tensorrt as trt
import torch
import numpy as np
import cv2
import torchvision
import contextlib
import time
from collections import OrderedDict, namedtuple
import torch.nn as nn

from utils.general import non_max_suppression,scale_boxes,Profile
from utils.augmentations import letterbox
import platform
import logging

class DetectMultiBackend():
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5n.engine', device=torch.device('cuda:0')):
        if not os.path.exists(weights): raise FileNotFoundError(weights)  # check file
        w = str(weights[0] if isinstance(weights, list) else weights)
        engine = True
        self.fp16 = engine  # FP16
        self.device = device
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        engine = weights.endswith('.engine')  # TensorRT
        if engine:  # TensorRT
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            self.fp16 = False  # default updated below
            self.dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                if name.isdecimal(): continue  # skip numbered tensorrt inputs
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        d = model.get_binding_shape(i)
                        self.dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        self.fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            self.batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')
        self.__dict__.update(locals())  # assign all variables to self

    def __call__(self, im):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.__call__(im)  # warmup
                


class YOLOv5TRT:
    def __init__(self, engine_path):
        self.fp16 = True
        self.device = torch.device('cuda:0')
        self.model = DetectMultiBackend(engine_path, device=self.device)
        self.model.warmup()

    def __call__(self, im):
        dt = (Profile(), Profile(), Profile())  # data, infer, nms
        im0 = im.clone() if isinstance(im, torch.Tensor) else im.copy() 
        with dt[0]:
            im = letterbox(im, 640, stride=32, auto=False)[0]  # padded resize
            # cv2.imshow("im", im)
            # cv2.waitKey(500)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im) 
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im).to(self.device)
                # im = im.permute(2, 0, 1)  # uint8 to fp16/32
                im = im.half() if self.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            print(f"预处理时间： {dt[0].t:.4f} seconds")

        # Inference
        with dt[1]:
            pred = self.model(im)
            print(f"推理时间 {dt[1].t:.4f} seconds")

        # NMS
        with dt[2]: 
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=300)
            for i, det in enumerate(pred):
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            print(f"后处理时间：{dt[2].t:.4f} seconds")
        return pred

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    
    def warmup(self, imgsz=(1, 3, 640, 640)):
        im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        self.forward(im)  # warmup
    
if __name__ == '__main__':
    engine_path = 'yolov5n.engine'
    yolov5 = YOLOv5TRT(engine_path)
    image_path = r'D:/project/FixedPointPosition/imgs_old/2023-05-12-18-03-04_ori.jpg'
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (640, 640))
    for i  in range(10):
        t = time.time()
        pred = yolov5(image)
        print("time: ", time.time() - t)
    for det in pred:
        for i in det:
            cv2.putText(image, str(i[5].cpu().numpy()), (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), 2)
    # print("pred: ", pred)
    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    cv2.imshow("image", image)
    cv2.waitKey(0)
        
    