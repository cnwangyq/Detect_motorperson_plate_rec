import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from utils.strLabelConverter import strLabelConverter

class BidirectionalLSTM(nn.Module):
    """
    实际这里是单向 LSTM（保持与你原始实现一致）
    Input:  [T, B, C]
    Output: [T, B, O]
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, bidirectional: bool) -> None:
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=n_in,
            hidden_size=n_hidden,
            bidirectional=bidirectional
        )
        if bidirectional:
            self.embedding = nn.Linear(n_hidden * 2, n_out)
        else:
            self.embedding = nn.Linear(n_hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(x)  # [T, B, H]

        t, b, h = recurrent.shape
        recurrent = recurrent.reshape(t * b, h)
        output = self.embedding(recurrent)

        return output.reshape(t, b, -1)

# class HSwish(nn.Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * F.relu6(x + 3) / 6

class CRNN(nn.Module):
    def __init__(
        self,
        img_h: int,
        nc: int,
        nclass: int,
        nh: int,
        export_shape: int = 80,
        leaky_relu: bool = False
    ) -> None:
        super().__init__()

        assert img_h % 16 == 0, "img_h must be multiple of 16"

        self.export_shape = export_shape

        # 单层
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        # # 双层
        # ks = [3, 3, 3, 3, 3, 3, (1, 2)]
        # ps = [1, 1, 1, 1, 1, 1, (0, 0)]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        self.cnn = self._build_cnn(
            nc=nc,
            ks=ks,
            ps=ps,
            ss=ss,
            nm=nm,
            leaky_relu=leaky_relu
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh, True), # 单层
            # BidirectionalLSTM(1024, nh, nh, True), # 双层
            BidirectionalLSTM(nh, nh, nclass, True)
        )

    def _build_cnn(
        self,
        nc: int,
        ks: list,
        ps: list,
        ss: list,
        nm: list,
        leaky_relu: bool
    ) -> nn.Sequential:

        cnn = nn.Sequential()

        def conv_relu(i: int, batch_norm: bool = False):
            n_in = nc if i == 0 else nm[i - 1]
            n_out = nm[i]

            cnn.add_module(
                f"conv{i}",
                nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i])
            )

            if batch_norm:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(n_out))

            activation = (
                nn.LeakyReLU(0.2, inplace=True)
                if leaky_relu
                else nn.ReLU(inplace=True)
            )

            cnn.add_module(f"relu{i}", activation)

        conv_relu(0)
        cnn.add_module("pooling0", nn.MaxPool2d(2, 2))

        conv_relu(1)
        cnn.add_module("pooling1", nn.MaxPool2d(2, 2))

        conv_relu(2, batch_norm=True)
        conv_relu(3)
        cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        conv_relu(4, batch_norm=True)
        conv_relu(5)
        cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        conv_relu(6, batch_norm=True)

        return cnn

    def init_weights(self, pretrained: str) -> None:
        checkpoint = torch.load(pretrained, map_location="cpu")

        new_state = OrderedDict()

        for key, value in checkpoint.items():
            if key.startswith("module.") and ".rnn" not in key:
                new_state[key[7:]] = value

        self.load_state_dict(new_state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN
        x = self.cnn(x)

        # [B, C, 1, W] -> [B, C, W] 单层
        x = x.squeeze(2)
        # [B, C, W] -> [W, B, C]
        x = x.permute(2, 0, 1)

        # # [B, C, 2, W] -> [W, B, C * 2] 双层拼接为单层
        # b, c, h, w = x.size()
        # x = x.permute(3, 0, 1, 2)  # [W, B, C, H]
        # x = x.reshape(w, b, c * h)  # [W, B, 1024]



        # RNN
        x = self.rnn(x)

        # [T, B, C] -> [B, T, C]
        x = x.permute(1, 0, 2)

        # Export logic
        if self.export_shape == 52:
            pred = x.argmax(dim=2, keepdim=True).float()
            conf = x.softmax(dim=2).max(dim=2, keepdim=True)[0]
            x = torch.cat([pred, conf], dim=1)

        elif self.export_shape == 27:
            pred = x.argmax(dim=2, keepdim=True).float()
            conf = x.softmax(dim=2).max(dim=2, keepdim=True)[0]
            x = torch.cat([pred, conf], dim=1)[:, :27]

        x = x.squeeze(-1)

        return x




if __name__ == '__main__':
    alphabet = '0123456789abcdefghjklmnpqrstuvwxyz京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新挂警学港澳使应急民航机场领电*莆田'  # 车牌号字符集
    # alphabet = '0123456789abcdefghjklmnpqrstuvwxyz京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新挂警学港澳使应急民航机场领电*'  # 车牌号字符集

    model = CRNN(32,1,len(alphabet)+1,256)
    print(model)
    # exit()
    # checkpoint = torch.load("../trained_models/20250423_plateRec.pth", map_location=torch.device('cpu'))
    checkpoint = torch.load("../output/CRNN_2026_3_18_18_45/best_all.pth", map_location=torch.device('cpu'))

    checkpoint = {
        k.replace('module.', ''): v
        for k, v in checkpoint.items()
    }
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)

    print(len(alphabet))
    converter = strLabelConverter(alphabet)

    image = '../sample/莆田52K23_processed.jpg'
    # image = '../sample/川B003GV_国道G324埭里卡口往市区-1_20260305124326_0_92_02.jpg'
    image = Image.open(image)
    image = image.convert('L')
    transformer = transforms.Compose([
        transforms.Resize((32, 100), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transformer(image)
    image = image.view(1, *image.size())
    print(image.shape)

    preds = model(image)
    print(preds.shape)

    batch_size = 1
    preds_size = torch.LongTensor([preds.size(1)] * batch_size)
    preds = preds.permute(1, 0, 2)  # .log_softmax(2)#(T,b,c)
    preds = preds.softmax(dim=2)
    probs, preds = preds.max(2)
    # for p in list(preds.numpy()):
    #     print(int(p))
    #     print(alphabet[int(p)])
    # raw = ''.join([alphabet[int(p)] for p in list(preds.numpy())])
    preds = preds.transpose(1, 0).contiguous().view(-1)  # (b*t)
    probs = probs.transpose(1, 0).contiguous().view(-1)  # (b*t)
    sim_pred, score_lists, scores = converter.decode_with_score(probs.data, preds.data, preds_size.data, raw=False)
    print(sim_pred)
    print(score_lists)
    print(scores)

    # onnx_path = '../trained_models/crnn-27.onnx'
    # sess = onnxruntime.InferenceSession(onnx_path)
    # image = image.numpy().astype(np.float32)
    # preds = sess.run(None, {"input": image})[0]
    # preds = torch.tensor(preds)
    # print(preds.shape)
    #
    # if np.max(preds.shape) == 52:
    #     scores = torch.mean(preds[0, 26:])
    # else:
    #     scores = preds[0, -1]
    # preds = preds[:, :26]
    # preds_size = torch.IntTensor([preds.size(1)])
    # preds = preds.transpose(1, 0).contiguous().view(-1).type(torch.IntTensor)  # (b*t)
    # sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print(sim_pred)