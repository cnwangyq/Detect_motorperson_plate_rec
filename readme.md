## 车牌检测
### 步骤1
1. 导入车辆检测模型至 weights ，支持onnx/pt模型
2. 车辆标签数据导入至 plate_data ，格式为yaml
### 步骤2
1. 导入车牌检测模型至 plate_weights ，支持onnx/pt模型
2. 车牌检测模型的标签数据导入至 plate_data ，格式为yaml
### 步骤3
1. 导入车牌识别模型（OCR模型）至 rec_weights ，支持pt模型
2. 项目使用的OCR模型为 models/crnn_260318
3. PLATE_ALPHABET = "0123456789abcdefghjklmnpqrstuvwxyz京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新挂警学港澳使应急民航机场领电*"
