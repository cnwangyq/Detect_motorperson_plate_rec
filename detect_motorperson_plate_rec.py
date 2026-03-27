# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Two-stage detection: detect motorperson first, then detect plate inside motorperson crops, then run OCR."""

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from detect_plate_rec import (
    DEFAULT_PLATE_CLASSES,
    PlateRecognizer,
    resolve_annotator_font,
    resolve_plate_class_ids,
)
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


DEFAULT_MOTORPERSON_CLASSES = ("motorperson",)
DEFAULT_PLATE_DET_CLASSES = DEFAULT_PLATE_CLASSES


def to_int_box(xyxy):
    return [int(v.item()) if isinstance(v, torch.Tensor) else int(v) for v in xyxy]


def expand_box(xyxy, im_shape, gain=1.0, pad=0):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    h, w = im_shape[:2]
    bw = (x2 - x1) * gain + pad
    bh = (y2 - y1) * gain + pad
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nx1 = max(0, min(w, int(round(cx - bw / 2.0))))
    ny1 = max(0, min(h, int(round(cy - bh / 2.0))))
    nx2 = max(0, min(w, int(round(cx + bw / 2.0))))
    ny2 = max(0, min(h, int(round(cy + bh / 2.0))))
    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return [nx1, ny1, nx2, ny2]


def preprocess_array(image, model, imgsz):
    im = letterbox(image, imgsz, stride=model.stride, auto=model.pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    return im


@torch.inference_mode()
def detect_on_array(
    model,
    image,
    imgsz,
    conf_thres,
    iou_thres,
    classes=None,
    agnostic_nms=False,
    max_det=1000,
    augment=False,
):
    if image is None or image.size == 0:
        return torch.empty((0, 6))

    im = preprocess_array(image, model, imgsz)
    pred = model(im, augment=augment, visualize=False)
    det = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()
    return det


@smart_inference_mode()
def run(
    weights=ROOT / "trained_model/car_face_det/v5s_256x256_1b.onnx",
    source=ROOT / "data/images",
    data=ROOT / "nonMoto.yaml",
    imgsz=(256, 256),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
    plate_weights=ROOT / "trained_model/yolov5s-plate-detect.pt",
    plate_data=ROOT / "plate_det.yaml",
    plate_imgsz=(640, 640),
    plate_conf_thres=0.25,
    plate_iou_thres=0.45,
    plate_max_det=20,
    plate_agnostic_nms=False,
    motorperson_classes=DEFAULT_MOTORPERSON_CLASSES,
    plate_classes=DEFAULT_PLATE_DET_CLASSES,
    motorperson_crop_gain=1.02,
    motorperson_crop_pad=10,
    rec_weights="",
    rec_conf_thres=0.0,
    rec_crop_gain=1.02,
    rec_crop_pad=10,
    font_path="",
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    save_img = True

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)

    main_model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    main_stride, main_names, main_pt = main_model.stride, main_model.names, main_model.pt
    imgsz = check_img_size(imgsz, s=main_stride)

    plate_model = DetectMultiBackend(plate_weights, device=device, dnn=dnn, data=plate_data, fp16=half)
    plate_stride, plate_names, plate_pt = plate_model.stride, plate_model.names, plate_model.pt
    plate_imgsz = check_img_size(plate_imgsz, s=plate_stride)

    motorperson_class_ids = resolve_plate_class_ids(main_names, motorperson_classes)
    stage1_classes = classes if classes is not None else sorted(motorperson_class_ids) if motorperson_class_ids else None
    plate_class_ids = resolve_plate_class_ids(plate_names, plate_classes)
    stage2_classes = sorted(plate_class_ids) if plate_class_ids else None

    plate_recognizer = PlateRecognizer(rec_weights, device=device) if rec_weights else None
    annotator_font = resolve_annotator_font(font_path)

    if not motorperson_class_ids:
        LOGGER.warning(f"No motorperson classes matched {motorperson_classes}.")
    if not plate_class_ids:
        LOGGER.warning(f"No plate classes matched {plate_classes}.")
    if plate_recognizer:
        LOGGER.info(f"Loaded plate recognizer from {rec_weights}")
    LOGGER.info(f"Motorperson detector: {weights}")
    LOGGER.info(f"Plate detector: {plate_weights}")
    LOGGER.info(f"Motorperson classes: {sorted(motorperson_class_ids)}")
    LOGGER.info(f"Plate classes: {sorted(plate_class_ids)}")
    LOGGER.info(f"Annotator font: {annotator_font}")

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=main_stride, auto=main_pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=main_stride, auto=main_pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=main_stride, auto=main_pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    main_model.warmup(imgsz=(1 if main_pt or main_model.triton else bs, 3, *imgsz))
    plate_model.warmup(imgsz=(1 if plate_pt or plate_model.triton else 1, 3, *plate_imgsz))

    seen, windows = 0, []
    dt = (Profile(device=device), Profile(device=device), Profile(device=device))

    csv_path = save_dir / "predictions.csv"
    csv_fields = [
        "Image Name",
        "Motorperson Class",
        "Motorperson Confidence",
        "Motorperson X1",
        "Motorperson Y1",
        "Motorperson X2",
        "Motorperson Y2",
        "Plate Prediction",
        "Plate Class Name",
        "Plate Detection Confidence",
        "Plate OCR Confidence",
        "Plate X1",
        "Plate Y1",
        "Plate X2",
        "Plate Y2",
        "Latency",
    ]

    def write_to_csv(data_row):
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)

    for path, im, im0s, vid_cap, s in dataset:
        frame_start = time.time()
        if main_model.device.type == "cuda":
            torch.cuda.synchronize()

        with dt[0]:
            im = torch.from_numpy(im).to(main_model.device)
            im = im.half() if main_model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if main_model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize_dir = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if main_model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = main_model(image, augment=augment, visualize=visualize_dir).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, main_model(image, augment=augment, visualize=visualize_dir).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = main_model(im, augment=augment, visualize=visualize_dir)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, stage1_classes, agnostic_nms, max_det=max_det)

        if main_model.device.type == "cuda":
            torch.cuda.synchronize()

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, font=annotator_font, example="京A12345")

            motorperson_count = 0
            plate_count = 0

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for det_idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    class_id = int(cls)
                    if motorperson_class_ids and class_id not in motorperson_class_ids:
                        continue

                    motorperson_count += 1
                    class_name = main_names[class_id]
                    motorperson_conf = float(conf)
                    motorperson_box = to_int_box(xyxy)
                    motorperson_crop_box = expand_box(
                        motorperson_box,
                        im0.shape,
                        gain=motorperson_crop_gain,
                        pad=motorperson_crop_pad,
                    )

                    if save_img or save_crop or view_img:
                        motorperson_label = None if hide_labels else class_name
                        annotator.box_label(motorperson_box, motorperson_label, color=colors(class_id, True))

                    if save_crop:
                        save_one_box(
                            motorperson_crop_box,
                            imc,
                            file=save_dir / "crops" / "motorperson" / f"{p.stem}_{frame}_{det_idx}.jpg",
                            BGR=True,
                        )

                    crop = im0[
                        motorperson_crop_box[1] : motorperson_crop_box[3],
                        motorperson_crop_box[0] : motorperson_crop_box[2],
                    ]

                    plate_det = detect_on_array(
                        plate_model,
                        crop,
                        plate_imgsz,
                        plate_conf_thres,
                        plate_iou_thres,
                        classes=stage2_classes,
                        agnostic_nms=plate_agnostic_nms,
                        max_det=plate_max_det,
                        augment=augment,
                    )

                    if not len(plate_det):
                        if save_csv:
                            write_to_csv(
                                {
                                    "Image Name": p.name,
                                    "Motorperson Class": class_name,
                                    "Motorperson Confidence": f"{motorperson_conf:.4f}",
                                    "Motorperson X1": motorperson_box[0],
                                    "Motorperson Y1": motorperson_box[1],
                                    "Motorperson X2": motorperson_box[2],
                                    "Motorperson Y2": motorperson_box[3],
                                    "Plate Prediction": "",
                                    "Plate Class Name": "",
                                    "Plate Detection Confidence": "",
                                    "Plate OCR Confidence": "",
                                    "Plate X1": "",
                                    "Plate Y1": "",
                                    "Plate X2": "",
                                    "Plate Y2": "",
                                    "Latency": f"{(time.time() - frame_start) * 1000:.2f}",
                                }
                            )
                        continue

                    for plate_idx, (*plate_xyxy, plate_conf, plate_cls) in enumerate(reversed(plate_det)):
                        plate_count += 1
                        plate_cls_id = int(plate_cls)
                        plate_class_name = plate_names[plate_cls_id]
                        plate_det_conf = float(plate_conf)
                        plate_box_rel = to_int_box(plate_xyxy)
                        plate_box = [
                            plate_box_rel[0] + motorperson_crop_box[0],
                            plate_box_rel[1] + motorperson_crop_box[1],
                            plate_box_rel[2] + motorperson_crop_box[0],
                            plate_box_rel[3] + motorperson_crop_box[1],
                        ]

                        prediction = plate_class_name
                        plate_text = ""
                        plate_score = None

                        if plate_recognizer:
                            rec_box = expand_box(plate_box, im0.shape, gain=rec_crop_gain, pad=rec_crop_pad)
                            rec_crop = im0[rec_box[1] : rec_box[3], rec_box[0] : rec_box[2]]
                            if rec_crop.size:
                                try:
                                    plate_text, plate_score = plate_recognizer.infer(rec_crop)
                                except Exception as exc:
                                    LOGGER.warning(f"OCR failed on {p.name}: {exc}")
                            if plate_text and plate_score is not None and plate_score >= rec_conf_thres:
                                prediction = plate_text

                        if save_csv:
                            write_to_csv(
                                {
                                    "Image Name": p.name,
                                    "Motorperson Class": class_name,
                                    "Motorperson Confidence": f"{motorperson_conf:.4f}",
                                    "Motorperson X1": motorperson_box[0],
                                    "Motorperson Y1": motorperson_box[1],
                                    "Motorperson X2": motorperson_box[2],
                                    "Motorperson Y2": motorperson_box[3],
                                    "Plate Prediction": prediction,
                                    "Plate Class Name": plate_class_name,
                                    "Plate Detection Confidence": f"{plate_det_conf:.4f}",
                                    "Plate OCR Confidence": "" if plate_score is None else f"{plate_score:.6f}",
                                    "Plate X1": plate_box[0],
                                    "Plate Y1": plate_box[1],
                                    "Plate X2": plate_box[2],
                                    "Plate Y2": plate_box[3],
                                    "Latency": f"{(time.time() - frame_start) * 1000:.2f}",
                                }
                            )

                        if save_txt:
                            final_xyxy = torch.tensor(plate_box).view(1, 4)
                            if save_format == 0:
                                coords = (xyxy2xywh(final_xyxy) / gn).view(-1).tolist()
                            else:
                                coords = (final_xyxy / gn).view(-1).tolist()
                            line = (plate_cls_id, *coords, plate_det_conf) if save_conf else (plate_cls_id, *coords)
                            with open(f"{txt_path}.txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or save_crop or view_img:
                            if hide_labels:
                                plate_label = None
                            elif prediction != plate_class_name and plate_score is not None:
                                plate_label = prediction if hide_conf else f"{prediction} rec:{plate_score:.2f}"
                            else:
                                plate_label = plate_class_name
                            annotator.box_label(plate_box, plate_label, color=colors(plate_cls_id + len(main_names), True))

                        if save_crop:
                            save_one_box(
                                plate_box,
                                imc,
                                file=save_dir / "crops" / "plate" / f"{p.stem}_{frame}_{det_idx}_{plate_idx}.jpg",
                                BGR=True,
                            )

            s += f"{motorperson_count} motorperson, {plate_count} plate, "
            im0 = annotator.result()

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{dt[1].dt * 1e3:.1f}ms")

    t = tuple(x.t / seen * 1e3 for x in dt) if seen else (0.0, 0.0, 0.0)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        saved = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{saved}")
    if update:
        strip_optimizer(weights[0] if isinstance(weights, list) else weights)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motorperson-weights",
        "--weights",
        dest="weights",
        nargs="+",
        type=str,
        default=ROOT / "trained_model/car_face_det/v5s_256x256_1b.onnx",
        help="motorperson detector weights",
    )
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument(
        "--motorperson-data",
        "--data",
        dest="data",
        type=str,
        default=ROOT / "nonMoto.yaml",
        help="motorperson detector dataset yaml",
    )
    parser.add_argument(
        "--motorperson-imgsz",
        "--imgsz",
        "--img",
        "--img-size",
        dest="imgsz",
        nargs="+",
        type=int,
        default=[256],
        help="motorperson detector inference size h,w",
    )
    parser.add_argument(
        "--motorperson-conf-thres",
        "--conf-thres",
        dest="conf_thres",
        type=float,
        default=0.25,
        help="motorperson detector confidence threshold",
    )
    parser.add_argument(
        "--motorperson-iou-thres",
        "--iou-thres",
        dest="iou_thres",
        type=float,
        default=0.45,
        help="motorperson detector NMS IoU threshold",
    )
    parser.add_argument(
        "--motorperson-max-det",
        "--max-det",
        dest="max_det",
        type=int,
        default=1000,
        help="motorperson detector maximum detections",
    )
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save final plate results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="save-txt coordinate format, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save plate detection confidence in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save motorperson and plate crops")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="optional main detector class filter")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS for main detector")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide OCR confidence")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--plate-weights", type=str, default=ROOT / "trained_model/yolov5s-plate-detect.pt", help="plate detector weights")
    parser.add_argument("--plate-data", type=str, default=ROOT / "plate_det.yaml", help="plate detector dataset yaml")
    parser.add_argument("--plate-imgsz", nargs="+", type=int, default=[640], help="plate detector inference size h,w")
    parser.add_argument("--plate-conf-thres", type=float, default=0.25, help="plate detector confidence threshold")
    parser.add_argument("--plate-iou-thres", type=float, default=0.45, help="plate detector NMS IoU threshold")
    parser.add_argument("--plate-max-det", type=int, default=20, help="plate detector maximum detections per motorperson")
    parser.add_argument("--plate-agnostic-nms", action="store_true", help="class-agnostic NMS for plate detector")
    parser.add_argument(
        "--motorperson-classes",
        nargs="+",
        default=list(DEFAULT_MOTORPERSON_CLASSES),
        help="main detector class names or ids that should trigger second-stage plate detection",
    )
    parser.add_argument(
        "--plate-classes",
        nargs="+",
        default=list(DEFAULT_PLATE_DET_CLASSES),
        help="plate detector class names or ids that should be kept",
    )
    parser.add_argument("--motorperson-crop-gain", type=float, default=1.02, help="crop gain before plate detection")
    parser.add_argument("--motorperson-crop-pad", type=int, default=10, help="crop pad before plate detection")
    parser.add_argument("--rec-weights", type=str, required=True, help="validation.py compatible OCR checkpoint (.pth)")
    parser.add_argument("--rec-conf-thres", type=float, default=0.0, help="minimum OCR confidence to replace plate label")
    parser.add_argument("--rec-crop-gain", type=float, default=1.02, help="crop gain before OCR")
    parser.add_argument("--rec-crop-pad", type=int, default=10, help="crop pad before OCR")
    parser.add_argument("--font-path", type=str, default="", help="font path for rendering Chinese labels")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    opt.plate_imgsz *= 2 if len(opt.plate_imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main():
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    # run(**vars(opt))
    run(
        source="./img_list.txt", # txt / folder
        plate_weights='./trained_model/car_face_det/v5s_256x256_1b.onnx', # 车牌检测模型 onnx / pt
        weights='./trained_model/car_face_det/1/v5s_416x416.onnx', # 车辆检测模型 onnx / pt
        plate_data='./non-motor-person.yaml', # 车牌检测标签
        data='./nonMoto.yaml', # 车辆检测标签
        conf_thres=0.4,
        plate_imgsz=(256, 256), # 车牌检测模型 imgsz
        imgsz=(416, 416), # 车辆检测模型 imgsz
        save_conf=True,
        # save_crop=True,
        project='./runs/detect',
        name='260327_plate_det_full',
        save_csv=True,
        rec_weights='./trained_model/CRNN_2026_3_27_11_55_19_simRealFusion_style2000/best.pth', # 车牌识别模型
        # save_txt=True,
        device="6"
    )


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    main()
