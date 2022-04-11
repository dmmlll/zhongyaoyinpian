import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith(
        '.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                                   exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (
                            cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label,color=colors[int(cls)], line_thickness=8)
                        # 中文输出
                        if names[int(cls)] == 'Anxixiang':
                            ch_text = '%s，%.2f' % ('安息香', conf)
                        elif names[int(cls)] == 'Baibiandou':
                            ch_text = '%s，%.2f' % ('白扁豆', conf)
                        elif names[int(cls)] == 'Baifan':
                            ch_text = '%s，%.2f' % ('白矾', conf)
                        elif names[int(cls)] == 'Baijiezi':
                            ch_text = '%s，%.2f' % ('白芥子', conf)
                        elif names[int(cls)] == 'Bailian':
                            ch_text = '%s，%.2f' % ('白蔹', conf)
                        elif names[int(cls)] == 'Baimaogen':
                            ch_text = '%s，%.2f' % ('白茅根', conf)
                        elif names[int(cls)] == 'Baiqian':
                            ch_text = '%s，%.2f' % ('白前', conf)
                        elif names[int(cls)] == 'Baishao':
                            ch_text = '%s，%.2f' % ('白芍', conf)
                        elif names[int(cls)] == 'Baizhu':
                            ch_text = '%s，%.2f' % ('白术', conf)
                        elif names[int(cls)] == 'Baiziren':
                            ch_text = '%s，%.2f' % ('柏子仁', conf)
                        # ---------------------------------------------------
                        elif names[int(cls)] == 'Banlangen':
                            ch_text = '%s，%.2f' % ('板蓝根', conf)
                        elif names[int(cls)] == 'Beishashen':
                            ch_text = '%s，%.2f' % ('北沙参', conf)
                        elif names[int(cls)] == 'Bibo':
                            ch_text = '%s，%.2f' % ('荜拨', conf)
                        elif names[int(cls)] == 'Bichengjia':
                            ch_text = '%s，%.2f' % ('荜澄茄', conf)
                        elif names[int(cls)] == 'Bohe':
                            ch_text = '%s，%.2f' % ('薄荷', conf)
                        elif names[int(cls)] == 'Cangzhu':
                            ch_text = '%s，%.2f' % ('苍术', conf)
                        elif names[int(cls)] == 'Caodoukou':
                            ch_text = '%s，%.2f' % ('草豆蔻', conf)
                        elif names[int(cls)] == 'Chaihu':
                            ch_text = '%s，%.2f' % ('柴胡', conf)
                        elif names[int(cls)] == 'Chenpi':
                            ch_text = '%s，%.2f' % ('陈皮', conf)
                        elif names[int(cls)] == 'Chenxiang':
                            ch_text = '%s，%.2f' % ('沉香', conf)
                        # ----------------------------------------------------
                        elif names[int(cls)] == 'Chishao':
                            ch_text = '%s，%.2f' % ('赤芍', conf)
                        elif names[int(cls)] == 'Chishizhi':
                            ch_text = '%s，%.2f' % ('赤石脂', conf)
                        elif names[int(cls)] == 'Chuanlianzi':
                            ch_text = '%s，%.2f' % ('川楝子', conf)
                        elif names[int(cls)] == 'Chuanmuxiang':
                            ch_text = '%s，%.2f' % ('川木香', conf)
                        elif names[int(cls)] == 'Chuanniuxi':
                            ch_text = '%s，%.2f' % ('川牛膝', conf)
                        elif names[int(cls)] == 'Chuanqiong':
                            ch_text = '%s，%.2f' % ('川穹', conf)
                        elif names[int(cls)] == 'Dafupi':
                            ch_text = '%s，%.2f' % ('大腹皮', conf)
                        elif names[int(cls)] == 'Dandouchi':
                            ch_text = '%s，%.2f' % ('淡豆豉', conf)
                        elif names[int(cls)] == 'Danggui':
                            ch_text = '%s，%.2f' % ('当归', conf)
                        elif names[int(cls)] == 'Danshen':
                            ch_text = '%s，%.2f' % ('丹参', conf)
                        # -----------------------------------------------------------
                        elif names[int(cls)] == 'Daoya':
                            ch_text = '%s，%.2f' % ('稻芽', conf)
                        elif names[int(cls)] == 'Daqingye':
                            ch_text = '%s，%.2f' % ('大青叶', conf)
                        elif names[int(cls)] == 'Dilong':
                            ch_text = '%s，%.2f' % ('地龙', conf)
                        elif names[int(cls)] == 'Fangfeng':
                            ch_text = '%s，%.2f' % ('防风', conf)
                        elif names[int(cls)] == 'Fanxieye':
                            ch_text = '%s，%.2f' % ('番泻叶', conf)
                        elif names[int(cls)] == 'Fengfang':
                            ch_text = '%s，%.2f' % ('蜂房', conf)
                        elif names[int(cls)] == 'Fuling':
                            ch_text = '%s，%.2f' % ('茯苓', conf)
                        elif names[int(cls)] == 'Gancao':
                            ch_text = '%s，%.2f' % ('甘草', conf)
                        elif names[int(cls)] == 'Ganjiang':
                            ch_text = '%s，%.2f' % ('干姜', conf)
                        elif names[int(cls)] == 'Gansong':
                            ch_text = '%s，%.2f' % ('甘松', conf)
                        # ---------------------------------------------------
                        elif names[int(cls)] == 'Gongdingxiang':
                            ch_text = '%s，%.2f' % ('公丁香', conf)
                        elif names[int(cls)] == 'Guizhi':
                            ch_text = '%s，%.2f' % ('桂枝', conf)
                        elif names[int(cls)] == 'Gujingcao':
                            ch_text = '%s，%.2f' % ('谷精草', conf)
                        elif names[int(cls)] == 'Guya':
                            ch_text = '%s，%.2f' % ('谷芽', conf)
                        elif names[int(cls)] == 'Haipiaoshao':
                            ch_text = '%s，%.2f' % ('海螵蛸', conf)
                        elif names[int(cls)] == 'Haoben':
                            ch_text = '%s，%.2f' % ('蒿本', conf)
                        elif names[int(cls)] == 'Hehuanpi':
                            ch_text = '%s，%.2f' % ('合欢皮', conf)
                        elif names[int(cls)] == 'Huangbai':
                            ch_text = '%s，%.2f' % ('黄柏', conf)
                        elif names[int(cls)] == 'Huangqi':
                            ch_text = '%s，%.2f' % ('黄芪', conf)
                        elif names[int(cls)] == 'Huangqin':
                            ch_text = '%s，%.2f' % ('黄芩', conf)
                        # ---------------------------------------------------
                        elif names[int(cls)] == 'Huoxiang':
                            ch_text = '%s，%.2f' % ('藿香', conf)
                        elif names[int(cls)] == 'Jiangcan':
                            ch_text = '%s，%.2f' % ('僵蚕', conf)
                        elif names[int(cls)] == 'Jiguanhua':
                            ch_text = '%s，%.2f' % ('鸡冠花', conf)
                        elif names[int(cls)] == 'Jindenglong':
                            ch_text = '%s，%.2f' % ('锦灯笼', conf)
                        elif names[int(cls)] == 'Jingyesui':
                            ch_text = '%s，%.2f' % ('荆芥穗', conf)
                        elif names[int(cls)] == 'Jinyinhua':
                            ch_text = '%s，%.2f' % ('金银花', conf)
                        elif names[int(cls)] == 'Jiuxiangchong':
                            ch_text = '%s，%.2f' % ('九香虫', conf)
                        elif names[int(cls)] == 'Juhe':
                            ch_text = '%s，%.2f' % ('橘核', conf)
                        elif names[int(cls)] == 'Kudiding':
                            ch_text = '%s，%.2f' % ('苦地丁', conf)
                        elif names[int(cls)] == 'Laifuzi':
                            ch_text = '%s，%.2f' % ('莱菔子', conf)
                        # -----------------------------------------------
                        elif names[int(cls)] == 'Lianqiao':
                            ch_text = '%s，%.2f' % ('连翘', conf)
                        elif names[int(cls)] == 'Lianxu':
                            ch_text = '%s，%.2f' % ('莲须', conf)
                        elif names[int(cls)] == 'Lianzi':
                            ch_text = '%s，%.2f' % ('莲子', conf)
                        elif names[int(cls)] == 'Lianzixin':
                            ch_text = '%s，%.2f' % ('莲子心', conf)
                        elif names[int(cls)] == 'Lingzhi':
                            ch_text = '%s，%.2f' % ('灵芝', conf)
                        elif names[int(cls)] == 'Lizhihe':
                            ch_text = '%s，%.2f' % ('荔枝核', conf)
                        elif names[int(cls)] == 'Longyan':
                            ch_text = '%s，%.2f' % ('龙眼', conf)
                        elif names[int(cls)] == 'Lugen':
                            ch_text = '%s，%.2f' % ('芦根', conf)
                        elif names[int(cls)] == 'Lulutong':
                            ch_text = '%s，%.2f' % ('路路通', conf)
                        elif names[int(cls)] == 'Mahuang':
                            ch_text = '%s，%.2f' % ('麻黄', conf)
                        # -------------------------------------------------
                        elif names[int(cls)] == 'Maidong':
                            ch_text = '%s，%.2f' % ('麦冬', conf)
                        elif names[int(cls)] == 'Niubangzi':
                            ch_text = '%s，%.2f' % ('牛蒡子', conf)
                        elif names[int(cls)] == 'Qianghuo':
                            ch_text = '%s，%.2f' % ('羌活', conf)
                        elif names[int(cls)] == 'Qiannianjian':
                            ch_text = '%s，%.2f' % ('千年健', conf)
                        elif names[int(cls)] == 'Qinghao':
                            ch_text = '%s，%.2f' % ('青蒿', conf)
                        elif names[int(cls)] == 'Qinpi':
                            ch_text = '%s，%.2f' % ('秦皮', conf)
                        elif names[int(cls)] == 'Rendongteng':
                            ch_text = '%s，%.2f' % ('忍冬藤', conf)
                        elif names[int(cls)] == 'Renshen':
                            ch_text = '%s，%.2f' % ('人参', conf)
                        elif names[int(cls)] == 'Roudoukou':
                            ch_text = '%s，%.2f' % ('肉豆蔻', conf)
                        elif names[int(cls)] == 'Sangjisheng':
                            ch_text = '%s，%.2f' % ('桑寄生', conf)
                        # ------------------------------------------------
                        elif names[int(cls)] == 'Sangpiaoshao':
                            ch_text = '%s，%.2f' % ('桑螵蛸', conf)
                        elif names[int(cls)] == 'Sangshen':
                            ch_text = '%s，%.2f' % ('桑葚', conf)
                        elif names[int(cls)] == 'Shancigu':
                            ch_text = '%s，%.2f' % ('山慈菇', conf)
                        elif names[int(cls)] == 'Shannai':
                            ch_text = '%s，%.2f' % ('山奈', conf)
                        elif names[int(cls)] == 'Shanyao':
                            ch_text = '%s，%.2f' % ('山药', conf)
                        elif names[int(cls)] == 'Shayuanzi':
                            ch_text = '%s，%.2f' % ('沙苑子', conf)
                        elif names[int(cls)] == 'Shegan':
                            ch_text = '%s，%.2f' % ('射干', conf)
                        elif names[int(cls)] == 'Shengjineijin':
                            ch_text = '%s，%.2f' % ('鸡内金', conf)
                        elif names[int(cls)] == 'Shengwalengzi':
                            ch_text = '%s，%.2f' % ('瓦楞子', conf)
                        elif names[int(cls)] == 'Shiliupi':
                            ch_text = '%s，%.2f' % ('石榴皮', conf)
                        # ----------------------------------------------------
                        elif names[int(cls)] == 'Sigualuo':
                            ch_text = '%s，%.2f' % ('丝瓜络', conf)
                        elif names[int(cls)] == 'Suanzaoren':
                            ch_text = '%s，%.2f' % ('酸枣仁', conf)
                        elif names[int(cls)] == 'Sumu':
                            ch_text = '%s，%.2f' % ('苏木', conf)
                        elif names[int(cls)] == 'Taizishen':
                            ch_text = '%s，%.2f' % ('太子参', conf)
                        elif names[int(cls)] == 'Tianhuafen':
                            ch_text = '%s，%.2f' % ('天花粉', conf)
                        elif names[int(cls)] == 'Tianma':
                            ch_text = '%s，%.2f' % ('天麻', conf)
                        elif names[int(cls)] == 'Tujingpi':
                            ch_text = '%s，%.2f' % ('土荆皮', conf)
                        elif names[int(cls)] == 'Wujiapi':
                            ch_text = '%s，%.2f' % ('五加皮', conf)
                        elif names[int(cls)] == 'Xingren':
                            ch_text = '%s，%.2f' % ('杏仁', conf)
                        elif names[int(cls)] == 'Xixin':
                            ch_text = '%s，%.2f' % ('细辛', conf)
                        # ------------------------------------------------
                        elif names[int(cls)] == 'Yinchaihu':
                            ch_text = '%s，%.2f' % ('银柴胡', conf)
                        elif names[int(cls)] == 'Yiren':
                            ch_text = '%s，%.2f' % ('薏仁', conf)
                        elif names[int(cls)] == 'Yvjin':
                            ch_text = '%s，%.2f' % ('郁金', conf)
                        elif names[int(cls)] == 'Zhebeimu':
                            ch_text = '%s，%.2f' % ('浙贝母', conf)
                        elif names[int(cls)] == 'Zhiqiao':
                            ch_text = '%s，%.2f' % ('枳壳', conf)
                        elif names[int(cls)] == 'Zhuru':
                            ch_text = '%s，%.2f' % ('竹茹', conf)
                        elif names[int(cls)] == 'Zhuyazao':
                            ch_text = '%s，%.2f' % ('猪牙皂', conf)

                        im0 = plot_one_box(xyxy, im0, label=label, ch_text=ch_text, color=colors[int(cls)],line_thickness=8)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            # save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            # 参考 https://xugaoxiang.com/2021/08/20/opencv-h264-videowrite
                            save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
