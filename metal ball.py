# -*- coding: utf-8 -*-
"""
近距离效果还可以，但是远距离效果一般
K230 CanMV 实时检测程序。
模型和部署配置需要放在：/sdcard/mp_deployment_source/
本程序使用 GC2093 摄像头、ST7701 800x480 LCD 和 AnchorBaseDet 后处理。
"""

import gc
import time

import aicube
import image
import nncase_runtime as nn
import ujson
import ulab.numpy as np
from libs.PipeLine import ScopedTiming
from libs.Utils import get_colors
from media.display import Display
from media.media import MediaManager
from media.sensor import (
    CAM_CHN_ID_0,
    CAM_CHN_ID_2,
    PIXEL_FORMAT_RGB_888_PLANAR,
    PIXEL_FORMAT_YUV_SEMIPLANAR_420,
    Sensor,
)


# 模型部署目录由指南规定，不能使用电脑端路径。
ROOT_PATH = "/sdcard/mp_deployment_source/"
CONFIG_PATH = ROOT_PATH + "deploy_config.json"

# 摄像头 AI 通道和 LCD 视频通道尺寸必须与实际硬件配置保持一致。
AI_FRAME_SIZE = [640, 360]
DISPLAY_SIZE = [800, 480]
STRIDES = [8, 16, 32]
DEBUG_MODE = 0
PRINT_INTERVAL_MS = 150
TRACK_ALPHA = 0.75
TRACK_MATCH_DISTANCE = 70
TRACK_MAX_MISSED = 4
TRACK_CONFIRM_HITS = 1
NMS_THRESHOLD_OVERRIDE = 0.35


def read_deploy_config(path):
    """读取部署配置，避免从模型文件名猜测模型参数。"""
    with open(path, "r") as json_file:
        return ujson.load(json_file)


def normalize_labels(raw_labels, num_classes):
    """统一类别名称，确保训练中出现的 metal 仍显示为 metal ball。"""
    labels = []
    for label in raw_labels:
        if label == "metal":
            label = "metal ball"
        if label not in labels:
            labels.append(label)
    if len(labels) != num_classes:
        raise ValueError("categories 数量与 num_classes 不一致")
    return labels


def two_side_pad_param(input_size, output_size):
    """计算保持比例缩放所需的上下左右 padding。"""
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    ratio = min(ratio_w, ratio_h)
    new_w = int(ratio * input_size[0])
    new_h = int(ratio * input_size[1])
    dw = (output_size[0] - new_w) / 2
    dh = (output_size[1] - new_h) / 2
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    return top, bottom, left, right


def load_runtime_config():
    """校验当前模型配置并展开 AnchorBaseDet 的 anchors。"""
    config = read_deploy_config(CONFIG_PATH)
    model_type = config["model_type"]
    if model_type != "AnchorBaseDet":
        raise ValueError("当前程序只支持 AnchorBaseDet")

    model_input_size = config["img_size"]
    num_classes = config["num_classes"]
    labels = normalize_labels(config["categories"], num_classes)
    anchors = config["anchors"][0] + config["anchors"][1] + config["anchors"][2]
    return {
        "model_path": ROOT_PATH + config["kmodel_path"],
        "labels": labels,
        "model_input_size": model_input_size,
        "num_classes": num_classes,
        "confidence_threshold": config["confidence_threshold"],
        # 重叠目标容易产生重复框，使用较低 NMS 阈值抑制同球重复检测。
        "nms_threshold": NMS_THRESHOLD_OVERRIDE,
        "nms_option": config["nms_option"],
        "anchors": anchors,
    }


class BallTrack:
    """保存一个球的平滑框和短时跟踪状态。"""

    def __init__(self, track_id, det_box):
        self.track_id = track_id
        self.bbox = [det_box[2], det_box[3], det_box[4], det_box[5]]
        self.score = det_box[1]
        self.hits = 1
        self.missed = 0

    def update(self, det_box):
        """用指数平滑更新坐标，降低单帧检测噪声。"""
        for index in range(4):
            current = det_box[index + 2]
            self.bbox[index] = (
                TRACK_ALPHA * current
                + (1.0 - TRACK_ALPHA) * self.bbox[index]
            )
        self.score = det_box[1]
        self.hits += 1
        self.missed = 0


def box_iou(box_a, box_b):
    """计算两个检测框的 IoU，用于跨帧匹配目标。"""
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_distance_squared(track_box, det_box):
    """计算跟踪框中心和新检测框中心的平方距离。"""
    track_x = (track_box[0] + track_box[2]) / 2
    track_y = (track_box[1] + track_box[3]) / 2
    det_x = (det_box[2] + det_box[4]) / 2
    det_y = (det_box[3] + det_box[5]) / 2
    dx = track_x - det_x
    dy = track_y - det_y
    return dx * dx + dy * dy


def update_tracks(det_boxes, tracks, next_track_id):
    """跨帧匹配检测结果，短暂漏检时保留原目标。"""
    detections = [] if not det_boxes else det_boxes
    candidates = []
    for track_index in range(len(tracks)):
        for det_index in range(len(detections)):
            det_box = detections[det_index]
            distance_squared = center_distance_squared(
                tracks[track_index].bbox,
                det_box,
            )
            iou = box_iou(tracks[track_index].bbox, det_box[2:6])
            if (
                distance_squared <= TRACK_MATCH_DISTANCE * TRACK_MATCH_DISTANCE
                or iou >= 0.05
            ):
                # IoU 越大、中心距离越小，越优先匹配。
                cost = distance_squared - iou * 4000
                candidates.append((cost, track_index, det_index))
    candidates.sort(key=lambda item: item[0])

    matched_tracks = []
    matched_detections = []
    for unused_cost, track_index, det_index in candidates:
        if track_index in matched_tracks or det_index in matched_detections:
            continue
        tracks[track_index].update(detections[det_index])
        matched_tracks.append(track_index)
        matched_detections.append(det_index)

    for track_index in range(len(tracks)):
        if track_index not in matched_tracks:
            tracks[track_index].missed += 1

    for det_index in range(len(detections)):
        if det_index not in matched_detections:
            tracks.append(BallTrack(next_track_id, detections[det_index]))
            next_track_id += 1

    tracks = [track for track in tracks if track.missed <= TRACK_MAX_MISSED]
    return tracks, next_track_id


def create_runtime(runtime):
    """初始化 KPU、AI2D、摄像头和 LCD 显示资源。"""
    model_input_size = runtime["model_input_size"]
    top, bottom, left, right = two_side_pad_param(AI_FRAME_SIZE, model_input_size)

    kpu = nn.kpu()
    kpu.load_kmodel(runtime["model_path"])

    ai2d = nn.ai2d()
    ai2d.set_dtype(
        nn.ai2d_format.NCHW_FMT,
        nn.ai2d_format.NCHW_FMT,
        np.uint8,
        np.uint8,
    )
    ai2d.set_pad_param(
        True,
        [0, 0, 0, 0, top, bottom, left, right],
        0,
        [114, 114, 114],
    )
    ai2d.set_resize_param(
        True,
        nn.interp_method.tf_bilinear,
        nn.interp_mode.half_pixel,
    )
    ai2d_builder = ai2d.build(
        [1, 3, AI_FRAME_SIZE[1], AI_FRAME_SIZE[0]],
        [1, 3, model_input_size[1], model_input_size[0]],
    )

    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.set_framesize(
        width=DISPLAY_SIZE[0],
        height=DISPLAY_SIZE[1],
        chn=CAM_CHN_ID_0,
    )
    sensor.set_pixformat(
        PIXEL_FORMAT_YUV_SEMIPLANAR_420,
        chn=CAM_CHN_ID_0,
    )
    sensor.set_framesize(
        width=AI_FRAME_SIZE[0],
        height=AI_FRAME_SIZE[1],
        chn=CAM_CHN_ID_2,
    )
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)

    sensor_bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
    Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)
    Display.init(Display.ST7701)
    MediaManager.init()
    sensor.run()

    data = np.ones(
        (1, 3, model_input_size[1], model_input_size[0]),
        dtype=np.uint8,
    )
    ai2d_output_tensor = nn.from_numpy(data)
    osd_img = image.Image(
        DISPLAY_SIZE[0],
        DISPLAY_SIZE[1],
        image.ARGB8888,
    )
    return kpu, ai2d_builder, ai2d_output_tensor, sensor, osd_img


def draw_results(osd_img, tracks, runtime, colors, print_result, fps):
    """将 AI 坐标映射到 LCD，并绘制检测框、类别、中心坐标、数量和 FPS。"""
    osd_img.clear()
    confirmed_tracks = [
        track for track in tracks if track.hits >= TRACK_CONFIRM_HITS
    ]
    count = len(confirmed_tracks)
    osd_img.draw_string_advanced(
        12,
        10,
        28,
        "metal ball count: %d FPS: %.1f" % (count, fps),
        color=(255, 255, 0, 0),
    )
    if not confirmed_tracks:
        Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
        return

    labels = runtime["labels"]
    for track in confirmed_tracks:
        class_id = 0
        score = track.score
        x1, y1, x2, y2 = track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3]
        x = int(x1 * DISPLAY_SIZE[0] / AI_FRAME_SIZE[0])
        y = int(y1 * DISPLAY_SIZE[1] / AI_FRAME_SIZE[1])
        w = int((x2 - x1) * DISPLAY_SIZE[0] / AI_FRAME_SIZE[0])
        h = int((y2 - y1) * DISPLAY_SIZE[1] / AI_FRAME_SIZE[1])
        color = colors[class_id][1:]
        osd_img.draw_rectangle(x, y, w, h, color=color)
        text = labels[class_id]
        osd_img.draw_string_advanced(x, max(0, y - 32), 24, text, color=color)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        center_text = "center=(%d,%d)" % (center_x, center_y)
        osd_img.draw_string_advanced(
            x,
            min(DISPLAY_SIZE[1] - 24, y + h + 4),
            20,
            center_text,
            color=color,
        )
        if print_result:
            print(
                "ball center=(%d,%d) box=(%d,%d,%d,%d) score=%.2f"
                % (center_x, center_y, int(x1), int(y1), int(x2), int(y2), score)
            )
    Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)


def run_detection():
    """执行实时推理主循环，并按初始化状态释放资源。"""
    runtime = load_runtime_config()
    colors = get_colors(runtime["num_classes"])
    kpu = None
    ai2d_builder = None
    ai2d_output_tensor = None
    sensor = None
    osd_img = None
    display_initialized = False
    media_initialized = False
    last_print_ms = 0
    fps_window_start_ms = time.ticks_ms()
    fps_frame_count = 0
    fps = 0.0
    frame_count = 0
    tracks = []
    next_track_id = 0
    try:
        kpu, ai2d_builder, ai2d_output_tensor, sensor, osd_img = create_runtime(runtime)
        display_initialized = True
        media_initialized = True
        while True:
            with ScopedTiming("total", DEBUG_MODE > 0):
                rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
                if rgb888p_img.format() != image.RGBP888:
                    continue

                ai2d_input = rgb888p_img.to_numpy_ref()
                ai2d_input_tensor = nn.from_numpy(ai2d_input)
                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                kpu.set_input_tensor(0, ai2d_output_tensor)
                kpu.run()

                results = []
                for index in range(kpu.outputs_size()):
                    output_tensor = kpu.get_output_tensor(index)
                    output = output_tensor.to_numpy()
                    output = output.reshape((output.size,))
                    del output_tensor
                    results.append(output)

                det_boxes = aicube.anchorbasedet_post_process(
                    results[0],
                    results[1],
                    results[2],
                    runtime["model_input_size"],
                    AI_FRAME_SIZE,
                    STRIDES,
                    runtime["num_classes"],
                    runtime["confidence_threshold"],
                    runtime["nms_threshold"],
                    runtime["anchors"],
                    runtime["nms_option"],
                )
                tracks, next_track_id = update_tracks(
                    det_boxes,
                    tracks,
                    next_track_id,
                )
                fps_frame_count += 1
                now_ms = time.ticks_ms()
                fps_window_ms = time.ticks_diff(now_ms, fps_window_start_ms)
                if fps_window_ms >= 500:
                    fps = fps_frame_count * 1000.0 / fps_window_ms
                    fps_frame_count = 0
                    fps_window_start_ms = now_ms
                print_result = now_ms - last_print_ms >= PRINT_INTERVAL_MS
                draw_results(osd_img, tracks, runtime, colors, print_result, fps)
                if print_result:
                    last_print_ms = now_ms
                del ai2d_input_tensor
                del results
                frame_count += 1
                # 垃圾回收不必每帧执行，降低对实时推理的打断。
                if frame_count >= 30:
                    gc.collect()
                    frame_count = 0
    except KeyboardInterrupt:
        print("检测程序已停止")
    except BaseException as exc:
        print("runtime error: %s" % exc)
    finally:
        if sensor is not None:
            sensor.stop()
        if display_initialized:
            Display.deinit()
        if media_initialized:
            MediaManager.deinit()
        del ai2d_output_tensor
        gc.collect()
        nn.shrink_memory_pool()


if __name__ == "__main__":
    run_detection()
