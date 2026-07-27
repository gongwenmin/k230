# -*- coding: utf-8 -*-
"""K230 metal-ball detection and UART2 transmitter for MSPM0."""

import gc
import time

import aicube
import image
import nncase_runtime as nn
import ujson
import ulab.numpy as np
from machine import FPIOA, UART
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

ROOT_PATH = "/sdcard/mp_deployment_source/"
CONFIG_PATH = ROOT_PATH + "deploy_config.json"
AI_FRAME_SIZE = [640, 360]
DISPLAY_SIZE = [800, 480]
STRIDES = [8, 16, 32]
DEBUG_MODE = 0
SEND_INTERVAL_MS = 150
TRACK_ALPHA = 0.35
TRACK_MATCH_DISTANCE = 70
TRACK_MAX_MISSED = 4
TRACK_CONFIRM_HITS = 2
NMS_THRESHOLD_OVERRIDE = 0.35
MAX_BALLS = 16


def init_uart():
    print("init_uart entered")

    fpioa = FPIOA()
    fpioa.set_function(11, FPIOA.UART2_TXD)
    fpioa.set_function(12, FPIOA.UART2_RXD)

    uart = UART(
        UART.UART2,
        baudrate=115200,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
    )

    print("UART2 ready")
    return uart


def send_ball_frame(uart, tracks):
    confirmed = [track for track in tracks if track.hits >= TRACK_CONFIRM_HITS]
    confirmed = confirmed[:MAX_BALLS]

    fields = ["<BALL", str(len(confirmed))]
    for track in confirmed:
        x1, y1, x2, y2 = track.bbox
        fields.extend([
            str(int((x1 + x2) / 2)),
            str(int((y1 + y2) / 2)),
        ])

    frame = ",".join(fields) + ">\r\n"
    uart.write(frame.encode())
    print("TX:", frame)


def read_deploy_config(path):
    with open(path, "r") as json_file:
        return ujson.load(json_file)


def normalize_labels(raw_labels, num_classes):
    labels = []
    for label in raw_labels:
        if label == "metal":
            label = "metal ball"
        if label not in labels:
            labels.append(label)
    if len(labels) != num_classes:
        raise ValueError("categories count does not match num_classes")
    return labels


def two_side_pad_param(input_size, output_size):
    ratio = min(output_size[0] / input_size[0], output_size[1] / input_size[1])
    new_w = int(ratio * input_size[0])
    new_h = int(ratio * input_size[1])
    dw = (output_size[0] - new_w) / 2
    dh = (output_size[1] - new_h) / 2
    return (
        int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
    )


def load_runtime_config():
    config = read_deploy_config(CONFIG_PATH)
    if config["model_type"] != "AnchorBaseDet":
        raise ValueError("only AnchorBaseDet is supported")
    labels = normalize_labels(config["categories"], config["num_classes"])
    anchors = config["anchors"][0] + config["anchors"][1] + config["anchors"][2]
    return {
        "model_path": ROOT_PATH + config["kmodel_path"],
        "labels": labels,
        "model_input_size": config["img_size"],
        "num_classes": config["num_classes"],
        "confidence_threshold": config["confidence_threshold"],
        "nms_threshold": NMS_THRESHOLD_OVERRIDE,
        "nms_option": config["nms_option"],
        "anchors": anchors,
    }


class BallTrack:
    def __init__(self, track_id, det_box):
        self.track_id = track_id
        self.bbox = [det_box[2], det_box[3], det_box[4], det_box[5]]
        self.score = det_box[1]
        self.hits = 1
        self.missed = 0

    def update(self, det_box):
        for index in range(4):
            current = det_box[index + 2]
            self.bbox[index] = TRACK_ALPHA * current + (1.0 - TRACK_ALPHA) * self.bbox[index]
        self.score = det_box[1]
        self.hits += 1
        self.missed = 0


def box_iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_distance_squared(track_box, det_box):
    track_x = (track_box[0] + track_box[2]) / 2
    track_y = (track_box[1] + track_box[3]) / 2
    det_x = (det_box[2] + det_box[4]) / 2
    det_y = (det_box[3] + det_box[5]) / 2
    return (track_x - det_x) ** 2 + (track_y - det_y) ** 2


def update_tracks(det_boxes, tracks, next_track_id):
    detections = [] if not det_boxes else det_boxes
    candidates = []
    for track_index, track in enumerate(tracks):
        for det_index, det_box in enumerate(detections):
            distance = center_distance_squared(track.bbox, det_box)
            iou = box_iou(track.bbox, det_box[2:6])
            if distance <= TRACK_MATCH_DISTANCE ** 2 or iou >= 0.05:
                candidates.append((distance - iou * 4000, track_index, det_index))
    candidates.sort(key=lambda item: item[0])
    matched_tracks = []
    matched_detections = []
    for _, track_index, det_index in candidates:
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
    return [track for track in tracks if track.missed <= TRACK_MAX_MISSED], next_track_id


def create_runtime(runtime):
    model_input_size = runtime["model_input_size"]
    top, bottom, left, right = two_side_pad_param(AI_FRAME_SIZE, model_input_size)
    kpu = nn.kpu()
    kpu.load_kmodel(runtime["model_path"])
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0, 0, 0, 0, top, bottom, left, right], 0, [114, 114, 114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
    ai2d_builder = ai2d.build(
        [1, 3, AI_FRAME_SIZE[1], AI_FRAME_SIZE[0]],
        [1, 3, model_input_size[1], model_input_size[0]],
    )
    sensor = Sensor(id=2)
    sensor.reset()
    sensor.set_hmirror(False)
    sensor.set_vflip(False)
    sensor.set_framesize(width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1], chn=CAM_CHN_ID_0)
    sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420, chn=CAM_CHN_ID_0)
    sensor.set_framesize(width=AI_FRAME_SIZE[0], height=AI_FRAME_SIZE[1], chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)
    sensor_bind_info = sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_0)
    Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)
    Display.init(Display.ST7701)
    MediaManager.init()
    sensor.run()
    data = np.ones((1, 3, model_input_size[1], model_input_size[0]), dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(data)
    osd_img = image.Image(DISPLAY_SIZE[0], DISPLAY_SIZE[1], image.ARGB8888)
    return kpu, ai2d_builder, ai2d_output_tensor, sensor, osd_img


def draw_results(osd_img, tracks, runtime, colors):
    osd_img.clear()
    confirmed = [track for track in tracks if track.hits >= TRACK_CONFIRM_HITS]
    osd_img.draw_string_advanced(12, 10, 28, "metal ball count: %d" % len(confirmed), color=(255, 255, 0, 0))
    for track in confirmed:
        x1, y1, x2, y2 = track.bbox
        x = int(x1 * DISPLAY_SIZE[0] / AI_FRAME_SIZE[0])
        y = int(y1 * DISPLAY_SIZE[1] / AI_FRAME_SIZE[1])
        w = int((x2 - x1) * DISPLAY_SIZE[0] / AI_FRAME_SIZE[0])
        h = int((y2 - y1) * DISPLAY_SIZE[1] / AI_FRAME_SIZE[1])
        color = colors[0][1:]
        osd_img.draw_rectangle(x, y, w, h, color=color)
        osd_img.draw_string_advanced(x, max(0, y - 32), 24, "%s %.2f" % (runtime["labels"][0], track.score), color=color)
    Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)


def run_detection():
    runtime = load_runtime_config()
    colors = get_colors(runtime["num_classes"])
    uart = init_uart()

    # Temporary UART link test. Remove this loop after the MSPM0 receives the fixed frame.
    while True:
        frame = b"<BALL,2,120,85,306,174>\r\n"
        uart.write(frame)
        print("TX:", frame)
        time.sleep_ms(500)

    kpu = None
    ai2d_builder = None
    ai2d_output_tensor = None
    sensor = None
    osd_img = None
    last_send_ms = time.ticks_ms() - SEND_INTERVAL_MS
    tracks = []
    next_track_id = 0
    frame_count = 0
    try:
        kpu, ai2d_builder, ai2d_output_tensor, sensor, osd_img = create_runtime(runtime)
        while True:
            with ScopedTiming("total", DEBUG_MODE > 0):
                rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
                if rgb888p_img.format() != image.RGBP888:
                    continue
                ai2d_input_tensor = nn.from_numpy(rgb888p_img.to_numpy_ref())
                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                kpu.set_input_tensor(0, ai2d_output_tensor)
                kpu.run()
                results = []
                for index in range(kpu.outputs_size()):
                    output_tensor = kpu.get_output_tensor(index)
                    output_array = output_tensor.to_numpy()
                    output = output_array.reshape((output_array.size,))
                    del output_tensor
                    results.append(output)
                det_boxes = aicube.anchorbasedet_post_process(
                    results[0], results[1], results[2], runtime["model_input_size"],
                    AI_FRAME_SIZE, STRIDES, runtime["num_classes"],
                    runtime["confidence_threshold"], runtime["nms_threshold"],
                    runtime["anchors"], runtime["nms_option"],
                )
                tracks, next_track_id = update_tracks(det_boxes, tracks, next_track_id)
                draw_results(osd_img, tracks, runtime, colors)
                now_ms = time.ticks_ms()
                if time.ticks_diff(now_ms, last_send_ms) >= SEND_INTERVAL_MS:
                    send_ball_frame(uart, tracks)
                    last_send_ms = now_ms
                del ai2d_input_tensor
                del results
                frame_count += 1
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
        if osd_img is not None:
            Display.deinit()
        MediaManager.deinit()
        uart.deinit()
        del ai2d_output_tensor
        gc.collect()
        nn.shrink_memory_pool()

if __name__ == "__main__":
    run_detection()
