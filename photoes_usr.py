#按下b21（板载usr按键就拍照存到canmv的photo文件夹里）
import os
import time

from machine import FPIOA, Pin
from media.display import Display
from media.media import MediaManager
from media.sensor import *


# Lushan Pi K230 onboard GC2093 uses CSI 2.
SENSOR_ID = 2
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
PHOTO_DIR = "/sdcard/photo"
BUTTON_GPIO = 53
DEBOUNCE_MS = 200


def ensure_photo_dir():
    """Create the photo directory when it is not present."""
    try:
        os.stat(PHOTO_DIR)
    except OSError:
        os.mkdir(PHOTO_DIR)


def get_next_photo_index():
    """Continue numbering after reboot instead of overwriting old photos."""
    next_index = 1
    try:
        for name in os.listdir(PHOTO_DIR):
            if not name.startswith("photo_") or not name.endswith(".jpg"):
                continue
            number_text = name[6:-4]
            try:
                number = int(number_text)
                if number >= next_index:
                    next_index = number + 1
            except ValueError:
                pass
    except OSError:
        pass
    return next_index


def init_button():
    """Configure the onboard USER button: GPIO53, pull-down, pressed=1."""
    fpioa = FPIOA()
    fpioa.set_function(BUTTON_GPIO, FPIOA.GPIO53)
    return Pin(BUTTON_GPIO, Pin.IN, pull=Pin.PULL_DOWN)


def save_photo(image, photo_index):
    """Save the current RGB565 frame as a numbered JPEG file."""
    filename = "%s/photo_%06d.jpg" % (PHOTO_DIR, photo_index)
    image.save(filename)
    print("photo saved: %s" % filename)
    return photo_index + 1


def main():
    sensor = None
    sensor_started = False
    display_started = False
    media_started = False

    try:
        ensure_photo_dir()
        button = init_button()
        photo_index = get_next_photo_index()

        sensor = Sensor(id=SENSOR_ID)
        sensor.reset()
        sensor.set_hmirror(False)
        sensor.set_vflip(False)
        sensor.set_framesize(
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            chn=CAM_CHN_ID_0,
        )
        sensor.set_pixformat(
            Sensor.RGB565,
            chn=CAM_CHN_ID_0,
        )

        Display.init(Display.ST7701)
        display_started = True

        MediaManager.init()
        media_started = True
        sensor.run()
        sensor_started = True

        previous_state = button.value()
        last_photo_ms = time.ticks_ms() - DEBOUNCE_MS
        print("camera preview started")
        print("press onboard USER button GPIO53 to take a photo")
        print("photo directory: %s" % PHOTO_DIR)

        while True:
            os.exitpoint()
            image = sensor.snapshot(chn=CAM_CHN_ID_0)
            Display.show_image(image)

            current_state = button.value()
            now_ms = time.ticks_ms()

            # Detect the press edge and debounce the board button in software.
            if current_state == 1 and previous_state == 0:
                if time.ticks_diff(now_ms, last_photo_ms) >= DEBOUNCE_MS:
                    try:
                        photo_index = save_photo(image, photo_index)
                        last_photo_ms = now_ms
                    except Exception as exc:
                        print("photo save failed: %s" % exc)

            previous_state = current_state
            time.sleep_ms(10)

    except KeyboardInterrupt:
        print("user stop")
    except Exception as exc:
        print("runtime error: %s" % exc)
    finally:
        if sensor is not None and sensor_started:
            sensor.stop()
        if display_started:
            Display.deinit()
        if media_started:
            MediaManager.deinit()
        print("photo program stopped")


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
