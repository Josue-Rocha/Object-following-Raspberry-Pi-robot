import pygame
import time
import cv2
import numpy as np
from sabertooth import Sabertooth
from ps5_controller import PS5_Controller
from picamera2 import Picamera2

def main():
    try:
        ps5 = PS5_Controller()
        ps5.initialize_controller()

        saber = Sabertooth()
        saber.set_ramping(22)
        isMoving = False

        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(1)

        lower_hsv = np.array([25, 115, 165])
        upper_hsv = np.array([40, 140, 230])

        last_cx = None
        last_w = None

        while True:
            frame = picam2.capture_array()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cx = x + w // 2

                    # Compare to previous frame
                    if last_cx is not None and last_w is not None:
                        dx = cx - last_cx
                        dw = w - last_w

                        turn = -dx * 0.1  # Negative because leftward dx means turn left
                        speed = -dw * 0.5  # Negative because shrinking box means move forward

                        # Clamp speed and turn to acceptable values
                        turn = max(min(turn, 100), -100)
                        speed = max(min(speed, 100), -100)

                        saber.drive(speed, turn)
                        isMoving = True
                    last_cx = cx
                    last_w = w
                else:
                    if isMoving:
                        saber.stop()
                        isMoving = False
                        last_cx = None
                        last_w = None
            else:
                if isMoving:
                    saber.stop()
                    isMoving = False
                    last_cx = None
                    last_w = None

            time.sleep(0.04)  # 25 FPS max, reasonable pace for control loop

    except KeyboardInterrupt:
        print("Exiting program...")

    finally:
        pygame.joystick.quit()
        pygame.quit()
        saber.close()
        picam2.stop()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
