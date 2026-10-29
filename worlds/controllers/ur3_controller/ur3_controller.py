# Physical AI応用1講座 最終課題用コード
# Webotで動作するコードです
# りんごを転がして遊んでください

from controller import Robot
from ultralytics import YOLO

import numpy as np
import cv2

# =====================================
# Robot
# =====================================

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# =====================================
# Camera
# =====================================

camera = robot.getDevice("camera")
camera.enable(timestep)

# =====================================
# RangeFinder
# =====================================

rangefinder = robot.getDevice("range-finder")
rangefinder.enable(timestep)

# =====================================
# Joints
# =====================================

pan = robot.getDevice("shoulder_pan_joint")
lift = robot.getDevice("shoulder_lift_joint")
elbow = robot.getDevice("elbow_joint")

w1 = robot.getDevice("wrist_1_joint")
w2 = robot.getDevice("wrist_2_joint")
w3 = robot.getDevice("wrist_3_joint")

# =====================================
# Gripper
# =====================================

left_finger = robot.getDevice(
    "ROBOTIQ 2F-85 Gripper::left finger joint"
)

right_finger = robot.getDevice(
    "ROBOTIQ 2F-85 Gripper::right finger joint"
)

# =====================================
# Initial Pose
# =====================================

pan.setPosition(-0.44)
lift.setPosition(-0.80)
elbow.setPosition(1.4)

# 真下
#w1.setPosition(-0.9)
#w2.setPosition(0.0)
w3.setPosition(-0.5)

left_finger.setPosition(0.0)
right_finger.setPosition(0.0)

# =====================================
# YOLO
# =====================================

model = YOLO(
    './model/yolov8n.pt'
)

width = camera.getWidth()
height = camera.getHeight()

# =====================================
# FSM
# =====================================

ALIGN = 0
GRAB  = 1
LIFT  = 2
DONE  = 3

state = ALIGN

counter = 0

print("READY")

while robot.step(timestep) != -1:

    image = camera.getImage()

    frame = np.frombuffer(
        image,
        dtype=np.uint8
    ).reshape(
        (height, width, 4)
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGRA2BGR
    )

    depth = np.array(
        rangefinder.getRangeImage(),
        dtype=np.float32
    )

    depth = depth.reshape(
        (
            rangefinder.getHeight(),
            rangefinder.getWidth()
        )
    )

    results = model(
        frame,
        conf=0.25,
        verbose=False
    )

    annotated = results[0].plot()

    target_box = None
    largest_area = 0

    for box in results[0].boxes:

        cls = int(box.cls[0])

        label = model.names[cls]

        if label != "apple":
            continue

        conf = float(box.conf[0])

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = box.xyxy[0]

        area = (x2 - x1) * (y2 - y1)

        if area > largest_area:

            largest_area = area
            target_box = box

    if target_box is None:

        print("APPLE NOT FOUND")

        cv2.imshow(
            "APPLE_TRACK",
            annotated
        )

        cv2.waitKey(1)

        continue

    x1, y1, x2, y2 = target_box.xyxy[0]

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    dx = int(
        cx * depth.shape[1] / width
    )

    dy = int(
        cy * depth.shape[0] / height
    )

    distance = depth[dy, dx]

    if not np.isfinite(distance):

        print("APPLE DEPTH INVALID")

        continue

    error_x = cx - width // 2
    error_y = cy - height // 2

    print(
        f"STATE={state} "
        f"cx={cx} "
        f"cy={cy} "
        f"dist={distance:.3f}"
    )

    # =====================================
    # ALIGN
    # =====================================

    if state == ALIGN:

        # ---------------------------------
        # 画面誤差
        # ---------------------------------

        error_x = cx - width // 2
        error_y = cy - height // 2

        # ---------------------------------
        # 左右追尾
        # ---------------------------------

        pan_target = (
            -0.44
            - error_x * 0.001
        )

        # ---------------------------------
        # 上下追尾
        # ---------------------------------

        lift_target = (
            -0.80
            + error_y * 0.001
        )

        # ---------------------------------
        # 肘補正
        # ---------------------------------

        elbow_target = (
            1.40
            - error_y * 0.001
        )



        # ---------------------------------
        # 制限
        # ---------------------------------

        #pan_target = np.clip(
        #    pan_target,
        #    -2.8,
        #    2.8
        #)

        lift_target = np.clip(
            lift_target,
            -2.5,
            0.0
        )

        #elbow_target = np.clip(
        #    elbow_target,
        #    0.5,
        #    2.3
        #)

        # ---------------------------------
        # 反映
        # ---------------------------------

        pan.setPosition(
            pan_target
        )

        lift.setPosition(
            lift_target
        )

        elbow.setPosition(
            elbow_target
        )



        # 真下向き維持

        #w1.setPosition(1.57)
        #w2.setPosition(0.0)
        #w3.setPosition(0.0)

        print(
            f"cx={cx} "
            f"cy={cy} "
            f"ex={error_x} "
            f"ey={error_y} "
            f"dist={distance:.3f}"
        )

        if (
            abs(error_x) < 70
            and
            abs(error_y) < 100
            and
            distance <= 0.15
        ):

            print("ALIGN OK")

            state = GRAB
            counter = 0
        
    # =====================================
    # GRAB
    # =====================================

    elif state == GRAB:

        if counter == 0:

            print("GRAB")

            left_finger.setPosition(
                0.8
            )

            right_finger.setPosition(
                0.8
            )

        counter += 1

        if counter > 100:

            state = LIFT

    # =====================================
    # LIFT
    # =====================================

    elif state == LIFT:

        print("LIFT")

        lift.setPosition(
            -0.50
        )

        state = DONE

    # =====================================
    # DONE
    # =====================================

    elif state == DONE:

        pass

    cv2.imshow(
        "APPLE_TRACK",
        annotated
    )

    cv2.waitKey(1)