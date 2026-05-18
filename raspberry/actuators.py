class PlateActuators:
    def __init__(self, pins, servo_angles=None):
        self.pins = pins
        self.servo_angles = servo_angles or {"repos": 0, "chaleur": 90, "pluie": 55}
        self.enabled = False
        self.left = self.right = self.camera = None
        try:
            from gpiozero import AngularServo

            self.left = AngularServo(pins["servo_left"], min_angle=0, max_angle=90)
            self.right = AngularServo(pins["servo_right"], min_angle=0, max_angle=90)
            self.camera = AngularServo(pins.get("servo_camera", pins["servo_left"]), min_angle=0, max_angle=90)
            self.enabled = True
        except Exception as exc:
            print(f"[WARN] Servos en mode simulation: {exc}")

    def apply(self, mode):
        angle = self.servo_angles.get(mode, 0)
        if mode == "pluie":
            left_angle, right_angle, camera_angle = angle, angle, 35
        elif mode == "chaleur":
            left_angle, right_angle, camera_angle = 90, 90, 20
        else:
            left_angle, right_angle, camera_angle = 0, 0, 0

        if self.enabled:
            self.left.angle = left_angle
            self.right.angle = right_angle
            self.camera.angle = camera_angle
        print(f"[ACTUATOR] mode={mode} left={left_angle} right={right_angle} camera={camera_angle}")
