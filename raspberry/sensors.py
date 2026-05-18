import random


class SensorReader:
    def __init__(self, pins):
        self.pins = pins
        self.dht = None
        self.light = None
        self.rain_input = None
        try:
            import board
            import adafruit_dht

            self.dht = adafruit_dht.DHT22(getattr(board, f"D{pins['dht']}"))
        except Exception as exc:
            print(f"[WARN] DHT22 en mode simulation: {exc}")
        try:
            import board
            import adafruit_bh1750

            self.light = adafruit_bh1750.BH1750(board.I2C())
        except Exception as exc:
            print(f"[WARN] BH1750 en mode simulation: {exc}")
        try:
            from gpiozero import DigitalInputDevice

            self.rain_input = DigitalInputDevice(pins["rain_digital"])
        except Exception as exc:
            print(f"[WARN] Capteur pluie en mode simulation: {exc}")

    def read(self):
        if self.dht:
            temperature = float(self.dht.temperature)
            humidity = float(self.dht.humidity)
        else:
            temperature = random.uniform(22, 33)
            humidity = random.uniform(50, 88)

        luminosity = float(self.light.lux) if self.light else random.uniform(250, 850)
        if self.rain_input:
            precipitation = 22.0 if not self.rain_input.value else 0.0
        else:
            precipitation = random.choice([0.0, 2.0, 6.0, 20.0])

        return {
            "temperature": round(temperature, 1),
            "humidity": round(humidity, 1),
            "precipitation": round(precipitation, 1),
            "luminosity": round(luminosity, 1),
            "soil_moisture": 0.0,
            "reservoir_level": 0.0,
        }
