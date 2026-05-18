import time


CROP_THRESHOLDS = {
    "tomato": {"temp_on": 29.0, "temp_off": 26.0, "humidity_on": 82.0, "humidity_off": 72.0, "rain_on": 18.0, "rain_off": 8.0, "luminosity_on": 720.0},
    "pepper": {"temp_on": 30.0, "temp_off": 27.0, "humidity_on": 80.0, "humidity_off": 70.0, "rain_on": 16.0, "rain_off": 7.0, "luminosity_on": 700.0},
    "potato": {"temp_on": 26.0, "temp_off": 23.0, "humidity_on": 85.0, "humidity_off": 75.0, "rain_on": 20.0, "rain_off": 9.0, "luminosity_on": 600.0},
    "default": {"temp_on": 29.0, "temp_off": 26.0, "humidity_on": 80.0, "humidity_off": 70.0, "rain_on": 18.0, "rain_off": 8.0, "luminosity_on": 650.0},
}


class HysteresisController:
    def __init__(self, crop="default", delay_seconds=8):
        self.crop = crop if crop in CROP_THRESHOLDS else "default"
        self.mode = "repos"
        self.last_change = 0
        self.delay_seconds = delay_seconds

    def decide(self, sensors, disease_command=None):
        th = CROP_THRESHOLDS[self.crop]
        rain_alert = sensors["precipitation"] >= th["rain_on"] or sensors["humidity"] >= th["humidity_on"]
        heat_alert = sensors["temperature"] >= th["temp_on"] or sensors["luminosity"] >= th["luminosity_on"]
        clear_rain = sensors["precipitation"] <= th["rain_off"] and sensors["humidity"] <= th["humidity_off"]
        clear_heat = sensors["temperature"] <= th["temp_off"] and sensors["luminosity"] < th["luminosity_on"] - 80

        disease_mode = self._disease_climate_mode(disease_command, heat_alert, rain_alert)
        if disease_mode:
            target = disease_mode
        elif self.mode == "pluie" and not clear_rain:
            target = "pluie"
        elif rain_alert:
            target = "pluie"
        elif self.mode == "chaleur" and not clear_heat:
            target = "chaleur"
        elif heat_alert:
            target = "chaleur"
        else:
            target = "repos"

        if target != self.mode and time.time() - self.last_change >= self.delay_seconds:
            self.mode = target
            self.last_change = time.time()
        return self.mode

    def _disease_climate_mode(self, disease_command, heat_alert, rain_alert):
        if not disease_command:
            return None
        if time.time() - disease_command.get("timestamp", 0) > disease_command.get("ttl_seconds", 1800):
            return None
        mode = disease_command.get("recommended_mechanism", "repos")
        heat_active = disease_command.get("heat_active", False) and heat_alert
        rain_active = disease_command.get("rain_active", False) and rain_alert
        risk = disease_command.get("instant_risk", 0)
        if risk < 0.70:
            return None
        if mode == "pluie" and rain_active:
            return "pluie"
        if mode == "chaleur" and heat_active:
            return "chaleur"
        return None
