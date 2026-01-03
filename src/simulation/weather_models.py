import numpy as np
import cv2

class FogSimulator:
    def __init__(self):
        pass

    def add_fog(self, image, depth_map, beta=0.01, atmospheric_light=None):
        """
        Implements Eq 1 and 2 from the paper:
        I_fog(x) = J(x)t(x) + A(1-t(x))
        t(x) = e^(-beta * d(x))
        """
        # Normalize image to 0-1 float
        img_float = image.astype(np.float32) / 255.0
        
        # Estimate atmospheric light A if not provided (usually max intensity pixel or constant)
        if atmospheric_light is None:
            # Simple assumption: brightest pixel or standard gray
            atmospheric_light = 1.0 # Pure white fog light
            
        # 1. Calculate Transmission Map t(x) = e^(-beta * d(x))
        # Ensure depth map is in meters or consistent units. 
        # Avoid zero depth to prevent issues? (Usually depth is > 0)
        transmission = np.exp(-beta * depth_map)
        
        # Expand transmission to 3 channels for broadcasting
        transmission = np.expand_dims(transmission, axis=-1)
        
        # 2. Apply Fog Equation
        # I_fog = J * t + A * (1 - t)
        foggy_image = img_float * transmission + atmospheric_light * (1 - transmission)
        
        # Clip to 0-1 and convert back to uint8
        foggy_image = np.clip(foggy_image, 0, 1) * 255
        return foggy_image.astype(np.uint8)

class RainSimulator:
    def __init__(self):
        pass

    def add_rain(self, image, depth_map, rainfall_rate=0.5):
        """
        Implements the rain model described:
        I_rain = T_rain * I_clear + A_rain
        Simplified implementation for volumetric rain + streaks.
        
        For this implementation, we will approximate:
        1. Volumetric attenuation (similar to fog but weaker)
        2. Streaks (random lines)
        """
        rows, cols, _ = image.shape
        img_float = image.astype(np.float32) / 255.0
        
        # --- 1. Volumetric Attenuation (Fog-like rain) ---
        # Rain causes less scattering than fog usually, but we model it similarly
        # derived from rainfall_rate
        beta_rain = 0.01 * rainfall_rate 
        transmission = np.exp(-beta_rain * depth_map)
        transmission = np.expand_dims(transmission, axis=-1)
        
        # Airlight for rain (A_rain) - usually grayish
        A_rain = 0.8 
        volumetric_rain = img_float * transmission + A_rain * (1 - transmission)
        
        # --- 2. Discrete Streaks ---
        # Add streaks based on rainfall_rate
        # Higher rate -> more streaks, longer streaks
        streak_image = self._generate_streaks(rows, cols, rainfall_rate)
        
        # Blend streaks: I_final = I_volumetric + I_streaks
        # Usually streaks are additive light or alpha blended
        final_image = volumetric_rain + streak_image
        
        final_image = np.clip(final_image, 0, 1) * 255
        return final_image.astype(np.uint8)

    def _generate_streaks(self, rows, cols, intensity):
        """
        Generates a layer of rain streaks.
        """
        # Create noise
        noise = np.random.rand(rows, cols)
        
        # Threshold to get drops (fewer drops for low intensity)
        # Intensity 0 to 1
        threshold = 1.0 - (0.01 + intensity * 0.05) 
        drops = np.zeros((rows, cols))
        drops[noise > threshold] = 1
        
        # Blur vertically to create streaks
        streak_length = int(10 + intensity * 20)
        kernel = np.zeros((streak_length, 1))
        kernel.fill(1.0 / streak_length)
        
        streaks = cv2.filter2D(drops, -1, kernel)
        
        # Enhance streak visibility
        streaks = streaks * 5.0 # Scale brightness
        streaks = np.expand_dims(streaks, axis=-1)
        return streaks

class SnowSimulator:
    def __init__(self):
        pass

    def add_snow(self, image, depth_map, snow_rate=0.5):
        """
        Simulates snow effect.
        1. Snowflake particles (similar to rain but larger, brighter, and different motion blur)
        2. Screen whitening / Contrast reduction
        """
        rows, cols, _ = image.shape
        img_float = image.astype(np.float32) / 255.0
        
        # 1. Whitening / Haze (Snow makes scenes brighter/whiter)
        # Mix original image with White based on depth (atmospheric scattering of snow)
        # Low snow_rate = little haze, High = generic whiteout
        beta_snow = 0.005 + snow_rate * 0.02
        transmission = np.exp(-beta_snow * depth_map)
        transmission = np.expand_dims(transmission, axis=-1)
        
        atmospheric_light = 1.0 # White
        snowy_bg = img_float * transmission + atmospheric_light * (1 - transmission)
        
        # 2. Snow Flakes
        # Random noise, larger particles
        noise = np.random.rand(rows, cols)
        threshold = 1.0 - (0.005 + snow_rate * 0.01) # Fewer particles than rain
        snow_mask = np.zeros((rows, cols))
        snow_mask[noise > threshold] = 1
        
        # Gaussian blur to make them look like soft flakes (not sharp points)
        snow_flakes = cv2.GaussianBlur(snow_mask, (3, 3), 0)
        snow_flakes = np.expand_dims(snow_flakes, axis=-1) * 2.0 # Brightness
        
        final_image = snowy_bg + snow_flakes
        final_image = np.clip(final_image, 0, 1) * 255
        
        return final_image.astype(np.uint8)
