import torch
import numpy as np
from src.simulation.weather_models import FogSimulator, RainSimulator, SnowSimulator
from src.utils.difficulty import calculate_difficulty

class CurriculumSampler:
    """
    Manages the curriculum by determining probability of sampling 
    different difficulty levels (lambdas) based on training progress (epoch).
    """
    def __init__(self, mode='linear', total_epochs=50):
        self.mode = mode
        self.total_epochs = total_epochs
        
    def get_difficulty_lambda(self, current_epoch):
        """
        Returns the difficulty scalar lambda (0.0 to 1.0) 
        appropriate for the current epoch.
        """
        progress = current_epoch / self.total_epochs
        
        if self.mode == 'linear':
            # Difficulty increases linearly from 0 to 1
            return min(1.0, progress)
        elif self.mode == 'step':
            # Steps every 20%
            return np.floor(progress * 5) / 5.0
        elif self.mode == 'random':
            # Random difficulty each time (Baseline: Standard Augmentation)
            return np.random.rand()
        elif self.mode == 'clear_only':
            return 0.0
        else:
            return 1.0 # Full difficulty constant

class WeatherAugmentor:
    """
    Wrapper to apply weather based on difficulty lambda.
    """
    def __init__(self):
        self.fog = FogSimulator()
        self.rain = RainSimulator()
        self.snow = SnowSimulator()
    
    def apply(self, image, depth, difficulty_lambda, weather_type='random'):
        """
        Apply weather with severity = difficulty_lambda.
        """
        if difficulty_lambda <= 0.05:
            return image # Clear enough
            
        if weather_type == 'random':
            weather_type = np.random.choice(['rain', 'fog', 'snow'])
            
        if weather_type == 'fog':
            # Map lambda [0, 1] to physical beta
            beta = difficulty_lambda * 0.1
            return self.fog.add_fog(image, depth, beta=beta)
            
        elif weather_type == 'rain':
            # Rain rate [0, 1.0]
            return self.rain.add_rain(image, depth, rainfall_rate=difficulty_lambda)

        elif weather_type == 'snow':
            # Snow rate [0, 1.0]
            return self.snow.add_snow(image, depth, snow_rate=difficulty_lambda)
            
        return image
