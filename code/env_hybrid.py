"""
Hybrid Environment: Switch between simulation and CARLA
Allows seamless switching for comparison
"""

import numpy as np
import carla
from config import *


class VehicleEnvironmentHybrid:
    """
    Hybrid environment supporting both simulation and CARLA
    Use mode='sim' for quick testing, mode='carla' for realistic validation
    """
    
    def __init__(self, mode='sim', carla_host='localhost', carla_port=2000):
        """
        Initialize hybrid environment
        
        Args:
            mode: 'sim' (simulation) or 'carla' (real simulator)
            carla_host: CARLA server host
            carla_port: CARLA server port
        """
        self.mode = mode
        self.current_state = None
        self.collision_count = 0
        self.distance_to_obstacle = 100.0
        
        if mode == 'carla':
            self._init_carla(carla_host, carla_port)
        else:
            self._init_simulation()
    
    def _init_simulation(self):
        """Initialize simple simulation"""
        self.speed = 0
        self.distance = 1
        self.emotion = EMOTION_CALM
        print("✓ Simulation mode initialized")
    
    def _init_carla(self, host, port):
        """Initialize CARLA connection"""
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.vehicle = None
            self.collision_sensor = None
            self.radar_sensor = None
            print(f"✓ CARLA mode initialized (connected to {host}:{port})")
        except Exception as e:
            print(f"✗ CARLA initialization failed: {e}")
            print("  Falling back to simulation mode")
            self.mode = 'sim'
            self._init_simulation()
    
    def reset(self):
        """Reset environment"""
        if self.mode == 'sim':
            return self._reset_simulation()
        else:
            return self._reset_carla()
    
    def _reset_simulation(self):
        """Reset simulation mode"""
        self.speed = SPEED_LOW
        self.distance = OBSTACLE_FAR
        self.emotion = EMOTION_CALM
        self.collision_count = 0
        self.current_state = (self.speed, self.distance, self.emotion)
        return self.current_state
    
    def _reset_carla(self):
        """Reset CARLA environment"""
        if self.vehicle:
            self.vehicle.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.radar_sensor:
            self.radar_sensor.destroy()
        
        # Spawn new vehicle
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Attach sensors
        collision_bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_count = 0
        
        radar_bp = bp_lib.find('sensor.other.radar')
        self.radar_sensor = self.world.spawn_actor(
            radar_bp,
            carla.Transform(carla.Location(x=2.5)),
            attach_to=self.vehicle
        )
        
        self.distance_to_obstacle = 100.0
        self.emotion = EMOTION_CALM
        
        return self._get_state()
    
    def step(self, action):
        """Execute action"""
        if self.mode == 'sim':
            return self._step_simulation(action)
        else:
            return self._step_carla(action)
    
    def _step_simulation(self, action):
        """Simulation mode step"""
        # Stochastic transitions
        if np.random.rand() < 0.3:
            self.distance = 1 - self.distance
        
        if np.random.rand() < 0.2:
            self.distance = 1
        
        stress = self.speed + (1 - self.distance)
        if stress >= 2 and np.random.rand() < 0.5:
            self.emotion = min(self.emotion + 1, EMOTION_OVERLOADED)
        
        # Reward
        if self.speed == SPEED_LOW or self.distance == OBSTACLE_FAR:
            reward = REWARD_SAFE_STATE
        else:
            reward = REWARD_UNSAFE_STATE
        
        next_state = (self.speed, self.distance, self.emotion)
        self.current_state = next_state
        
        return next_state, reward, False
    
    def _step_carla(self, action):
        """CARLA mode step"""
        # Apply action
        control = carla.VehicleControl()
        
        if action == ACTION_LOW_ALERT:
            control.throttle = 0.3
            control.brake = 0.0
        elif action == ACTION_MEDIUM_ALERT:
            control.throttle = 0.2
            control.brake = 0.1
        elif action == ACTION_HIGH_ALERT:
            control.throttle = 0.0
            control.brake = 0.4
        elif action == ACTION_ASSISTANCE:
            control.throttle = 0.0
            control.brake = 0.3
        
        self.vehicle.apply_control(control)
        self.world.tick()
        
        # Get state
        next_state = self._get_state()
        
        # Compute reward
        speed_state, dist_state, emotion = next_state
        
        if speed_state == SPEED_LOW or dist_state == OBSTACLE_FAR:
            reward = REWARD_SAFE_STATE
        else:
            reward = REWARD_UNSAFE_STATE
        
        # Collision penalty
        reward -= self.collision_count * 5
        self.collision_count = 0
        
        return next_state, reward, False
    
    def _get_state(self):
        """Get current state in CARLA mode"""
        # Speed
        vel = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        
        if speed_kmh < 30:
            speed_state = SPEED_LOW
        elif speed_kmh < 80:
            speed_state = SPEED_MEDIUM
        else:
            speed_state = SPEED_HIGH
        
        # Distance
        distance_state = (
            OBSTACLE_FAR if self.distance_to_obstacle >= 10 
            else OBSTACLE_NEAR
        )
        
        # Emotion
        stress = speed_state + (1 - distance_state)
        if stress >= 2 and np.random.rand() < 0.5:
            self.emotion = min(self.emotion + 1, EMOTION_OVERLOADED)
        
        return (speed_state, distance_state, self.emotion)
    
    def close(self):
        """Cleanup"""
        if self.mode == 'carla' and self.vehicle:
            self.vehicle.destroy()
            if self.collision_sensor:
                self.collision_sensor.destroy()
            if self.radar_sensor:
                self.radar_sensor.destroy()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HYBRID ENVIRONMENT TEST")
    print("="*60)
    
    # Test simulation mode
    print("\n[Test 1] Simulation Mode")
    env_sim = VehicleEnvironmentHybrid(mode='sim')
    state = env_sim.reset()
    print(f"  Initial state: {state}")
    
    for i in range(5):
        action = np.random.randint(0, 4)
        next_state, reward, done = env_sim.step(action)
        print(f"  Step {i+1}: action={action}, state={next_state}, reward={reward}")
    
    env_sim.close()
    
    # Test CARLA mode (if available)
    print("\n[Test 2] CARLA Mode")
    try:
        env_carla = VehicleEnvironmentHybrid(mode='carla')
        state = env_carla.reset()
        print(f"  Initial state: {state}")
        
        for i in range(5):
            action = np.random.randint(0, 4)
            next_state, reward, done = env_carla.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.2f}")
        
        env_carla.close()
    except Exception as e:
        print(f"  CARLA mode skipped: {e}")
    
    print("\n✓ Hybrid environment test complete\n")
