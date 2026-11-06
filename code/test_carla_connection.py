"""
CARLA 0.9.15 Connection Test
Verifies CARLA server is accessible and functional
"""

import carla
import sys


def test_carla_connection(host='localhost', port=2000):
    """Test CARLA server connection"""
    try:
        print("\n" + "="*60)
        print("CARLA 0.9.15 CONNECTION TEST")
        print("="*60)
        
        # Connect to CARLA
        print(f"\n[1] Connecting to CARLA at {host}:{port}...")
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        
        # Get world
        print("[2] Loading world...")
        world = client.get_world()
        
        # Get map info
        map_name = world.get_map().name
        print(f"[3] World loaded: {map_name}")
        
        # Test spawning a vehicle
        print("[4] Testing vehicle spawn...")
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"[5] Vehicle spawned at {spawn_point.location}")
        
        # Test sensors
        print("[6] Testing collision sensor...")
        sensor_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(
            sensor_bp, 
            carla.Transform(), 
            attach_to=vehicle
        )
        print("[7] Collision sensor attached")
        
        # Cleanup
        print("[8] Cleaning up...")
        collision_sensor.destroy()
        vehicle.destroy()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - CARLA 0.9.15 IS READY")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ CONNECTION FAILED: {e}")
        print("\nMake sure:")
        print("  1. CARLA server is running")
        print("  2. Server address is correct (default: localhost:2000)")
        print("  3. Firewall allows port 2000")
        return False


if __name__ == "__main__":
    success = test_carla_connection()
    sys.exit(0 if success else 1)
