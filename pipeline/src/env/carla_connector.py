import carla
import yaml


class CarlaConnector:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['carla']
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.map = None
        
    def connect(self):  # connect to carla
        try:
            # Create client
            self.client = carla.Client(
                self.config['host'], 
                self.config['port']
            )
            
            # Set timeout
            self.client.set_timeout(self.config['timeout'])
            
            # Get world
            self.world = self.client.get_world()
            
            # Get blueprint library
            self.blueprint_library = self.world.get_blueprint_library()
            
            # Get map
            self.map = self.world.get_map()
            
            # Configure synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = self.config['synchronous_mode']
            settings.fixed_delta_seconds = self.config['fixed_delta_seconds']
            self.world.apply_settings(settings)
            
            return True
            
        except RuntimeError as e:
            raise ConnectionError(f"Failed to connect to CARLA server: {e}")
    
    def get_world(self): # return world objs
        if self.world is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self.world
    
    def get_blueprint_library(self): # blueprint lib
        if self.blueprint_library is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self.blueprint_library
    
    def get_map(self): # return map from API
        if self.map is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self.map
