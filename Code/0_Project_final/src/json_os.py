class json_os:
    
    def __init__(self):
        pass
    
    def read_json(self, path):
        with open(path, "r") as file:
            return json.load(file)
        
    def write_json(self, data, path):
        pass

    def save_json(self, data, path):
        pass