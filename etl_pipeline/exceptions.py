class DataIngestionError(Exception):
    #Custom exception for data ingestion errors
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class DataTransformationError(Exception):
    #Custom exception for data transformation errors
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class ModelTrainingError(Exception):
    #Custom exception for model training errors
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
