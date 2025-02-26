from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger

class RedditPostValidator:
    def __init__(self):
        self.logger = ValidatorLogger()
        
        # Create validators for required string fields
        self.title_validator = (
            NotEmptyValidator("title", self.logger)
            .set_next(OnlyStringValidator("title", self.logger))
        )
        
        self.selftext_validator = (
            NotEmptyValidator("selftext", self.logger)
            .set_next(OnlyStringValidator("selftext", self.logger))
        )
        
        self.name_validator = (
            NotEmptyValidator("name", self.logger)
            .set_next(OnlyStringValidator("name", self.logger))
        )

    def validate_post(self, post):
        """Validate a single Reddit post"""
        # Validate required fields
        title_result = self.title_validator.validate(post)
        if "error" in title_result:
            return title_result
            
        selftext_result = self.selftext_validator.validate(post)
        if "error" in selftext_result:
            return selftext_result
            
        name_result = self.name_validator.validate(post)
        if "error" in name_result:
            return name_result
            
        return {"success": True}

    def validate_request(self, request_data):
        """Validate the entire request"""
        if not isinstance(request_data, dict) or "posts" not in request_data:
            return {"error": "Request must contain 'posts' field"}
            
        if not isinstance(request_data["posts"], list):
            return {"error": "'posts' must be a list"}
            
        # Validate each post
        for i, post in enumerate(request_data["posts"]):
            result = self.validate_post(post)
            if "error" in result:
                return {"error": f"Post {i}: {result['error']}"}
                
        return {"success": True} 