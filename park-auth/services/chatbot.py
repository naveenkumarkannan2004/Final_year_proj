"""
Basic rule-based chatbot for park activity guidance
"""
import re
from typing import List, Tuple

class ParkChatbot:
    """Simple rule-based chatbot for park guidance"""
    
    def __init__(self):
        """Initialize chatbot with predefined responses"""
        self.patterns_responses = [
            # Greetings
            (r'\b(hi|hello|hey|greetings)\b', [
                "Hello! I'm here to help you with park activity monitoring. How can I assist you?",
                "Hi there! Ask me anything about park rules and authorized activities.",
                "Hey! I can help you understand what activities are allowed in the park."
            ]),
            
            # Authorized activities
            (r'\b(authorized|allowed|permitted|can i|what.*allowed)\b', [
                "Authorized activities in the park include: walking, jogging, cycling on designated paths, picnicking in designated areas, playing in playgrounds, and enjoying nature responsibly.",
                "You're allowed to: exercise, walk pets on leash, use designated sports areas, take photos, and relax in open spaces.",
                "Permitted activities: recreational walking, supervised children's play, organized sports in designated areas, and peaceful enjoyment of park facilities."
            ]),
            
            # Unauthorized activities
            (r'\b(unauthorized|not allowed|prohibited|forbidden|banned)\b', [
                "Unauthorized activities include: vandalism, littering, unauthorized vehicles, loud music after hours, damaging plants/property, and any dangerous behavior.",
                "Prohibited activities: destruction of property, unauthorized camping, fires outside designated areas, harassment, and activities that disturb others.",
                "Not allowed: graffiti, unauthorized construction, hunting, motorized vehicles (except authorized), and any illegal activities."
            ]),
            
            # How to use the system
            (r'\b(how.*use|how.*work|upload|detect)\b', [
                "To use the system: 1) Upload an image or video using the file uploader, 2) The AI will analyze it for authorized/unauthorized activities, 3) View results and any flagged unauthorized activities in the separate section below.",
                "Simply upload your media file (image or video), and our YOLO-based AI will automatically detect and classify activities as authorized or unauthorized.",
                "Upload process: Click the file uploader → Select your image/video → Wait for AI analysis → Review detection results and any alerts."
            ]),
            
            # Detection accuracy
            (r'\b(accurate|accuracy|reliable|trust|confidence)\b', [
                "Our system uses a trained YOLO model with confidence thresholds. Detections above 50% confidence are shown. Higher confidence scores indicate more reliable detections.",
                "The AI model has been trained specifically for park activities. Check the confidence score for each detection - higher scores mean more reliable results.",
                "Detection reliability depends on image quality, lighting, and camera angle. Clear, well-lit footage produces the most accurate results."
            ]),
            
            # Video processing
            (r'\b(video|frame|extract|clip)\b', [
                "For videos, the system processes each frame and extracts frames containing unauthorized activities. You can review these in the 'Unauthorized Activity Frames' section.",
                "Video analysis: Each frame is analyzed individually. Frames with unauthorized activities are automatically extracted and displayed for review.",
                "The system scans your entire video and creates a collection of frames showing any detected unauthorized activities."
            ]),
            
            # Park rules
            (r'\b(rules|regulations|guidelines|policy)\b', [
                "Park rules: Respect nature, stay on paths, dispose of waste properly, keep pets leashed, respect quiet hours, and report any suspicious activities.",
                "General guidelines: Be considerate of others, follow posted signs, use facilities as intended, and help keep the park clean and safe.",
                "Key policies: No littering, respect wildlife, use designated areas for specific activities, and maintain a family-friendly environment."
            ]),
            
            # Help
            (r'\b(help|support|assist|guide)\b', [
                "I can help you with: understanding authorized/unauthorized activities, using the detection system, interpreting results, and learning about park rules.",
                "Need assistance? Ask me about: what's allowed in the park, how to use the upload feature, what the AI detects, or general park guidelines.",
                "I'm here to guide you through the park monitoring system and answer questions about park activities and rules."
            ]),
            
            # Thanks
            (r'\b(thank|thanks|appreciate)\b', [
                "You're welcome! Feel free to ask if you have more questions.",
                "Happy to help! Let me know if you need anything else.",
                "My pleasure! I'm here whenever you need assistance."
            ]),
            
            # Goodbye
            (r'\b(bye|goodbye|see you|exit)\b', [
                "Goodbye! Have a great day and enjoy the park responsibly!",
                "See you later! Remember to follow park rules.",
                "Take care! Feel free to come back if you have more questions."
            ])
        ]
        
        self.default_responses = [
            "I'm not sure I understand. Could you ask about authorized activities, park rules, or how to use the detection system?",
            "I can help with questions about park activities, system usage, or detection results. What would you like to know?",
            "Try asking about: What activities are allowed? How do I upload a video? What does unauthorized mean?",
            "I'm here to help with park monitoring questions. Ask me about authorized activities, system features, or park guidelines."
        ]
    
    def get_response(self, user_message: str) -> str:
        """
        Get chatbot response based on user message
        
        Args:
            user_message: User's input message
            
        Returns:
            Chatbot response string
        """
        user_message = user_message.lower().strip()
        
        # Check each pattern
        for pattern, responses in self.patterns_responses:
            if re.search(pattern, user_message, re.IGNORECASE):
                import random
                return random.choice(responses)
        
        # Default response if no pattern matches
        import random
        return random.choice(self.default_responses)
    
    def get_quick_replies(self) -> List[str]:
        """Get suggested quick reply buttons"""
        return [
            "What activities are authorized?",
            "How do I use this system?",
            "What is prohibited in the park?",
            "How accurate is the detection?"
        ]
