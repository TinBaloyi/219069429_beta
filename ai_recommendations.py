"""
AI-powered recommendation engine for bottleneck analysis
Supports: OpenAI, Anthropic, and Ollama
"""
import os
import json
import requests
from typing import List, Dict, Any

class AIEngine:
    """Base class for AI recommendation engines"""
    
    def generate_recommendations(self, bottleneck: Dict[str, Any], max_recommendations: int = 5) -> List[Dict[str, str]]:
        """Generate recommendations for a bottleneck"""
        raise NotImplementedError

class OllamaEngine(AIEngine):
    """Ollama local AI engine"""
    
    def __init__(self, model: str = None, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        # Use environment variable or default to llama3.2:1b
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        
    def generate_recommendations(self, bottleneck: Dict[str, Any], max_recommendations: int = 5) -> List[Dict[str, str]]:
        """Generate recommendations using Ollama"""
        
        prompt = self._create_prompt(bottleneck, max_recommendations)
        
        try:
            # Correct Ollama API endpoint
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 200
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("response", "")
                return self._parse_recommendations(text, max_recommendations)
            else:
                raise Exception(f"Ollama request failed: {response.status_code}")
                
        except Exception as e:
            print(f"AI generation failed: {e}. Using fallback.")
            return self._fallback_recommendations(bottleneck, max_recommendations)
    
    def _create_prompt(self, bottleneck: Dict[str, Any], max_recommendations: int) -> str:
        """Create prompt for AI model"""
        return f"""You are an expert in manufacturing operations and process optimization. 

Analyze this bottleneck and provide {max_recommendations} specific, actionable recommendations:

Bottleneck Details:
- Type: {bottleneck.get('bottleneck_type', 'Unknown')}
- Entity: {bottleneck.get('entity', 'Unknown')}
- Severity Score: {bottleneck.get('severity_score', 0)}
- Metric: {bottleneck.get('metric', 'Unknown')}
- Value: {bottleneck.get('metric_value', 'Unknown')}
- Reason: {bottleneck.get('reason', 'Unknown')}

Provide recommendations in this EXACT JSON format:
[
  {{
    "priority": "High",
    "action": "Specific action to take",
    "verification": "How to verify it worked"
  }}
]

Requirements:
- Be specific and actionable
- Priority must be: High, Medium, or Low
- Keep actions concise (under 100 characters)
- Provide clear verification methods
- Return ONLY valid JSON, no other text"""
    
    def _parse_recommendations(self, text: str, max_recommendations: int) -> List[Dict[str, str]]:
        """Parse AI response into recommendations"""
        try:
            # Try to find JSON in the response
            start = text.find('[')
            end = text.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_text = text[start:end]
                recommendations = json.loads(json_text)
                
                # Validate and limit recommendations
                valid_recs = []
                for rec in recommendations[:max_recommendations]:
                    if all(k in rec for k in ['priority', 'action', 'verification']):
                        valid_recs.append(rec)
                
                if valid_recs:
                    return valid_recs
            
        except json.JSONDecodeError:
            pass
        
        # Fallback if parsing fails
        return self._fallback_recommendations({}, max_recommendations)
    
    def _fallback_recommendations(self, bottleneck: Dict[str, Any], max_recommendations: int) -> List[Dict[str, str]]:
        """Fallback recommendations when AI fails"""
        bottleneck_type = bottleneck.get('bottleneck_type', 'Unknown')
        
        recommendations = {
            'High Error Rate': [
                {'priority': 'High', 'action': 'Inspect quality control processes and gates', 'verification': 'Monitor error rate reduction'},
                {'priority': 'High', 'action': 'Review and update equipment calibration', 'verification': 'Check calibration logs'},
                {'priority': 'Medium', 'action': 'Analyze defect patterns and root causes', 'verification': 'Defect analysis report'},
                {'priority': 'Medium', 'action': 'Implement additional operator training', 'verification': 'Training completion rate'},
                {'priority': 'Low', 'action': 'Consider automation for error-prone steps', 'verification': 'ROI analysis'}
            ],
            'Low Throughput': [
                {'priority': 'High', 'action': 'Analyze and optimize workflow bottlenecks', 'verification': 'Throughput increase measurement'},
                {'priority': 'High', 'action': 'Review resource allocation and scheduling', 'verification': 'Capacity utilization reports'},
                {'priority': 'Medium', 'action': 'Investigate equipment performance issues', 'verification': 'Equipment efficiency metrics'},
                {'priority': 'Medium', 'action': 'Implement lean manufacturing principles', 'verification': 'Cycle time reduction'},
                {'priority': 'Low', 'action': 'Consider parallel processing options', 'verification': 'Throughput comparison'}
            ],
            'Long Lead Time': [
                {'priority': 'High', 'action': 'Map and optimize the entire process flow', 'verification': 'Lead time tracking'},
                {'priority': 'High', 'action': 'Identify and eliminate wait times', 'verification': 'Process time analysis'},
                {'priority': 'Medium', 'action': 'Review supplier and logistics performance', 'verification': 'Delivery time metrics'},
                {'priority': 'Medium', 'action': 'Implement just-in-time practices', 'verification': 'Inventory turnover rate'},
                {'priority': 'Low', 'action': 'Automate manual handoff points', 'verification': 'Handoff time reduction'}
            ]
        }
        
        return recommendations.get(bottleneck_type, [
            {'priority': 'High', 'action': 'Conduct root cause analysis', 'verification': 'Analysis report completion'},
            {'priority': 'Medium', 'action': 'Monitor key performance indicators', 'verification': 'KPI dashboard updates'},
            {'priority': 'Low', 'action': 'Document improvement actions taken', 'verification': 'Documentation review'}
        ])[:max_recommendations]

class OpenAIEngine(AIEngine):
    """OpenAI GPT engine"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
    
    def generate_recommendations(self, bottleneck: Dict[str, Any], max_recommendations: int = 5) -> List[Dict[str, str]]:
        """Generate recommendations using OpenAI"""
        try:
            import openai
            openai.api_key = self.api_key
            
            prompt = self._create_prompt(bottleneck, max_recommendations)
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in manufacturing operations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            text = response.choices[0].message.content
            return self._parse_recommendations(text, max_recommendations)
            
        except Exception as e:
            print(f"OpenAI generation failed: {e}. Using fallback.")
            return OllamaEngine()._fallback_recommendations(bottleneck, max_recommendations)
    
    def _create_prompt(self, bottleneck: Dict[str, Any], max_recommendations: int) -> str:
        return OllamaEngine()._create_prompt(bottleneck, max_recommendations)
    
    def _parse_recommendations(self, text: str, max_recommendations: int) -> List[Dict[str, str]]:
        return OllamaEngine()._parse_recommendations(text, max_recommendations)

class AnthropicEngine(AIEngine):
    """Anthropic Claude engine"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
    
    def generate_recommendations(self, bottleneck: Dict[str, Any], max_recommendations: int = 5) -> List[Dict[str, str]]:
        """Generate recommendations using Anthropic"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            prompt = self._create_prompt(bottleneck, max_recommendations)
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            text = response.content[0].text
            return self._parse_recommendations(text, max_recommendations)
            
        except Exception as e:
            print(f"Anthropic generation failed: {e}. Using fallback.")
            return OllamaEngine()._fallback_recommendations(bottleneck, max_recommendations)
    
    def _create_prompt(self, bottleneck: Dict[str, Any], max_recommendations: int) -> str:
        return OllamaEngine()._create_prompt(bottleneck, max_recommendations)
    
    def _parse_recommendations(self, text: str, max_recommendations: int) -> List[Dict[str, str]]:
        return OllamaEngine()._parse_recommendations(text, max_recommendations)

def get_ai_engine() -> AIEngine:
    """Get the configured AI engine"""
    provider = os.getenv("AI_PROVIDER", "none").lower()
    
    if provider == "ollama":
        return OllamaEngine()
    elif provider == "openai":
        return OpenAIEngine()
    elif provider == "anthropic":
        return AnthropicEngine()
    else:
        # Return Ollama with fallback-only mode
        return OllamaEngine()

def generate_recommendations_for_event(event: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
    """Main entry point for generating recommendations"""
    max_recommendations = kwargs.get('max_recommendations', 5)
    
    try:
        engine = get_ai_engine()
        return engine.generate_recommendations(event, max_recommendations)
    except Exception as e:
        print(f"Failed to generate recommendations: {e}")
        return OllamaEngine()._fallback_recommendations(event, max_recommendations)