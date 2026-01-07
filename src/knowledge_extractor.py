"""Knowledge extraction and judgment system"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI
from config import get_config


@dataclass
class ExtractedKnowledge:
    """Represents extracted knowledge"""
    content: str
    confidence: float
    source: str
    metadata: Dict
    should_store: bool
    reason: str


class KnowledgeExtractor:
    """Extracts and judges knowledge from various sources"""
    
    def __init__(self):
        config = get_config()
        self.client = OpenAI(api_key=config.openai.api_key)
        self.model = config.openai.model
        self.language = config.agent.language
    
    def extract_from_text(self, text: str, source: str = "chat") -> List[ExtractedKnowledge]:
        """
        Extract knowledge entries from text
        
        Args:
            text: Input text
            source: Source of text (chat, tool, etc.)
            
        Returns:
            List of extracted knowledge entries
        """
        if not text or len(text.strip()) < 10:
            return []
        
        prompt = self._build_extraction_prompt(text, source)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            return self._parse_extraction_response(content, source)
        except Exception as e:
            print(f"Error extracting knowledge: {e}")
            return []
    
    def _build_extraction_prompt(self, text: str, source: str) -> str:
        """Build prompt for knowledge extraction"""
        language_instruction = {
            "vi": "Trả lời bằng tiếng Việt",
            "en": "Respond in English",
            "ja": "日本語で答えてください",
            "zh": "用中文回答"
        }.get(self.language, "Respond in English")
        
        return f"""Bạn là một chuyên gia trích xuất kiến thức. Phân tích văn bản sau và trích xuất các kiến thức quan trọng.

Văn bản:
{text}

Nguồn: {source}

{language_instruction}

Yêu cầu:
1. Trích xuất các kiến thức/sự kiện quan trọng (tối đa 3 cái)
2. Mỗi kiến thức phải:
   - Độc lập và hoàn chỉnh
   - Có giá trị lâu dài
   - Không là thông tin tầm thường
3. Đánh giá độ tin cậy (0-1)
4. Quyết định có nên lưu vào cơ sở dữ liệu không

Định dạng JSON:
{{
  "extractions": [
    {{
      "content": "Nội dung kiến thức",
      "confidence": 0.9,
      "should_store": true,
      "reason": "Lý do lưu/không lưu"
    }}
  ]
}}

Chỉ trả lại JSON, không có text khác."""
    
    def _parse_extraction_response(self, response: str, source: str) -> List[ExtractedKnowledge]:
        """Parse extraction response"""
        import json, re
        
        def _safe_json_loads(s: str):
            """Attempt to load JSON with fallbacks for common mistakes"""
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                # 1. Replace single quotes with double quotes (naïve but helps)
                s_fixed = re.sub(r"'([^']*)'", r'"\\1"', s)
                try:
                    return json.loads(s_fixed)
                except json.JSONDecodeError:
                    # 2. Add quotes around unquoted keys
                    s_fixed = re.sub(r"(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\\1":', s_fixed)
                    try:
                        return json.loads(s_fixed)
                    except json.JSONDecodeError:
                        # Fallback to json5 if available
                        try:
                            import json5
                            return json5.loads(s)
                        except Exception:
                            raise
        
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start == -1 or end == 0:
                return []
            
            json_str = response[start:end]
            data = _safe_json_loads(json_str)
            
            extractions = []
            for item in data.get("extractions", []):
                extraction = ExtractedKnowledge(
                    content=item.get("content", "").strip(),
                    confidence=float(item.get("confidence", 0.5)),
                    source=source,
                    metadata={"extraction_method": "llm"},
                    should_store=item.get("should_store", True),
                    reason=item.get("reason", "")
                )
                
                if extraction.content:
                    extractions.append(extraction)
            
            return extractions
        except Exception as e:
            print(f"Error parsing extraction response: {e}")
            return []
    
    def judge_knowledge(self, content: str) -> Tuple[bool, float, str]:
        """
        Judge if knowledge should be stored
        
        Args:
            content: Knowledge content
            
        Returns:
            (should_store, confidence, reason)
        """
        prompt = f"""Đánh giá xem kiến thức sau có nên được lưu vào cơ sở dữ liệu dài hạn không.

Kiến thức: {content}

Tiêu chí:
- Có giá trị lâu dài?
- Là thông tin chính xác?
- Không phải spam/rác?
- Không quá tầm thường?

Trả lời JSON:
{{
  "should_store": true/false,
  "confidence": 0.0-1.0,
  "reason": "Lý do"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200
            )
            
            import json
            content_text = response.choices[0].message.content
            start = content_text.find('{')
            end = content_text.rfind('}') + 1
            
            if start != -1 and end > start:
                data = json.loads(content_text[start:end])
                return (
                    data.get("should_store", False),
                    float(data.get("confidence", 0.5)),
                    data.get("reason", "")
                )
        except Exception as e:
            print(f"Error judging knowledge: {e}")
        
        return False, 0.5, "Error in judgment"
    
    def extract_from_tool_output(self, tool_name: str, output: str) -> List[ExtractedKnowledge]:
        """
        Extract knowledge from tool output
        
        Args:
            tool_name: Name of the tool
            output: Tool output
            
        Returns:
            List of extracted knowledge
        """
        return self.extract_from_text(output, source=f"tool:{tool_name}")

