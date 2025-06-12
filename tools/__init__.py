"""
Custom tools for ADK agents.

This module provides reusable tools that can be integrated with ADK agents
to extend their capabilities beyond the built-in tools.
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import random
import re
from pathlib import Path

from utils import logger


def weather_api_tool(location: str, api_key: Optional[str] = None) -> str:
    """
    Get weather information for a location using a weather API.
    
    Args:
        location: The location to get weather for
        api_key: API key for weather service (optional for demo)
        
    Returns:
        Weather information as a string
    """
    # Mock weather data for demonstration
    # In production, integrate with actual weather API like OpenWeatherMap
    
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "overcast"]
    temperature = random.randint(15, 30)
    condition = random.choice(weather_conditions)
    humidity = random.randint(40, 80)
    wind_speed = random.randint(5, 25)
    
    weather_data = {
        "location": location,
        "temperature": f"{temperature}°C",
        "condition": condition,
        "humidity": f"{humidity}%",
        "wind_speed": f"{wind_speed} km/h",
        "timestamp": datetime.now().isoformat()
    }
    
    return json.dumps(weather_data, indent=2)


def calculator_tool(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation or error message
    """
    try:
        # Simple validation for security
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression: {expression}"
        
        # Evaluate safely (in production, use a proper math parser)
        result = eval(expression)
        return f"Result: {expression} = {result}"
    
    except ZeroDivisionError:
        return f"Error: Division by zero in expression: {expression}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def text_analyzer_tool(text: str, analysis_type: str = "sentiment") -> str:
    """
    Analyze text for various properties.
    
    Args:
        text: Text to analyze
        analysis_type: Type of analysis ('sentiment', 'keywords', 'readability')
        
    Returns:
        Analysis results as JSON string
    """
    analysis_result = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(re.split(r'[.!?]+', text)),
        "analysis_type": analysis_type,
        "timestamp": datetime.now().isoformat()
    }
    
    if analysis_type == "sentiment":
        # Mock sentiment analysis
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        analysis_result.update({
            "sentiment": sentiment,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        })
    
    elif analysis_type == "keywords":
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        analysis_result.update({
            "top_keywords": [{"word": word, "frequency": freq} for word, freq in keywords],
            "unique_words": len(set(words))
        })
    
    elif analysis_type == "readability":
        # Simple readability metrics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        avg_chars_per_word = sum(len(word) for word in words) / max(len(words), 1)
        
        analysis_result.update({
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "avg_chars_per_word": round(avg_chars_per_word, 2),
            "readability_score": "intermediate"  # Simplified
        })
    
    return json.dumps(analysis_result, indent=2)


async def web_scraper_tool(url: str, selector: Optional[str] = None) -> str:
    """
    Scrape content from a web page.
    
    Args:
        url: URL to scrape
        selector: CSS selector for specific content (optional)
        
    Returns:
        Scraped content or error message
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Simple text extraction (in production, use BeautifulSoup)
                    # Remove HTML tags for basic text extraction
                    import re
                    text_content = re.sub(r'<[^>]+>', ' ', content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
                    # Limit content length
                    if len(text_content) > 1000:
                        text_content = text_content[:1000] + "..."
                    
                    result = {
                        "url": url,
                        "status": "success",
                        "content_length": len(text_content),
                        "content": text_content,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return json.dumps(result, indent=2)
                else:
                    return f"Error: Failed to fetch {url}, status code: {response.status}"
    
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


def file_manager_tool(action: str, file_path: str, content: Optional[str] = None) -> str:
    """
    Manage files (read, write, list).
    
    Args:
        action: Action to perform ('read', 'write', 'list', 'exists')
        file_path: Path to the file or directory
        content: Content to write (for write action)
        
    Returns:
        Result of the file operation
    """
    try:
        path = Path(file_path)
        
        if action == "read":
            if path.exists() and path.is_file():
                content = path.read_text(encoding='utf-8')
                return f"File content from {file_path}:\n{content}"
            else:
                return f"Error: File {file_path} does not exist"
        
        elif action == "write":
            if content is None:
                return "Error: No content provided for write operation"
            
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return f"Successfully wrote content to {file_path}"
        
        elif action == "list":
            if path.exists() and path.is_dir():
                files = [str(p.name) for p in path.iterdir()]
                return f"Contents of {file_path}:\n" + "\n".join(files)
            else:
                return f"Error: Directory {file_path} does not exist"
        
        elif action == "exists":
            exists = path.exists()
            return f"File/directory {file_path} {'exists' if exists else 'does not exist'}"
        
        else:
            return f"Error: Unknown action '{action}'. Supported: read, write, list, exists"
    
    except Exception as e:
        return f"Error performing {action} on {file_path}: {str(e)}"


def data_converter_tool(data: str, from_format: str, to_format: str) -> str:
    """
    Convert data between different formats.
    
    Args:
        data: Data to convert
        from_format: Source format ('json', 'csv', 'xml', 'yaml')
        to_format: Target format ('json', 'csv', 'xml', 'yaml')
        
    Returns:
        Converted data or error message
    """
    try:
        # Parse input data
        if from_format.lower() == "json":
            parsed_data = json.loads(data)
        elif from_format.lower() == "csv":
            # Simple CSV parsing (in production, use pandas)
            lines = data.strip().split('\n')
            headers = lines[0].split(',')
            rows = [line.split(',') for line in lines[1:]]
            parsed_data = [dict(zip(headers, row)) for row in rows]
        else:
            return f"Error: Unsupported input format '{from_format}'"
        
        # Convert to output format
        if to_format.lower() == "json":
            return json.dumps(parsed_data, indent=2)
        elif to_format.lower() == "csv":
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                headers = list(parsed_data[0].keys())
                csv_lines = [','.join(headers)]
                for item in parsed_data:
                    csv_lines.append(','.join(str(item.get(h, '')) for h in headers))
                return '\n'.join(csv_lines)
            else:
                return "Error: Cannot convert non-list data to CSV"
        else:
            return f"Error: Unsupported output format '{to_format}'"
    
    except Exception as e:
        return f"Error converting from {from_format} to {to_format}: {str(e)}"


def task_scheduler_tool(action: str, task_name: str = "", schedule_time: str = "", 
                       task_data: str = "") -> str:
    """
    Simple task scheduling tool.
    
    Args:
        action: Action to perform ('schedule', 'list', 'cancel')
        task_name: Name of the task
        schedule_time: When to schedule the task (ISO format)
        task_data: Data associated with the task
        
    Returns:
        Result of the scheduling operation
    """
    # Simple in-memory task storage (in production, use a proper scheduler)
    if not hasattr(task_scheduler_tool, '_tasks'):
        task_scheduler_tool._tasks = {}
    
    try:
        if action == "schedule":
            if not task_name or not schedule_time:
                return "Error: task_name and schedule_time are required for scheduling"
            
            # Parse schedule time
            scheduled_dt = datetime.fromisoformat(schedule_time.replace('Z', '+00:00'))
            
            task_scheduler_tool._tasks[task_name] = {
                "name": task_name,
                "scheduled_time": schedule_time,
                "data": task_data,
                "status": "scheduled",
                "created_at": datetime.now().isoformat()
            }
            
            return f"Task '{task_name}' scheduled for {schedule_time}"
        
        elif action == "list":
            if not task_scheduler_tool._tasks:
                return "No scheduled tasks"
            
            task_list = []
            for task_name, task_info in task_scheduler_tool._tasks.items():
                task_list.append(f"- {task_name}: {task_info['scheduled_time']} ({task_info['status']})")
            
            return "Scheduled tasks:\n" + "\n".join(task_list)
        
        elif action == "cancel":
            if task_name in task_scheduler_tool._tasks:
                del task_scheduler_tool._tasks[task_name]
                return f"Task '{task_name}' cancelled"
            else:
                return f"Task '{task_name}' not found"
        
        else:
            return f"Error: Unknown action '{action}'. Supported: schedule, list, cancel"
    
    except Exception as e:
        return f"Error in task scheduler: {str(e)}"


# Tool registry for easy access
CUSTOM_TOOLS = {
    "weather": weather_api_tool,
    "calculator": calculator_tool,
    "text_analyzer": text_analyzer_tool,
    "web_scraper": web_scraper_tool,
    "file_manager": file_manager_tool,
    "data_converter": data_converter_tool,
    "task_scheduler": task_scheduler_tool
}


def get_tool(tool_name: str):
    """Get a tool by name from the registry."""
    return CUSTOM_TOOLS.get(tool_name)


def list_available_tools() -> List[str]:
    """List all available custom tools."""
    return list(CUSTOM_TOOLS.keys())


def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific tool."""
    tool = get_tool(tool_name)
    if tool:
        return {
            "name": tool_name,
            "function": tool.__name__,
            "description": tool.__doc__.strip() if tool.__doc__ else "No description",
            "module": tool.__module__
        }
    return None
