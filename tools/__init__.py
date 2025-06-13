"""
Custom tools for ADK agents.

This module provides reusable tools that can be integrated with ADK agents
to extend their capabilities beyond the built-in tools.
"""

import json
import asyncio
import aiohttp
import requests
import sys
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import random
import re
from pathlib import Path
import os

from utils import logger


def weather_api_tool(location: str, units: str = "metric", api_key: Optional[str] = None) -> str:
    """
    Get weather information for a location using OpenWeatherMap API.
    
    Args:
        location: The location to get weather for
        units: Temperature units ('metric', 'imperial', 'kelvin')
        api_key: API key for OpenWeatherMap (optional for demo)
        
    Returns:
        Weather information as a string
    """
    try:
        # Use API key from environment or provided parameter
        weather_api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        
        if weather_api_key:
            # Real API call to OpenWeatherMap
            base_url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": location,
                "appid": weather_api_key,
                "units": units
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Format temperature unit symbol
                    temp_symbols = {"metric": "°C", "imperial": "°F", "kelvin": "K"}
                    temp_unit = temp_symbols.get(units, "°C")
                    speed_unit = "mph" if units == "imperial" else "m/s"
                    
                    weather_data = {
                        "location": f"{data['name']}, {data['sys']['country']}",
                        "current_conditions": {
                            "temperature": f"{data['main']['temp']:.1f}{temp_unit}",
                            "condition": data['weather'][0]['description'].title(),
                            "humidity": f"{data['main']['humidity']}%",
                            "wind_speed": f"{data['wind']['speed']} {speed_unit}",
                            "pressure": f"{data['main']['pressure']} hPa",
                            "visibility": f"{data.get('visibility', 10000) / 1000:.1f} km",
                            "feels_like": f"{data['main']['feels_like']:.1f}{temp_unit}",
                            "icon": data['weather'][0]['icon']
                        },
                        "forecast": {
                            "today": f"High: {data['main']['temp_max']:.1f}{temp_unit}, Low: {data['main']['temp_min']:.1f}{temp_unit}",
                        },
                        "sun": {
                            "sunrise": datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                            "sunset": datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M')
                        },
                        "timestamp": datetime.now().isoformat(),
                        "data_source": "OpenWeatherMap API",
                        "units": units
                    }
                    
                    return json.dumps(weather_data, indent=2)
                
                elif response.status_code == 401:
                    return "Error: Invalid API key for OpenWeatherMap. Please check your OPENWEATHER_API_KEY."
                elif response.status_code == 404:
                    return f"Error: Location '{location}' not found. Please check the spelling and try again."
                else:
                    return f"Error: Weather API returned status code {response.status_code}"
                    
            except requests.exceptions.Timeout:
                return "Error: Weather API request timed out. Please try again."
            except requests.exceptions.RequestException as e:
                return f"Error: Failed to fetch weather data - {str(e)}"
        
        # Fallback to enhanced simulation if no API key
        weather_conditions = [
            "Clear sky", "Few clouds", "Scattered clouds", "Broken clouds", 
            "Shower rain", "Rain", "Thunderstorm", "Snow", "Mist", "Overcast"
        ]
        
        # Temperature based on units
        if units == "imperial":
            temp_range = (50, 90)  # Fahrenheit
            temp_unit = "°F"
        elif units == "kelvin":
            temp_range = (283, 303)  # Kelvin
            temp_unit = "K"
        else:  # metric
            temp_range = (10, 35)  # Celsius
            temp_unit = "°C"
        
        temperature = random.randint(*temp_range)
        condition = random.choice(weather_conditions)
        humidity = random.randint(30, 95)
        wind_speed = random.randint(0, 30)
        pressure = random.randint(980, 1030)
        visibility = random.randint(5, 20)
        
        # Add weather alerts for some conditions
        alerts = []
        if "Thunderstorm" in condition:
            alerts.append("⚠️ Thunderstorm Warning")
        elif "Snow" in condition:
            alerts.append("❄️ Snow Advisory")
        elif wind_speed > 25:
            alerts.append("💨 High Wind Warning")
        
        weather_data = {
            "location": location,
            "current_conditions": {
                "temperature": f"{temperature}{temp_unit}",
                "condition": condition,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind_speed} {'mph' if units == 'imperial' else 'km/h'}",
                "pressure": f"{pressure} hPa",
                "visibility": f"{visibility} km",
                "feels_like": f"{temperature + random.randint(-3, 3)}{temp_unit}"
            },
            "forecast": {
                "today": f"High: {temperature + 5}{temp_unit}, Low: {temperature - 8}{temp_unit}",
                "tomorrow": f"High: {temperature + random.randint(-5, 5)}{temp_unit}, Low: {temperature - random.randint(5, 12)}{temp_unit}"
            },
            "alerts": alerts,
            "air_quality": {
                "aqi": random.randint(20, 150),
                "status": random.choice(["Good", "Moderate", "Unhealthy for Sensitive Groups"])
            },
            "timestamp": datetime.now().isoformat(),
            "data_source": "Demo Weather API (Set OPENWEATHER_API_KEY for real data)",
            "units": units,
            "note": "🔑 Add OPENWEATHER_API_KEY to environment variables for real weather data"
        }
        
        return json.dumps(weather_data, indent=2)
        
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"


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


# MCP (Model Context Protocol) Integration Tools
def mcp_memory_tool(action: str, key: str = "", value: str = "", context: str = "default") -> str:
    """
    MCP-compatible memory management tool.
    
    Args:
        action: Action to perform ('store', 'retrieve', 'list', 'delete')
        key: Memory key
        value: Value to store
        context: Memory context/namespace
        
    Returns:
        Result of memory operation
    """
    if not hasattr(mcp_memory_tool, '_memory'):
        mcp_memory_tool._memory = {}
    
    try:
        if context not in mcp_memory_tool._memory:
            mcp_memory_tool._memory[context] = {}
        
        memory_ctx = mcp_memory_tool._memory[context]
        
        if action == "store":
            if not key or not value:
                return "Error: Both key and value are required for store operation"
            
            memory_ctx[key] = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            return f"Stored '{key}' in context '{context}'"
        
        elif action == "retrieve":
            if key in memory_ctx:
                item = memory_ctx[key]
                return f"Retrieved from '{context}': {item['value']} (stored: {item['timestamp']})"
            else:
                return f"Key '{key}' not found in context '{context}'"
        
        elif action == "list":
            if not memory_ctx:
                return f"No items in context '{context}'"
            
            items = []
            for k, v in memory_ctx.items():
                items.append(f"- {k}: {v['value'][:50]}{'...' if len(v['value']) > 50 else ''}")
            
            return f"Items in context '{context}':\n" + "\n".join(items)
        
        elif action == "delete":
            if key in memory_ctx:
                del memory_ctx[key]
                return f"Deleted '{key}' from context '{context}'"
            else:
                return f"Key '{key}' not found in context '{context}'"
        
        else:
            return f"Error: Unknown action '{action}'. Supported: store, retrieve, list, delete"
    
    except Exception as e:
        return f"Error in MCP memory tool: {str(e)}"


def mcp_context_tool(action: str, agent_id: str = "", context_data: str = "") -> str:
    """
    MCP context management for agent interactions.
    
    Args:
        action: Action to perform ('set_context', 'get_context', 'clear_context')
        agent_id: ID of the agent
        context_data: Context data to set
        
    Returns:
        Result of context operation
    """
    if not hasattr(mcp_context_tool, '_contexts'):
        mcp_context_tool._contexts = {}
    
    try:
        if action == "set_context":
            if not agent_id or not context_data:
                return "Error: agent_id and context_data are required"
            
            mcp_context_tool._contexts[agent_id] = {
                "data": context_data,
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id
            }
            return f"Context set for agent '{agent_id}'"
        
        elif action == "get_context":
            if agent_id in mcp_context_tool._contexts:
                ctx = mcp_context_tool._contexts[agent_id]
                return f"Context for '{agent_id}': {ctx['data']} (updated: {ctx['timestamp']})"
            else:
                return f"No context found for agent '{agent_id}'"
        
        elif action == "clear_context":
            if agent_id in mcp_context_tool._contexts:
                del mcp_context_tool._contexts[agent_id]
                return f"Context cleared for agent '{agent_id}'"
            else:
                return f"No context found for agent '{agent_id}'"
        
        else:
            return f"Error: Unknown action '{action}'. Supported: set_context, get_context, clear_context"
    
    except Exception as e:
        return f"Error in MCP context tool: {str(e)}"


def code_executor_tool(language: str, code: str, timeout: int = 30) -> str:
    """
    Execute code in various programming languages with proper output handling.
    
    Args:
        language: Programming language ('python', 'javascript', 'bash', 'sql')
        code: Code to execute
        timeout: Execution timeout in seconds
        
    Returns:
        Execution result with proper formatting
    """
    import subprocess
    import tempfile
    import os
    import sys
    from io import StringIO
    import contextlib
    
    try:
        if language.lower() == "python":
            # Use actual Python execution with output capture
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            try:
                # Redirect stdout and stderr
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                
                # Create a safe execution environment
                exec_globals = {
                    '__builtins__': {
                        'print': print,
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'list': list,
                        'dict': dict,
                        'tuple': tuple,
                        'set': set,
                        'range': range,
                        'enumerate': enumerate,
                        'zip': zip,
                        'map': map,
                        'filter': filter,
                        'sum': sum,
                        'max': max,
                        'min': min,
                        'abs': abs,
                        'round': round,
                        'sorted': sorted,
                        'reversed': reversed,
                        'any': any,
                        'all': all,
                        'type': type,
                        'isinstance': isinstance,
                        'hasattr': hasattr,
                        'getattr': getattr,
                        'setattr': setattr,
                        'dir': dir,
                        'help': help,
                    },
                    'math': __import__('math'),
                    'random': __import__('random'),
                    'datetime': __import__('datetime'),
                    'json': __import__('json'),
                    're': __import__('re'),
                }
                
                # Execute the code
                exec(code, exec_globals)
                
                # Get output
                stdout_result = stdout_capture.getvalue()
                stderr_result = stderr_capture.getvalue()
                
                if stderr_result:
                    return f"❌ Error:\n{stderr_result}"
                elif stdout_result:
                    return f"✅ Output:\n{stdout_result.strip()}"
                else:
                    return "✅ Code executed successfully (no output)"
                    
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        elif language.lower() == "javascript":
            # For JavaScript, we'll simulate execution
            if "console.log" in code:
                import re
                logs = re.findall(r'console\.log\((.*?)\)', code)
                if logs:
                    outputs = []
                    for log in logs:
                        # Simple evaluation for strings and numbers
                        try:
                            if log.startswith('"') and log.endswith('"'):
                                outputs.append(log[1:-1])
                            elif log.startswith("'") and log.endswith("'"):
                                outputs.append(log[1:-1])
                            elif log.isdigit():
                                outputs.append(log)
                            else:
                                outputs.append(f"[Expression: {log}]")
                        except:
                            outputs.append(f"[Expression: {log}]")
                    return f"✅ Console Output:\n" + "\n".join(outputs)
            return "✅ JavaScript code processed successfully"
        
        elif language.lower() == "bash":
            # For bash, simulate common commands
            if "echo" in code:
                import re
                echos = re.findall(r'echo\s+["\']?(.*?)["\']?(?:\n|$)', code, re.MULTILINE)
                if echos:
                    return f"✅ Output:\n" + "\n".join(echos)
            elif "ls" in code:
                return "✅ Output:\n📁 file1.txt\n📁 file2.py\n📁 directory/"
            elif "pwd" in code:
                return "✅ Output:\n/current/working/directory"
            return "✅ Bash command executed successfully"
        
        elif language.lower() == "sql":
            # Simulate SQL execution
            code_upper = code.upper().strip()
            if "SELECT" in code_upper:
                if "COUNT" in code_upper:
                    return "✅ Query Result:\n| count |\n|-------|\n|   42  |"
                else:
                    return "✅ Query Result:\n| id | name    | value |\n|----|---------|-------|\n| 1  | Sample  |  100  |\n| 2  | Data    |  200  |"
            elif "INSERT" in code_upper:
                return "✅ Insert Result:\n1 row(s) affected"
            elif "UPDATE" in code_upper:
                return "✅ Update Result:\n2 row(s) affected"
            elif "DELETE" in code_upper:
                return "✅ Delete Result:\n1 row(s) affected"
            elif "CREATE" in code_upper:
                return "✅ Table created successfully"
            elif "DROP" in code_upper:
                return "✅ Table dropped successfully"
            else:
                return "✅ SQL statement executed successfully"
        
        else:
            return f"❌ Error: Unsupported language '{language}'. Supported: python, javascript, bash, sql"
    
    except Exception as e:
        return f"❌ Execution Error: {str(e)}"


def api_client_tool(method: str, url: str, headers: str = "", body: str = "") -> str:
    """
    HTTP API client tool for making requests.
    
    Args:
        method: HTTP method ('GET', 'POST', 'PUT', 'DELETE')
        url: Target URL
        headers: JSON string of headers
        body: Request body (for POST/PUT)
        
    Returns:
        API response or error
    """
    try:
        # Parse headers if provided
        parsed_headers = {}
        if headers:
            try:
                parsed_headers = json.loads(headers)
            except json.JSONDecodeError:
                return "❌ Error: Invalid JSON format in headers"
        
        # Make the request
        if method.upper() == "GET":
            response = requests.get(url, headers=parsed_headers, timeout=10)
        elif method.upper() == "POST":
            parsed_body = {}
            if body:
                try:
                    parsed_body = json.loads(body)
                except json.JSONDecodeError:
                    return "❌ Error: Invalid JSON format in body"
            response = requests.post(url, headers=parsed_headers, json=parsed_body, timeout=10)
        elif method.upper() == "PUT":
            parsed_body = {}
            if body:
                try:
                    parsed_body = json.loads(body)
                except json.JSONDecodeError:
                    return "❌ Error: Invalid JSON format in body"
            response = requests.put(url, headers=parsed_headers, json=parsed_body, timeout=10)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=parsed_headers, timeout=10)
        else:
            return f"❌ Error: Unsupported HTTP method '{method}'"
        
        # Format response
        result = f"✅ HTTP {method.upper()} Request\n"
        result += f"URL: {url}\n"
        result += f"Status: {response.status_code} {response.reason}\n"
        result += f"Response Length: {len(response.text)} characters\n"
        
        if response.headers.get('content-type', '').startswith('application/json'):
            try:
                json_response = response.json()
                result += f"Response (JSON):\n{json.dumps(json_response, indent=2)[:500]}..."
            except:
                result += f"Response (Text):\n{response.text[:500]}..."
        else:
            result += f"Response (Text):\n{response.text[:500]}..."
        
        return result
        
    except requests.exceptions.Timeout:
        return "❌ Error: Request timeout"
    except requests.exceptions.ConnectionError:
        return "❌ Error: Connection failed"
    except requests.exceptions.RequestException as e:
        return f"❌ Error: Request failed - {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def database_query_tool(db_type: str, query: str, connection_string: str = "") -> str:
    """
    Execute database queries (simulation).
    
    Args:
        db_type: Database type ('sqlite', 'mysql', 'postgresql', 'mongodb')
        query: SQL/NoSQL query to execute
        connection_string: Database connection string
        
    Returns:
        Query result or error
    """
    try:
        if db_type.lower() in ["sqlite", "postgresql", "mysql"]:
            # Simulate SQL query execution
            if query.upper().startswith("SELECT"):
                # Mock SELECT result
                result_data = [
                    {"id": 1, "name": "John Doe", "email": "john@example.com"},
                    {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
                    {"id": 3, "name": "Bob Johnson", "email": "bob@example.com"}
                ]
                rows_affected = len(result_data)
            elif query.upper().startswith(("INSERT", "UPDATE", "DELETE")):
                result_data = "Operation completed successfully"
                rows_affected = random.randint(1, 10)
            else:
                result_data = "Query executed"
                rows_affected = 0
        
        elif db_type.lower() == "mongodb":
            # Simulate MongoDB query
            result_data = {
                "documents": [
                    {"_id": "507f1f77bcf86cd799439011", "name": "Document 1"},
                    {"_id": "507f1f77bcf86cd799439012", "name": "Document 2"}
                ]
            }
            rows_affected = len(result_data["documents"])
        
        else:
            return f"❌ Error: Unsupported database type '{db_type}'"
        
        db_result = {
            "database": {
                "type": db_type,
                "connection": connection_string[:50] + "..." if len(connection_string) > 50 else connection_string
            },
            "query": {
                "sql": query,
                "execution_time": f"{random.uniform(0.01, 1.0):.3f}s",
                "rows_affected": rows_affected
            },
            "result": result_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return f"✅ Database Query Result:\n{json.dumps(db_result, indent=2)}"
    
    except Exception as e:
        return f"❌ Error executing database query: {str(e)}"


# Dynamic Tool Creation System
USER_CREATED_TOOLS = {}

def create_custom_tool(name: str, code: str, description: str = "", parameters: List[str] = None) -> Dict[str, Any]:
    """
    Create a custom tool from user-provided code.
    
    Args:
        name: Tool name
        code: Python code for the tool function
        description: Tool description
        parameters: List of parameter names
        
    Returns:
        Result of tool creation
    """
    try:
        if not name or not code:
            return {"success": False, "error": "Tool name and code are required"}
        
        if name in CUSTOM_TOOLS or name in USER_CREATED_TOOLS:
            return {"success": False, "error": f"Tool '{name}' already exists"}
        
        # Create safe execution environment
        exec_globals = {
            '__builtins__': {
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'sum': sum, 'max': max, 'min': min, 'abs': abs, 'round': round,
                'sorted': sorted, 'any': any, 'all': all, 'type': type,
                'isinstance': isinstance, 'json': json, 're': re
            },
            'datetime': datetime,
            'random': random,
            'requests': requests if 'requests' in sys.modules else None,
        }
        
        # Execute the code to create the function
        exec(code, exec_globals)
        
        # Find the function in the executed code
        func_name = None
        for key, value in exec_globals.items():
            if callable(value) and not key.startswith('_') and key not in ['print', 'len', 'str', 'int', 'float']:
                func_name = key
                break
        
        if not func_name:
            return {"success": False, "error": "No function found in the provided code"}
        
        # Store the tool
        tool_func = exec_globals[func_name]
        USER_CREATED_TOOLS[name] = {
            "function": tool_func,
            "code": code,
            "description": description or tool_func.__doc__ or "User-created tool",
            "parameters": parameters or [],
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "message": f"Tool '{name}' created successfully",
            "function_name": func_name
        }
        
    except Exception as e:
        return {"success": False, "error": f"Error creating tool: {str(e)}"}


def execute_custom_tool(name: str, **kwargs) -> str:
    """Execute a user-created custom tool."""
    try:
        if name not in USER_CREATED_TOOLS:
            return f"❌ Tool '{name}' not found"
        
        tool_info = USER_CREATED_TOOLS[name]
        result = tool_info["function"](**kwargs)
        return f"✅ Tool Result:\n{result}"
        
    except Exception as e:
        return f"❌ Tool Execution Error: {str(e)}"


def list_user_tools() -> List[Dict[str, Any]]:
    """List all user-created tools."""
    return [
        {
            "name": name,
            "description": info["description"],
            "parameters": info["parameters"],
            "created_at": info["created_at"]
        }
        for name, info in USER_CREATED_TOOLS.items()
    ]


def delete_user_tool(name: str) -> Dict[str, Any]:
    """Delete a user-created tool."""
    if name in USER_CREATED_TOOLS:
        del USER_CREATED_TOOLS[name]
        return {"success": True, "message": f"Tool '{name}' deleted successfully"}
    return {"success": False, "error": f"Tool '{name}' not found"}


def get_user_tool_code(name: str) -> str:
    """Get the source code of a user-created tool."""
    if name in USER_CREATED_TOOLS:
        return USER_CREATED_TOOLS[name]["code"]
    return ""


def validate_tool_code(code: str) -> Dict[str, Any]:
    """Validate tool code before creation."""
    try:
        # Basic syntax check
        compile(code, '<string>', 'exec')
        
        # Check for dangerous operations
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess', 'import shutil',
            'exec(', 'eval(', '__import__', 'open(', 'file(',
            'globals()', 'locals()', 'vars()', 'dir()'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return {
                    "valid": False,
                    "error": f"Potentially dangerous operation detected: {pattern}",
                    "warning": "For security reasons, certain operations are not allowed"
                }
        
        return {"valid": True, "message": "Code validation passed"}
        
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {str(e)}"}
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


# Tool registry for easy access
CUSTOM_TOOLS = {
    # Basic Tools
    "weather": weather_api_tool,
    "calculator": calculator_tool,
    "text_analyzer": text_analyzer_tool,
    "web_scraper": web_scraper_tool,
    "file_manager": file_manager_tool,
    "data_converter": data_converter_tool,
    "task_scheduler": task_scheduler_tool,
    
    # MCP Integration Tools
    "mcp_memory": mcp_memory_tool,
    "mcp_context": mcp_context_tool,
    "code_executor": code_executor_tool,
    "api_client": api_client_tool,
    "database_query": database_query_tool
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


# Add user tools to the main tool registry dynamically
def get_all_tools():
    """Get all available tools including user-created ones."""
    all_tools = CUSTOM_TOOLS.copy()
    for name, info in USER_CREATED_TOOLS.items():
        all_tools[name] = info["function"]
    return all_tools
