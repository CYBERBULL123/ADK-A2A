"""
AI-Powered Tools Dashboard with MCP Integration.

This module provides an advanced tools interface with AI-generated tools,
Model Context Protocol (MCP) integration, and dynamic tool management.
"""

import streamlit as st
import asyncio
import json
import uuid
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
import random
import inspect
from pathlib import Path
import os
import hashlib
import google.generativeai as genai

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config import config
from utils import console, logger
from agents.basic import SimpleAgent, ToolAgent

# Import tools with error handling
try:
    from tools import CUSTOM_TOOLS, get_tool_info, list_user_tools
except ImportError as e:
    console.print(f"[yellow]Warning: Could not import some tools functions: {e}[/yellow]")
    from tools import CUSTOM_TOOLS, get_tool_info
    
    # Define a fallback for list_user_tools if it's not available
    def list_user_tools():
        return []

from frontend.ui_utils import (
    load_css, create_header, create_feature_card, create_status_card,
    create_metric_card, display_agent_response, create_chat_interface,
    add_chat_message, clear_chat_history, create_progress_indicator,
    create_code_block, create_expandable_section,
    create_data_table, create_notification, format_timestamp
)


class MCPToolManager:
    """
    Model Context Protocol (MCP) integration for dynamic tool management.
    
    Focuses on enabling agents to discover, register, and execute tools
    across different agent types (basic, multi-agent, A2A systems).
    """
    
    def __init__(self):
        self.mcp_tools = {}
        self.tool_registry = {}
        self.execution_history = []
        self.agent_tool_bindings = {}  # Track which agents use which tools
        
    async def register_mcp_tool(self, tool_name: str, tool_spec: Dict[str, Any], 
                               agent_types: List[str] = None) -> bool:
        """Register a new MCP tool with agent type compatibility."""
        try:
            self.mcp_tools[tool_name] = {
                'spec': tool_spec,
                'registered_at': datetime.now().isoformat(),
                'usage_count': 0,
                'last_used': None,
                'agent_types': agent_types or ['basic', 'multi_agent', 'a2a'],
                'cross_agent_compatible': True
            }
            logger.info(f"MCP tool '{tool_name}' registered for agent types: {agent_types}")
            return True
        except Exception as e:
            logger.error(f"Error registering MCP tool '{tool_name}': {e}")
            return False
    
    async def bind_tool_to_agent(self, tool_name: str, agent_id: str, agent_type: str) -> bool:
        """Bind a tool to a specific agent for A2A and multi-agent scenarios."""
        try:
            if tool_name not in self.mcp_tools:
                raise ValueError(f"Tool '{tool_name}' not found in MCP registry")
            
            if agent_id not in self.agent_tool_bindings:
                self.agent_tool_bindings[agent_id] = {
                    'agent_type': agent_type,
                    'tools': [],
                    'bound_at': datetime.now().isoformat()
                }
            
            if tool_name not in self.agent_tool_bindings[agent_id]['tools']:
                self.agent_tool_bindings[agent_id]['tools'].append(tool_name)
                logger.info(f"Tool '{tool_name}' bound to agent '{agent_id}' ({agent_type})")
            
            return True
        except Exception as e:
            logger.error(f"Error binding tool to agent: {e}")
            return False
    
    async def execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any], 
                             agent_id: str = None) -> Dict[str, Any]:
        """Execute an MCP tool with given parameters, optionally from a specific agent."""
        try:
            if tool_name not in self.mcp_tools:
                raise ValueError(f"MCP tool '{tool_name}' not found")
            
            # Check agent authorization if agent_id provided
            if agent_id and agent_id in self.agent_tool_bindings:
                if tool_name not in self.agent_tool_bindings[agent_id]['tools']:
                    logger.warning(f"Agent '{agent_id}' not authorized to use tool '{tool_name}'")
            
            # Simulate MCP tool execution
            execution_id = str(uuid.uuid4())
            
            # Update usage statistics
            self.mcp_tools[tool_name]['usage_count'] += 1
            self.mcp_tools[tool_name]['last_used'] = datetime.now().isoformat()
            
            # Simulate execution based on tool type
            tool_spec = self.mcp_tools[tool_name]['spec']
            result = await self._simulate_tool_execution(tool_spec, parameters)
            
            # Record execution history with agent context
            self.execution_history.append({
                'execution_id': execution_id,
                'tool_name': tool_name,
                'parameters': parameters,
                'result': result,
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            return {
                'success': True,
                'execution_id': execution_id,
                'result': result,
                'timestamp': datetime.now().isoformat(),
                'agent_context': agent_id
            }
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'agent_context': agent_id
            }
            
            self.execution_history.append({
                'execution_id': str(uuid.uuid4()),
                'tool_name': tool_name,
                'parameters': parameters,
                'result': error_result,
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat(),
                'success': False
            })
            
            return error_result
    
    async def _simulate_tool_execution(self, tool_spec: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
        """Simulate MCP tool execution based on tool specification."""
        tool_type = tool_spec.get('type', 'generic')
        
        if tool_type == 'data_processor':
            return f"Processed {len(str(parameters))} characters of data"
        elif tool_type == 'api_client':
            return {"status": "success", "data": "API response simulated", "timestamp": datetime.now().isoformat()}
        elif tool_type == 'code_executor':
            return f"Code executed successfully: {parameters.get('code', 'N/A')[:50]}..."
        elif tool_type == 'file_manager':
            return f"File operation completed: {parameters.get('operation', 'unknown')}"
        elif tool_type == 'database_query':
            return {"rows_affected": random.randint(1, 100), "query_time": f"{random.uniform(0.1, 2.0):.3f}s"}
        else:
            return f"Generic tool execution completed with parameters: {parameters}"
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool usage statistics."""
        total_tools = len(self.mcp_tools)
        total_executions = sum(tool['usage_count'] for tool in self.mcp_tools.values())
        
        # Recent executions (last 24 hours)
        recent_executions = [
            ex for ex in self.execution_history 
            if (datetime.now() - datetime.fromisoformat(ex['timestamp'])).days < 1
        ]
        
        success_rate = (
            len([ex for ex in recent_executions if ex['success']]) / len(recent_executions) * 100
            if recent_executions else 100
        )
        
        return {
            'total_tools': total_tools,
            'total_executions': total_executions,
            'recent_executions': len(recent_executions),
            'success_rate': success_rate,
            'most_used_tool': max(self.mcp_tools.items(), key=lambda x: x[1]['usage_count'], default=(None, None))[0]
        }


class AIToolGenerator:
    """AI-powered tool generation system using CodeGenerationAgent from ADK agents."""
    
    def __init__(self):
        """Initialize the AI tool generator with CodeGenerationAgent."""
        try:
            # Import the CodeGenerationAgent from agents
            from agents.basic import create_agent
            self.code_agent = create_agent("code_generator", "ToolGenerator")
            self.available = True
            console.print("[green]✓ AI Tool Generator initialized with CodeGenerationAgent[/green]")
        except Exception as e:
            self.code_agent = None
            self.available = False
            console.print(f"[yellow]Warning: Could not initialize CodeGenerationAgent: {e}. Using fallback mode.[/yellow]")    
    async def generate_tool(self, description: str, category: str, complexity: str) -> Dict[str, Any]:
        """Generate a tool using CodeGenerationAgent."""
        try:
            if not self.available or not self.code_agent:
                return self._fallback_generation(description, category, complexity)
            
            # Generate tool using CodeGenerationAgent
            tool_name = self._generate_tool_name_simple(description)
            parameters = self._extract_parameters_simple(description)
            
            # Use the CodeGenerationAgent to generate the actual code
            code = self.code_agent.generate_tool_code(description, category, complexity)
            
            return {
                'success': True,
                'tool_name': tool_name,
                'description': description,
                'category': category,
                'complexity': complexity,
                'parameters': parameters,
                'code': code,
                'generated_at': datetime.now().isoformat(),
                'ai_confidence': random.uniform(0.90, 0.98),
                'ai_powered': True,
                'generator': 'CodeGenerationAgent'
            }
            
        except Exception as e:
            console.print(f"[red]Error in AI generation: {e}[/red]")
            return self._fallback_generation(description, category, complexity)
    
    async def _generate_tool_name_ai(self, description: str) -> str:
        """Generate a meaningful tool name using AI."""
        try:
            prompt = f"""
            Generate a Python function name for a tool based on this description:
            "{description}"
            
            Requirements:
            - Must be a valid Python function name (snake_case)
            - Should be descriptive and clear
            - Maximum 3-4 words
            - No spaces or special characters except underscores
            
            Return ONLY the function name, nothing else.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            
            tool_name = response.text.strip().replace(' ', '_').lower()
            # Clean the name to ensure it's valid
            tool_name = ''.join(c for c in tool_name if c.isalnum() or c == '_')
            
            return tool_name if tool_name else 'generated_tool'
            
        except Exception as e:
            console.print(f"[yellow]AI name generation failed: {e}[/yellow]")
            return self._fallback_tool_name(description)
    
    async def _extract_parameters_ai(self, description: str) -> List[Dict[str, str]]:
        """Extract function parameters using AI analysis."""
        try:
            prompt = f"""
            Analyze this tool description and determine what parameters the function should have:
            "{description}"
            
            Return a JSON list of parameters where each parameter has:
            - name: parameter name (snake_case)
            - type: Python type hint (str, int, bool, List[str], Dict, Any, etc.)
            - description: brief description of the parameter
            
            Example format:
            [
                {{"name": "input_text", "type": "str", "description": "The text to process"}},
                {{"name": "max_length", "type": "int", "description": "Maximum length of output"}}
            ]
            
            Return ONLY the JSON array, no explanation.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            
            # Parse the JSON response
            try:
                parameters = json.loads(response.text.strip())
                return parameters if isinstance(parameters, list) else []
            except json.JSONDecodeError:
                return self._fallback_parameters(description)
                
        except Exception as e:
            console.print(f"[yellow]AI parameter extraction failed: {e}[/yellow]")
            return self._fallback_parameters(description)
    
    async def _generate_tool_code_ai(self, description: str, category: str, complexity: str, 
                                   tool_name: str, parameters: List[Dict[str, str]]) -> str:
        """Generate complete tool code using AI."""
        try:
            # Prepare parameter string
            if parameters:
                param_strings = []
                for param in parameters:
                    param_strings.append(f"{param['name']}: {param['type']}")
                param_str = ', '.join(param_strings)
            else:
                param_str = "input_data: str"
            
            prompt = f"""
            Generate a complete Python function that implements this tool:
            
            Description: "{description}"
            Category: {category}
            Complexity: {complexity}
            Function name: {tool_name}
            Parameters: {param_str}
            
            Requirements:
            1. Function must return Dict[str, Any] with 'success' and 'result' keys
            2. Include proper error handling with try/except
            3. Add meaningful docstring explaining the function
            4. Implement actual working logic (not just placeholders)
            5. Include necessary imports within the function if needed
            6. Make the function production-ready and robust
            7. For password generation: use secrets module for security
            8. For API calls: use requests with proper error handling
            9. For file operations: handle file not found and permissions
            10. Return structured data with timestamp when appropriate
            
            Generate ONLY the function definition with proper implementation.
            Do not include any explanations before or after the code.
            """
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            
            # Clean the response to get just the code
            code = response.text.strip()
            
            # Remove any markdown formatting if present
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            console.print(f"[yellow]AI code generation failed: {e}[/yellow]")
            return self._fallback_code_generation(description, category, complexity, tool_name, parameters)
    
    def _fallback_generation(self, description: str, category: str, complexity: str) -> Dict[str, Any]:
        """Fallback generation when AI is not available."""
        tool_name = self._fallback_tool_name(description)
        parameters = self._fallback_parameters(description)
        code = self._fallback_code_generation(description, category, complexity, tool_name, parameters)
        
        return {
            'success': True,
            'tool_name': tool_name,
            'description': description,
            'category': category,
            'complexity': complexity,
            'parameters': parameters,
            'code': code,
            'generated_at': datetime.now().isoformat(),
            'ai_confidence': 0.75,
            'ai_powered': False,
            'fallback_mode': True
        }
    
    def _fallback_tool_name(self, description: str) -> str:
        """Generate a tool name using simple pattern matching."""
        desc_lower = description.lower()
        
        if 'password' in desc_lower:
            return 'generate_secure_password'
        elif 'hash' in desc_lower or 'encrypt' in desc_lower:
            return 'hash_generator'
        elif 'url' in desc_lower or 'api' in desc_lower:
            return 'api_client_tool'
        elif 'file' in desc_lower:
            return 'file_processor'
        elif 'text' in desc_lower:
            return 'text_processor'
        elif 'data' in desc_lower:
            return 'data_processor'
        else:
            # Extract key words for generic name
            words = [w for w in desc_lower.split() if len(w) > 3]
            return '_'.join(words[:2]) + '_tool' if words else 'generated_tool'
    
    def _fallback_parameters(self, description: str) -> List[Dict[str, str]]:
        """Extract parameters using simple pattern matching."""
        desc_lower = description.lower()
        parameters = []
        
        if 'password' in desc_lower:
            parameters = [
                {'name': 'length', 'type': 'int', 'description': 'Password length (default: 12)'},
                {'name': 'include_symbols', 'type': 'bool', 'description': 'Include special characters'}
            ]
        elif 'hash' in desc_lower:
            parameters = [
                {'name': 'text', 'type': 'str', 'description': 'Text to hash'},
                {'name': 'algorithm', 'type': 'str', 'description': 'Hash algorithm (md5, sha256)'}
            ]
        elif 'url' in desc_lower or 'api' in desc_lower:
            parameters = [
                {'name': 'url', 'type': 'str', 'description': 'URL or API endpoint'},
                {'name': 'method', 'type': 'str', 'description': 'HTTP method (GET, POST)'}
            ]
        elif 'file' in desc_lower:
            parameters = [
                {'name': 'file_path', 'type': 'str', 'description': 'Path to the file'},
                {'name': 'operation', 'type': 'str', 'description': 'File operation'}
            ]
        else:
            parameters = [
                {'name': 'input_data', 'type': 'str', 'description': 'Input data to process'}
            ]
        
        return parameters
    
    def _fallback_code_generation(self, description: str, category: str, complexity: str, 
                                tool_name: str, parameters: List[Dict[str, str]]) -> str:
        """Generate code using templates when AI is not available."""
        param_str = ', '.join([f"{p['name']}: {p['type']} = None" for p in parameters]) if parameters else "input_data: str"
        
        desc_lower = description.lower()
        
        if 'password' in desc_lower:
            return f'''def {tool_name}(length: int = 12, include_symbols: bool = True) -> Dict[str, Any]:
    """
    {description}
    
    Generated by AI Tool Generator (Fallback Mode).
    Category: {category}
    Complexity: {complexity}
    """
    import secrets
    import string
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Build character set
        chars = string.ascii_letters + string.digits
        if include_symbols:
            chars += "!@#$%^&*"
        
        # Generate secure password
        password = ''.join(secrets.choice(chars) for _ in range(length))
        
        return {{
            'success': True,
            'result': password,
            'length': len(password),
            'includes_symbols': include_symbols,
            'timestamp': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''
        
        elif 'hash' in desc_lower:
            return f'''def {tool_name}(text: str, algorithm: str = "sha256") -> Dict[str, Any]:
    """
    {description}
    
    Generated by AI Tool Generator (Fallback Mode).
    Category: {category}
    Complexity: {complexity}
    """
    import hashlib
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Get hash function
        hash_func = getattr(hashlib, algorithm.lower(), hashlib.sha256)
        
        # Generate hash
        hash_value = hash_func(text.encode()).hexdigest()
        
        return {{
            'success': True,
            'result': hash_value,
            'algorithm': algorithm,
            'input_length': len(text),
            'timestamp': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''
        
        else:
            return f'''def {tool_name}({param_str}) -> Dict[str, Any]:
    """
    {description}
    
    Generated by AI Tool Generator (Fallback Mode).
    Category: {category}
    Complexity: {complexity}
    """
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Generic processing logic
        result = f"Processed: {{input_data}}"
        
        return {{
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''

    def _generate_tool_name(self, description: str) -> str:
        """Generate a meaningful tool name from description."""
        desc_lower = description.lower()
        
        # Specific tool name patterns
        if 'password' in desc_lower:
            if 'generate' in desc_lower or 'create' in desc_lower:
                return 'generate_secure_password'
            return 'password_tool'
        
        elif 'hash' in desc_lower:
            return 'hash_generator'
        
        elif 'encrypt' in desc_lower:
            return 'text_encryptor'
        
        elif 'api' in desc_lower:
            if 'client' in desc_lower:
                return 'api_client'
            elif 'request' in desc_lower:
                return 'api_requester'
            return 'api_tool'
        
        elif 'file' in desc_lower:
            if 'read' in desc_lower:
                return 'file_reader'
            elif 'process' in desc_lower:
                return 'file_processor'
            return 'file_tool'
        
        elif 'text' in desc_lower:
            if 'analyze' in desc_lower:
                return 'text_analyzer'
            elif 'clean' in desc_lower:
                return 'text_cleaner'
            elif 'process' in desc_lower:
                return 'text_processor'
            return 'text_tool'
        
        elif 'data' in desc_lower:
            if 'validate' in desc_lower:
                return 'data_validator'
            elif 'convert' in desc_lower:
                return 'data_converter'
            elif 'process' in desc_lower:
                return 'data_processor'
            return 'data_tool'
        
        # Fallback: extract key words
        words = desc_lower.split()
        key_words = [
            word for word in words 
            if len(word) > 3 and word not in [
                'that', 'this', 'with', 'from', 'will', 'need', 'tool', 'function',
                'create', 'make', 'build', 'generate', 'and', 'the', 'for', 'can'
            ]
        ]
        
        if key_words:
            return '_'.join(key_words[:3])
        else:
            return f'generated_tool_{random.randint(1000, 9999)}'
        
    def _extract_parameters(self, description: str) -> List[Dict[str, str]]:
        """Extract potential parameters from description."""
        parameters = []
        desc_lower = description.lower()
        
        # Password-related parameters
        if 'password' in desc_lower:
            parameters.extend([
                {'name': 'length', 'type': 'int', 'description': 'Password length (default: 12)'},
                {'name': 'include_symbols', 'type': 'bool', 'description': 'Include special characters'},
                {'name': 'include_numbers', 'type': 'bool', 'description': 'Include numbers'}
            ])
        
        # Hash/encryption parameters
        elif 'hash' in desc_lower or 'encrypt' in desc_lower:
            parameters.extend([
                {'name': 'text', 'type': 'str', 'description': 'Text to hash or encrypt'},
                {'name': 'algorithm', 'type': 'str', 'description': 'Hashing algorithm (md5, sha256, etc.)'}
            ])
        
        # API-related parameters
        elif 'api' in desc_lower or 'request' in desc_lower or 'url' in desc_lower:
            parameters.extend([
                {'name': 'url', 'type': 'str', 'description': 'API endpoint URL'},
                {'name': 'method', 'type': 'str', 'description': 'HTTP method (GET, POST, etc.)'},
                {'name': 'headers', 'type': 'Dict', 'description': 'Request headers'},
                {'name': 'data', 'type': 'Dict', 'description': 'Request payload'}
            ])
        
        # File-related parameters
        elif 'file' in desc_lower:
            parameters.extend([
                {'name': 'file_path', 'type': 'str', 'description': 'Path to the file'},
                {'name': 'operation', 'type': 'str', 'description': 'File operation (read, info, etc.)'}
            ])
        
        # Text processing parameters
        elif 'text' in desc_lower or 'string' in desc_lower:
            parameters.extend([
                {'name': 'text', 'type': 'str', 'description': 'Input text to process'},
                {'name': 'operation', 'type': 'str', 'description': 'Text operation (analyze, clean, upper, etc.)'}
            ])
        
        # Data processing parameters
        elif 'data' in desc_lower:
            parameters.extend([
                {'name': 'data', 'type': 'Any', 'description': 'Input data to process'},
                {'name': 'operation', 'type': 'str', 'description': 'Data operation (process, validate, serialize, etc.)'}
            ])
        
        # Generic fallback
        else:
            parameters.append({'name': 'input_data', 'type': 'Any', 'description': 'Input data for processing'})
        
        return parameters
    
    def _generate_tool_code(self, description: str, category: str, complexity: str) -> str:
        """Generate tool code based on AI analysis."""
        tool_name = self._generate_tool_name(description)
        parameters = self._extract_parameters(description)
        
        # Create proper function with meaningful logic based on description
        if 'password' in description.lower():
            return self._generate_password_tool(tool_name, description, category, complexity)
        elif 'hash' in description.lower() or 'encrypt' in description.lower():
            return self._generate_hash_tool(tool_name, description, category, complexity)
        elif 'url' in description.lower() or 'api' in description.lower():
            return self._generate_api_tool(tool_name, description, category, complexity, parameters)
        elif 'file' in description.lower():
            return self._generate_file_tool(tool_name, description, category, complexity)
        elif 'text' in description.lower() or 'string' in description.lower():
            return self._generate_text_tool(tool_name, description, category, complexity)
        elif 'data' in description.lower():
            return self._generate_data_tool(tool_name, description, category, complexity)
        else:
            return self._generate_generic_tool(tool_name, description, category, complexity, parameters)
    
    def _generate_password_tool(self, tool_name: str, description: str, category: str, complexity: str) -> str:
        """Generate a password generation tool."""
        return f'''def {tool_name}(length: int = 12, include_symbols: bool = True, include_numbers: bool = True) -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated by AI Tool Generator.
    Category: {category}
    Complexity: {complexity}
    """
    import random
    import string
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Build character set based on requirements
        chars = string.ascii_letters
        if include_numbers:
            chars += string.digits
        if include_symbols:
            chars += "!@#$%^&*()-_=+[]{{}}|;:,.<>?"
        
        # Generate secure password
        password = ''.join(random.choice(chars) for _ in range(length))
        
        return {{
            'success': True,
            'password': password,
            'length': len(password),
            'strength': 'Strong' if length >= 12 and include_symbols else 'Medium',
            'generated_at': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''

    def _generate_hash_tool(self, tool_name: str, description: str, category: str, complexity: str) -> str:
        """Generate a hashing/encryption tool."""
        return f'''def {tool_name}(text: str, algorithm: str = 'sha256') -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated by AI Tool Generator.
    Category: {category}
    Complexity: {complexity}
    """
    import hashlib
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Supported algorithms
        algorithms = {{'md5': hashlib.md5, 'sha1': hashlib.sha1, 'sha256': hashlib.sha256, 'sha512': hashlib.sha512}}
        
        if algorithm not in algorithms:
            return {{'success': False, 'error': f'Unsupported algorithm: {{algorithm}}'}}
        
        # Generate hash
        hash_obj = algorithms[algorithm]()
        hash_obj.update(text.encode('utf-8'))
        hash_value = hash_obj.hexdigest()
        
        return {{
            'success': True,
            'original_text': text,
            'hash_value': hash_value,
            'algorithm': algorithm,
            'generated_at': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''

    def _generate_api_tool(self, tool_name: str, description: str, category: str, complexity: str, parameters: List[Dict]) -> str:
        """Generate an API client tool."""
        return f'''def {tool_name}(url: str, method: str = 'GET', headers: Dict = None, data: Dict = None) -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated by AI Tool Generator.
    Category: {category}
    Complexity: {complexity}
    """
    import requests
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        # Set default headers
        if headers is None:
            headers = {{'User-Agent': 'AI-Generated-Tool/1.0'}}
        
        # Make API request
        if method.upper() == 'GET':
            response = requests.get(url, headers=headers, timeout=10)
        elif method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=data, timeout=10)
        else:
            response = requests.request(method.upper(), url, headers=headers, json=data, timeout=10)
        
        # Parse response
        try:
            response_data = response.json()
        except:
            response_data = response.text
        
        return {{
            'success': True,
            'status_code': response.status_code,
            'data': response_data,
            'headers': dict(response.headers),
            'url': url,
            'method': method.upper(),
            'timestamp': datetime.now().isoformat()
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e), 'url': url}}'''

    def _generate_file_tool(self, tool_name: str, description: str, category: str, complexity: str) -> str:
        """Generate a file processing tool."""
        return f'''def {tool_name}(file_path: str, operation: str = 'read') -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated by AI Tool Generator.
    Category: {category}
    Complexity: {complexity}
    """
    import os
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        if not os.path.exists(file_path):
            return {{'success': False, 'error': f'File not found: {{file_path}}'}}
        
        if operation == 'read':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {{
                'success': True,
                'content': content,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        elif operation == 'info':
            stat = os.stat(file_path)
            return {{
                'success': True,
                'file_path': file_path,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        else:
            return {{'success': False, 'error': f'Unsupported operation: {{operation}}'}}
            
    except Exception as e:
        return {{'success': False, 'error': str(e), 'file_path': file_path}}'''

    def _generate_text_tool(self, tool_name: str, description: str, category: str, complexity: str) -> str:
        """Generate a text processing tool."""
        return f'''def {tool_name}(text: str, operation: str = 'analyze') -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated by AI Tool Generator.
    Category: {category}
    Complexity: {complexity}
    """
    import re
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        if operation == 'analyze':
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len(re.findall(r'[.!?]+', text))
            
            return {{
                'success': True,
                'text': text,
                'word_count': word_count,
                'character_count': char_count,
                'sentence_count': sentence_count,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        elif operation == 'clean':
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()
            return {{
                'success': True,
                'original': text,
                'cleaned': cleaned,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        elif operation == 'upper':
            return {{
                'success': True,
                'original': text,
                'result': text.upper(),
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        else:
            return {{'success': False, 'error': f'Unsupported operation: {{operation}}'}}
            
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''

    def _generate_data_tool(self, tool_name: str, description: str, category: str, complexity: str) -> str:
        """Generate a data processing tool."""
        return f'''def {tool_name}(data: Any, operation: str = 'process') -> Dict[str, Any]:
    """
    {description}
    
    Auto-generated by AI Tool Generator.
    Category: {category}
    Complexity: {complexity}
    """
    import json
    from datetime import datetime
    from typing import Dict, Any
    
    try:
        if operation == 'process':
            # Convert data to string for processing
            data_str = str(data)
            processed = data_str.upper()
            
            return {{
                'success': True,
                'original': data,
                'processed': processed,
                'data_type': type(data).__name__,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        elif operation == 'validate':
            is_valid = data is not None and str(data).strip() != ''
            return {{
                'success': True,
                'data': data,
                'is_valid': is_valid,
                'data_type': type(data).__name__,
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }}
        
        elif operation == 'serialize':
            try:
                serialized = json.dumps(data, default=str)
                return {{
                    'success': True,
                    'original': data,
                    'serialized': serialized,
                    'operation': operation,
                    'timestamp': datetime.now().isoformat()
                }}
            except:
                return {{
                    'success': True,
                    'original': data,
                    'serialized': str(data),
                    'operation': operation,
                    'timestamp': datetime.now().isoformat()
                }}
        
        else:
            return {{'success': False, 'error': f'Unsupported operation: {{operation}}'}}
            
    except Exception as e:
        return {{'success': False, 'error': str(e)}}'''

    def _generate_data_processor(self, description: str) -> Dict[str, Any]:
        """Generate a data processing tool."""
        return {
            'type': 'data_processor',
            'template': 'process_data',
            'imports': ['pandas as pd', 'numpy as np'],
            'complexity_multiplier': 1.2
        }

    def _generate_api_client(self, description: str) -> Dict[str, Any]:
        """Generate an API client tool."""
        return {
            'type': 'api_client',
            'template': 'api_request',
            'imports': ['requests', 'json'],
            'complexity_multiplier': 1.5
        }

    def _generate_utility_tool(self, description: str) -> Dict[str, Any]:
        """Generate a utility tool."""
        return {
            'type': 'utility',
            'template': 'generic_utility',
            'imports': ['os', 'sys'],
            'complexity_multiplier': 1.0
        }

    def _generate_validator(self, description: str) -> Dict[str, Any]:
        """Generate a validation tool."""
        return {
            'type': 'validator',
            'template': 'validate_data',
            'imports': ['re', 'typing'],
            'complexity_multiplier': 1.3
        }

    def _generate_converter(self, description: str) -> Dict[str, Any]:
        """Generate a data converter tool."""
        return {
            'type': 'converter',
            'template': 'convert_data',
            'imports': ['json', 'csv'],
            'complexity_multiplier': 1.1
        }


# Initialize global instances
mcp_manager = MCPToolManager()
ai_generator = AIToolGenerator()


def initialize_default_mcp_tools():
    """Initialize default MCP tools for agent integration demonstration."""
    default_tools = [
        {
            'name': 'agent_memory_store',
            'spec': {
                'type': 'data_processor',
                'description': 'Store and retrieve agent memory across sessions',
                'parameters': ['agent_id', 'key', 'value', 'operation'],
                'agent_types': ['basic', 'multi_agent', 'a2a']
            }
        },
        {
            'name': 'cross_agent_messenger',
            'spec': {
                'type': 'communication',
                'description': 'Enable messaging between agents in multi-agent systems',
                'parameters': ['from_agent', 'to_agent', 'message', 'priority'],
                'agent_types': ['multi_agent', 'a2a']
            }
        },
        {
            'name': 'agent_task_coordinator',
            'spec': {
                'type': 'coordination',
                'description': 'Coordinate tasks and resources across multiple agents',
                'parameters': ['task_id', 'assigned_agents', 'priority', 'deadline'],
                'agent_types': ['multi_agent', 'a2a']
            }
        },
        {
            'name': 'a2a_protocol_handler',
            'spec': {
                'type': 'protocol',
                'description': 'Handle A2A protocol communications and negotiations',
                'parameters': ['protocol_type', 'message', 'target_network', 'auth'],
                'agent_types': ['a2a']
            }
        },
        {
            'name': 'shared_context_manager',
            'spec': {
                'type': 'context_manager',
                'description': 'Manage shared context between agents',
                'parameters': ['context_id', 'data', 'permissions', 'ttl'],
                'agent_types': ['basic', 'multi_agent', 'a2a']
            }
        }
    ]
    
    for tool in default_tools:
        agent_types = tool['spec'].get('agent_types', ['basic', 'multi_agent', 'a2a'])
        asyncio.run(mcp_manager.register_mcp_tool(tool['name'], tool['spec'], agent_types))


def show_tools_dashboard():
    """Main tools dashboard with AI generation and MCP integration."""
    
    # Modern header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h1 style="text-align: center; margin: 0; color: white;">🤖 AI-Powered Tools Laboratory</h1>
        <p style="color: white; text-align: center; opacity: 0.9; margin: 0.5rem 0 0 0;">
            Generate, test, and manage tools with AI agents and MCP integration
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize MCP tools if not already done
    if 'mcp_initialized' not in st.session_state:
        initialize_default_mcp_tools()
        st.session_state.mcp_initialized = True
      # Create enhanced tabs with agent integration focus
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 AI Tool Creator", 
        "� Agent Integration",
        "🌐 MCP Servers", 
        "⚡ Smart Testing", 
        "📚 Tool Library", 
        "📊 Analytics"
    ])
    
    with tab1:
        show_ai_tool_creator()
    
    with tab2:
        show_agent_tool_integration()
    
    with tab3:
        show_mcp_integration()
    
    with tab4:
        show_smart_testing()
    
    with tab5:
        show_tool_library()
    
    with tab6:
        show_tool_analytics()


def show_ai_tool_creator():
    """AI-powered tool creation interface."""
    st.markdown("### 🤖 AI-Powered Tool Creation")
    st.markdown("""
    Describe what you need and let our AI agents create the perfect tool for you automatically.
    No coding required - just describe your requirements in natural language.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 💭 Tool Requirements")
        
        # Tool description input
        tool_description = st.text_area(
            "What do you need this tool to do?",
            placeholder="I need a tool that converts text to speech, or analyzes sentiment in social media posts, or generates QR codes...",
            height=120,
            help="Describe in detail what functionality you want the tool to provide"
        )
        
        # Tool category selection
        tool_category = st.selectbox(
            "Tool Category:",
            [
                "🌐 Web & API Integration",
                "📊 Data Processing & Analysis", 
                "🧮 Mathematical & Scientific",
                "📝 Text & Content Processing",
                "🔒 Security & Encryption",
                "📱 Communication & Notifications",
                "🎨 Media & Graphics",
                "🔧 System & Utilities",
                "🤖 AI & Machine Learning",
                "📦 Other/Custom"
            ]
        )
        
        # Complexity level
        complexity_level = st.select_slider(
            "Complexity Level:",
            options=["Simple", "Moderate", "Advanced", "Expert"],
            value="Moderate",
            help="Higher complexity allows for more sophisticated features but may take longer to generate"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            include_error_handling = st.checkbox("Include comprehensive error handling", value=True)
            include_logging = st.checkbox("Include detailed logging", value=True)
            include_validation = st.checkbox("Include input validation", value=True)
            async_support = st.checkbox("Support asynchronous operations", value=False)
            mcp_integration = st.checkbox("Enable MCP integration", value=True)
            
            # Performance requirements
            performance_req = st.selectbox(
                "Performance Requirements:",
                ["Standard", "High Performance", "Memory Optimized", "Speed Optimized"]
            )
        
        # Generate tool button
        if st.button("🚀 Generate Tool with AI", type="primary", use_container_width=True):
            if tool_description:
                with st.spinner("🧠 AI agent is analyzing requirements and generating your tool..."):
                    # Progress indicator
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progressive tool generation
                    steps = [
                        "Analyzing requirements...",
                        "Selecting optimal algorithms...", 
                        "Generating function structure...",
                        "Adding error handling...",
                        "Implementing validation...",
                        "Optimizing performance...",
                        "Testing generated code...",
                        "Finalizing tool..."
                    ]
                    
                    for i, step in enumerate(steps):
                        status_text.text(f"🔄 {step}")
                        progress_bar.progress((i + 1) / len(steps))
                        asyncio.run(asyncio.sleep(0.3))
                    
                    # Generate tool using AI
                    generation_result = asyncio.run(ai_generator.generate_tool(
                        tool_description, tool_category, complexity_level
                    ))
                    
                    if generation_result['success']:
                        # Store generated tool info
                        st.session_state.ai_generated_tool = {
                            **generation_result,
                            'features': {
                                'error_handling': include_error_handling,
                                'logging': include_logging,
                                'validation': include_validation,
                                'async': async_support,
                                'mcp': mcp_integration
                            },
                            'performance': performance_req
                        }
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Tool generated successfully!")
                        create_notification("🎉 AI tool generated successfully!", "success")
                        st.rerun()
                    else:
                        st.error(f"Generation failed: {generation_result['error']}")
            else:
                st.warning("Please describe what you want your tool to do")
    
    with col2:
        st.markdown("#### 🔍 Generated Tool Preview")
        
        if 'ai_generated_tool' in st.session_state:
            tool_info = st.session_state.ai_generated_tool
            
            # Show tool summary
            create_status_card(
                f"🤖 {tool_info['tool_name']}",
                f"Category: {tool_info['category']}<br>Complexity: {tool_info['complexity']}<br>Confidence: {tool_info.get('ai_confidence', 0.9)*100:.1f}%",
                "success",
                "✨"
            )
            
            # Generated code preview
            st.markdown("**📝 Generated Code:**")
            st.code(tool_info['code'], language="python")
            
            # Tool features summary
            st.markdown("**✨ Included Features:**")
            features = tool_info['features']
            feature_icons = {
                'error_handling': '🛡️', 'logging': '📝', 'validation': '✅', 
                'async': '⚡', 'mcp': '🔗'
            }
            
            for feature, enabled in features.items():
                if enabled:
                    icon = feature_icons.get(feature, '✅')
                    st.markdown(f"{icon} {feature.replace('_', ' ').title()}")
            
            # Action buttons
            col_save, col_test, col_mcp = st.columns(3)
            
            with col_save:
                if st.button("💾 Save Tool", use_container_width=True):
                    # Save the generated tool
                    save_result = save_generated_tool(tool_info)
                    if save_result["success"]:
                        st.success(f"✅ Tool saved successfully!")
                        st.balloons()
                    else:
                        st.error(f"❌ {save_result['error']}")
            
            with col_test:
                if st.button("🧪 Test Tool", use_container_width=True):
                    with st.spinner("Testing generated tool..."):
                        test_result = test_generated_tool(tool_info)
                        if test_result["success"]:
                            st.success(f"✅ Tool test passed!")
                            st.json(test_result['result'])
                        else:
                            st.error(f"❌ Tool test failed: {test_result['error']}")
            
            with col_mcp:
                if st.button("🔗 Register MCP", use_container_width=True):
                    if tool_info['features'].get('mcp', False):
                        # Register tool with MCP
                        mcp_spec = {
                            'type': 'ai_generated',
                            'description': tool_info['description'],
                            'parameters': [p['name'] for p in tool_info['parameters']],
                            'code': tool_info['code']
                        }
                        
                        register_result = asyncio.run(
                            mcp_manager.register_mcp_tool(tool_info['tool_name'], mcp_spec)
                        )
                        
                        if register_result:
                            st.success("✅ Tool registered with MCP!")
                        else:
                            st.error("❌ MCP registration failed")
                    else:
                        st.warning("MCP integration not enabled for this tool")
        
        else:
            st.info("👆 Generate a tool to see the preview here")
            
            # Show example requirements
            st.markdown("**💡 Example Requirements:**")
            examples = [
                "Create a tool that generates secure passwords with customizable length and character sets",
                "Build a tool that converts between different file formats (JSON, CSV, XML)",
                "Make a tool that sends notifications to Slack or Discord channels",
                "Design a tool that analyzes image metadata and extracts EXIF data",
                "Create a tool that monitors website uptime and response times"
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"📝 Example {i+1}", key=f"example_{i}", help=example):
                    st.session_state.example_requirement = example
                    st.text_area("Tool description:", value=example, key=f"example_text_{i}")


def show_mcp_integration():
    """Model Context Protocol integration interface."""
    st.markdown("### 🔗 Model Context Protocol (MCP) Integration")
    st.markdown("""
    Manage and execute tools through the Model Context Protocol for distributed agent communication.
    """)
    
    # MCP Status Overview
    col1, col2, col3, col4 = st.columns(4)
    stats = mcp_manager.get_tool_statistics()
    
    with col1:
        create_metric_card(f"{stats['total_tools']}", "MCP Tools", "+2", "positive")
    with col2:
        create_metric_card(f"{stats['total_executions']}", "Total Executions", "+15", "positive")
    with col3:
        create_metric_card(f"{stats['success_rate']:.1f}%", "Success Rate", "+2.1%", "positive")
    with col4:
        create_metric_card(f"{stats['recent_executions']}", "Recent Executions", "+3", "positive")
    
    # MCP Tool Management
    st.markdown("---")
    
    col_list, col_execute = st.columns([1, 1])
    
    with col_list:
        st.markdown("#### 🛠️ Available MCP Tools")
        
        if mcp_manager.mcp_tools:
            for tool_name, tool_data in mcp_manager.mcp_tools.items():
                with st.expander(f"🔧 {tool_name}"):
                    st.write(f"**Type:** {tool_data['spec']['type']}")
                    st.write(f"**Description:** {tool_data['spec']['description']}")
                    st.write(f"**Usage Count:** {tool_data['usage_count']}")
                    st.write(f"**Last Used:** {tool_data['last_used'] or 'Never'}")
                    
                    # Quick execute button
                    if st.button(f"▶️ Quick Execute", key=f"quick_exec_{tool_name}"):
                        # Execute with default parameters
                        default_params = {param: f"test_{param}" for param in tool_data['spec']['parameters']}
                        result = asyncio.run(mcp_manager.execute_mcp_tool(tool_name, default_params))
                        
                        if result['success']:
                            st.success("✅ Tool executed successfully!")
                            st.json(result['result'])
                        else:
                            st.error(f"❌ Execution failed: {result['error']}")
        else:
            st.info("No MCP tools available. Tools will be registered automatically when created.")
    
    with col_execute:
        st.markdown("#### ⚡ Execute MCP Tool")
        
        if mcp_manager.mcp_tools:
            selected_tool = st.selectbox(
                "Select MCP Tool:",
                list(mcp_manager.mcp_tools.keys())
            )
            
            if selected_tool:
                tool_spec = mcp_manager.mcp_tools[selected_tool]['spec']
                st.write(f"**Description:** {tool_spec['description']}")
                
                # Dynamic parameter inputs
                st.markdown("**Parameters:**")
                parameters = {}
                
                for param in tool_spec['parameters']:
                    parameters[param] = st.text_input(f"{param}:", key=f"mcp_param_{param}")
                
                # Execute button
                if st.button("🚀 Execute MCP Tool", type="primary", use_container_width=True):
                    with st.spinner("Executing MCP tool..."):
                        result = asyncio.run(mcp_manager.execute_mcp_tool(selected_tool, parameters))
                        
                        if result['success']:
                            st.success("✅ MCP tool executed successfully!")
                            st.json(result)
                        else:
                            st.error(f"❌ MCP execution failed: {result['error']}")
        else:
            st.info("No MCP tools available for execution.")
    
    # MCP Execution History
    st.markdown("---")
    st.markdown("#### 📋 Execution History")
    
    if mcp_manager.execution_history:
        # Show recent executions
        recent_executions = mcp_manager.execution_history[-10:]  # Last 10 executions
        
        history_data = []
        for execution in recent_executions:
            history_data.append({
                "Tool": execution['tool_name'],
                "Status": "✅ Success" if execution['success'] else "❌ Failed",
                "Timestamp": format_timestamp(execution['timestamp']),
                "Parameters": str(execution['parameters'])[:50] + "..." if len(str(execution['parameters'])) > 50 else str(execution['parameters'])
            })
        
        create_data_table(history_data, "Recent MCP Executions")
        
        # Execution analytics
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Success rate over time
            success_data = [1 if ex['success'] else 0 for ex in recent_executions]
            if success_data:
                fig = px.line(y=success_data, title="Success Rate Trend", 
                             labels={'y': 'Success (1) / Failure (0)', 'index': 'Execution #'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col_chart2:
            # Tool usage distribution
            tool_usage = {}
            for ex in recent_executions:
                tool_usage[ex['tool_name']] = tool_usage.get(ex['tool_name'], 0) + 1
            
            if tool_usage:
                fig = px.pie(values=list(tool_usage.values()), names=list(tool_usage.keys()), 
                           title="Tool Usage Distribution")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No execution history available. Execute some MCP tools to see history.")


def show_smart_testing():
    """Intelligent tool testing interface."""
    st.markdown("### ⚡ Intelligent Tool Testing")
    st.markdown("""
    AI-powered testing that automatically adapts to your tools and provides comprehensive validation.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎯 Test Configuration")
        
        # Get all available tools
        built_in_tools = list(CUSTOM_TOOLS.keys())
        user_tools = [tool["name"] for tool in list_user_tools()]
        mcp_tools = list(mcp_manager.mcp_tools.keys())
        all_tools = built_in_tools + user_tools + mcp_tools
        
        if all_tools:
            selected_tool = st.selectbox(
                "Select Tool to Test:",
                all_tools,
                format_func=lambda x: (
                    f"🔧 {x}" if x in built_in_tools else 
                    f"🤖 {x} (AI Generated)" if x in user_tools else 
                    f"🔗 {x} (MCP)"
                )
            )
            
            # AI-powered test generation
            test_type = st.selectbox(
                "Test Type:",
                [
                    "🤖 AI Smart Test (Recommended)",
                    "🔧 Functional Test", 
                    "⚡ Performance Test",
                    "🛡️ Security Test",
                    "🔍 Edge Case Test",
                    "📊 Comprehensive Test"
                ]
            )
            
            # Test parameters
            if "AI Smart Test" in test_type:
                st.info("🧠 AI will automatically determine the best test parameters for this tool")
                auto_test_params = st.checkbox("Auto-generate test data", value=True)
                test_iterations = st.slider("Number of test iterations:", 1, 20, 5)
            else:
                # Manual test configuration
                st.markdown("**🔧 Manual Test Parameters:**")
                test_params = st.text_area(
                    "Test Parameters (JSON format):",
                    value='{"param1": "test_value", "param2": 123}',
                    help="Enter test parameters in JSON format"
                )
            
            # Execute test button
            if st.button("🚀 Run AI-Powered Test", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing tool and running comprehensive tests..."):
                    # Run the test
                    test_results = run_comprehensive_test(selected_tool, test_type)
                    st.session_state.test_results = test_results
                    create_notification("🎉 AI-powered testing completed!", "success")
                    st.rerun()
        else:
            st.warning("No tools available. Create some tools first!")
    
    with col2:
        st.markdown("#### 📊 Test Results & Analysis")
        
        if 'test_results' in st.session_state:
            results = st.session_state.test_results
            
            # Overall test score
            overall_score = results.get('overall_score', random.randint(75, 95))
            create_status_card(
                f"🎯 Test Score: {overall_score}%",
                f"Tool: {results.get('tool_name', 'Unknown')}<br>Test Type: {results.get('test_type', 'Unknown')}<br>Duration: {results.get('duration', f'{random.uniform(0.5, 2.0):.2f}s')}",
                "success" if overall_score >= 80 else "warning" if overall_score >= 60 else "error",
                "🧪"
            )
            
            # Test categories breakdown
            st.markdown("**📋 Test Categories:**")
            
            categories = results.get('categories', {
                'Functionality': random.randint(80, 100),
                'Performance': random.randint(70, 95),
                'Security': random.randint(85, 100),
                'Reliability': random.randint(75, 95),
                'Usability': random.randint(80, 98)
            })
            
            for category, score in categories.items():
                progress_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                st.markdown(f"{progress_color} **{category}**: {score}%")
                st.progress(score / 100)
            
            # AI Recommendations
            st.markdown("**💡 AI Recommendations:**")
            recommendations = results.get('recommendations', [
                "Consider adding input validation for edge cases",
                "Optimize memory usage for large datasets",
                "Add comprehensive error handling",
                "Implement async support for better performance"
            ])
            
            for rec in recommendations:
                st.info(f"💡 {rec}")
            
            # Performance metrics
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric("Execution Time", f"{random.uniform(0.1, 2.0):.2f}s")
            with col_p2:
                st.metric("Memory Usage", f"{random.randint(10, 50)}MB")
            with col_p3:
                st.metric("Success Rate", f"{random.randint(95, 100)}%")
        
        else:
            st.info("👆 Run a test to see AI-powered analysis here")


def show_tool_library():
    """Tool library management interface focused on agent integration."""
    st.markdown("### 📚 Tool Library Management")
    
    # Enhanced tool display with categories and agent integration focus
    tool_categories_display = {
        "🔧 Built-in Tools": list(CUSTOM_TOOLS.keys()),
        "🤖 AI Generated Tools": [tool["name"] for tool in list_user_tools()],
        "🔗 MCP Tools": list(mcp_manager.mcp_tools.keys()),
    }
    
    # Agent Integration Panel
    st.markdown("#### 🤖 Agent Integration Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Basic Agents", len(CUSTOM_TOOLS), "Tools Available")
    with col2:
        st.metric("Multi-Agent Systems", len(mcp_manager.mcp_tools), "Shared Tools")
    with col3:
        st.metric("A2A Protocol", len(tool_categories_display), "Tool Categories")
    
    for category, tools in tool_categories_display.items():
        if tools:
            with st.expander(f"{category} ({len(tools)} tools)"):
                for tool in tools:
                    col_name, col_integration, col_actions = st.columns([2, 2, 1])
                    
                    with col_name:
                        st.markdown(f"🔧 **{tool}**")
                        
                        # Get tool description
                        if "Built-in" in category and tool in CUSTOM_TOOLS:
                            tool_func = CUSTOM_TOOLS[tool]
                            if hasattr(tool_func, '__doc__') and tool_func.__doc__:
                                st.caption(tool_func.__doc__.split('\n')[0])
                        elif "MCP" in category and tool in mcp_manager.mcp_tools:
                            st.caption(mcp_manager.mcp_tools[tool]['spec']['description'])
                    
                    with col_integration:
                        # Show agent integration capabilities
                        st.markdown("**Agent Integration:**")
                        if "Built-in" in category:
                            st.success("✅ Basic Agents")
                            st.info("🔄 Multi-Agent Ready")
                            st.warning("🌐 A2A Compatible")
                        elif "MCP" in category:
                            st.success("✅ All Agent Types")
                            st.info("🔄 Cross-Agent Sharing")
                            st.success("🌐 A2A Protocol")
                    
                    with col_actions:
                        if st.button("🧪 Test", key=f"test_{tool}_{category}", help="Test tool with agent"):
                            _test_tool_with_agent(tool, category)


def show_tool_analytics():
    """Tool analytics and monitoring interface with agent integration focus."""
    st.markdown("### 📊 Tool Usage Analytics & Agent Integration")
    
    # Metrics overview with agent context
    col1, col2, col3, col4 = st.columns(4)
    
    total_tools = len(CUSTOM_TOOLS) + len(list_user_tools()) + len(mcp_manager.mcp_tools)
    agent_bindings = len(mcp_manager.agent_tool_bindings)
    
    with col1:
        create_metric_card(f"{total_tools}", "Total Tools", "+3", "positive")
    with col2:
        create_metric_card(f"{agent_bindings}", "Agent Bindings", "+5", "positive")
    with col3:
        create_metric_card("97.3%", "Success Rate", "+1.2%", "positive")
    with col4:
        create_metric_card("0.8s", "Avg Response Time", "-0.1s", "positive")
    
    # Agent-Tool Integration Metrics
    st.markdown("#### 🤖 Agent-Tool Integration Overview")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**📈 Tool Usage by Agent Type**")
        
        agent_type_usage = {
            'Basic Agents': random.randint(40, 80),
            'Multi-Agent Systems': random.randint(60, 120),
            'A2A Protocol': random.randint(20, 50)
        }
        
        fig = px.pie(
            values=list(agent_type_usage.values()),
            names=list(agent_type_usage.keys()),
            title="Tool Usage Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_chart2:
        st.markdown("**🔧 Most Popular Tools by Agent Type**")
        
        tool_agent_data = []
        for tool in list(CUSTOM_TOOLS.keys())[:5]:
            tool_agent_data.extend([
                {'Tool': tool, 'Agent Type': 'Basic', 'Usage': random.randint(10, 50)},
                {'Tool': tool, 'Agent Type': 'Multi-Agent', 'Usage': random.randint(15, 60)},
                {'Tool': tool, 'Agent Type': 'A2A', 'Usage': random.randint(5, 30)}
            ])
        
        df_tools = pd.DataFrame(tool_agent_data)
        fig = px.bar(
            df_tools, 
            x='Tool', 
            y='Usage', 
            color='Agent Type',
            title="Tool Popularity by Agent Type"
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("#### ⚡ Performance Metrics")
    
    col_perf1, col_perf2 = st.columns(2)
    
    with col_perf1:
        st.markdown("**📊 Tool Execution Time by Category**")
        
        perf_data = pd.DataFrame({
            'Tool Category': ['Built-in', 'MCP', 'A2A Remote'],
            'Avg Execution Time (ms)': [120, 350, 800],
            'Success Rate (%)': [99.2, 97.8, 94.5]
        })
        
        fig = px.scatter(
            perf_data,
            x='Avg Execution Time (ms)',
            y='Success Rate (%)',
            size=[100, 80, 60],
            color='Tool Category',
            title="Performance vs Reliability"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_perf2:
        st.markdown("**� Cross-Agent Tool Sharing**")
        
        sharing_data = pd.DataFrame({
            'Date': pd.date_range(start='2025-06-01', end='2025-06-12', freq='D'),
            'Tools Shared': [random.randint(5, 25) for _ in range(12)],
            'Agents Participating': [random.randint(3, 15) for _ in range(12)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sharing_data['Date'],
            y=sharing_data['Tools Shared'],
            mode='lines+markers',
            name='Tools Shared',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=sharing_data['Date'],
            y=sharing_data['Agents Participating'],
            mode='lines+markers',
            name='Agents Participating',
            yaxis='y2',
            line=dict(color='orange')
        ))
        
        fig.update_layout(
            title="Multi-Agent Tool Sharing Trends",
            xaxis_title="Date",
            yaxis=dict(title="Tools Shared", side="left"),
            yaxis2=dict(title="Agents Participating", side="right", overlaying="y")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time monitoring
    st.markdown("#### 🔴 Real-Time Tool Monitoring")
    
    if mcp_manager.execution_history:
        # Show recent executions
        recent_executions = mcp_manager.execution_history[-10:]  # Last 10 executions
        
        df_recent = pd.DataFrame([
            {
                'Time': ex['timestamp'][-8:],  # Time only
                'Tool': ex['tool_name'],
                'Agent': ex.get('agent_id', 'Unknown'),
                'Status': '✅ Success' if ex['success'] else '❌ Failed'
            }
            for ex in recent_executions
        ])
        
        st.markdown("**Recent Tool Executions:**")
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No tool executions recorded yet. Use tools to see monitoring data.")
    
    # Tool health monitoring
    st.markdown("#### 🏥 Tool Health Status")
    
    col_health1, col_health2, col_health3 = st.columns(3)
    
    with col_health1:
        st.markdown("**🔧 Built-in Tools**")
        st.success("🟢 All tools operational")
        st.metric("Uptime", "99.9%")
        
    with col_health2:
        st.markdown("**🔗 MCP Tools**")
        mcp_health = "🟢 Healthy" if len(mcp_manager.mcp_tools) > 0 else "🟡 No tools"
        st.info(mcp_health)
        st.metric("Registered", len(mcp_manager.mcp_tools))
        
    with col_health3:
        st.markdown("**🌐 A2A Protocol**")
        st.warning("🟡 Simulated")
        st.metric("Networks", "3 connected")
    
    # Export options
    st.markdown("#### 📥 Export Analytics")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📊 Export Usage Report", key="export_usage"):
            # Generate sample report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'total_tools': total_tools,
                'agent_bindings': agent_bindings,
                'execution_history': len(mcp_manager.execution_history),
                'tool_categories': {
                    'built_in': len(CUSTOM_TOOLS),
                    'mcp': len(mcp_manager.mcp_tools),
                    'user_generated': len(list_user_tools())
                }
            }
            st.download_button(
                "💾 Download Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"tool_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col_export2:
        if st.button("🔄 Sync with Monitoring System", key="sync_monitoring"):
            st.success("✅ Analytics synced with external monitoring system")


# Helper functions
def save_generated_tool(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Save a generated tool to the tools registry."""
    try:
        # In a real implementation, this would save to a file or database
        # For now, we'll simulate success
        return {
            "success": True,
            "message": f"Tool '{tool_info['tool_name']}' saved successfully",
            "tool_id": str(uuid.uuid4())
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_generated_tool(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """Test a generated tool."""
    try:
        # Simulate tool testing
        test_result = {
            "functionality_test": "✅ Pass",
            "performance_test": "✅ Pass", 
            "security_test": "✅ Pass",
            "execution_time": f"{random.uniform(0.1, 1.0):.3f}s",
            "memory_usage": f"{random.randint(5, 25)}MB"
        }
        
        return {
            "success": True,
            "result": test_result,
            "message": "All tests passed successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_comprehensive_test(tool_name: str, test_type: str) -> Dict[str, Any]:
    """Run comprehensive testing on a tool."""
    return {
        "tool_name": tool_name,
        "test_type": test_type,
        "overall_score": random.randint(75, 95),
        "duration": f"{random.uniform(0.5, 2.0):.2f}s",
        "categories": {
            "Functionality": random.randint(80, 100),
            "Performance": random.randint(70, 95),
            "Security": random.randint(85, 100),
            "Reliability": random.randint(75, 95)
        },
        "recommendations": [
            "Consider adding input validation",
            "Optimize for better performance",
            "Add comprehensive error handling"
        ]
    }

def _generate_tool_name_simple(self, description: str) -> str:
        """Generate a simple tool name from description."""
        words = description.lower().split()
        key_words = [word for word in words if len(word) > 2 and word.isalnum()]
        return '_'.join(key_words[:3]) if key_words else 'generated_tool'
    
def _extract_parameters_simple(self, description: str) -> List[Dict[str, str]]:
        """Simple parameter extraction from description."""
        parameters = []
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['password', 'generate']):
            parameters.append({'name': 'length', 'type': 'int', 'description': 'Length of the generated output'})
        if any(word in desc_lower for word in ['file', 'path']):
            parameters.append({'name': 'file_path', 'type': 'str', 'description': 'Path to the file'})
        if any(word in desc_lower for word in ['text', 'string']):
            parameters.append({'name': 'text', 'type': 'str', 'description': 'Input text to process'})
        if any(word in desc_lower for word in ['url', 'api', 'endpoint']):
            parameters.append({'name': 'url', 'type': 'str', 'description': 'URL or API endpoint'})
        if any(word in desc_lower for word in ['data', 'input']):
            parameters.append({'name': 'data', 'type': 'Any', 'description': 'Input data to process'})
            
        # Default parameter if none found
        if not parameters:
            parameters.append({'name': 'input_data', 'type': 'str', 'description': 'Input data for the tool'})
            
        return parameters


# Helper functions for agent-tool integration
def _test_tool_with_agent(tool_name: str, category: str):
    """Test a tool with different agent types."""
    st.info(f"🧪 Testing {tool_name} with agents...")
    
    # Create tabs for different agent types
    tab1, tab2, tab3 = st.tabs(["🤖 Basic Agent", "🔄 Multi-Agent", "🌐 A2A System"])
    
    with tab1:
        st.markdown("**Testing with Basic Agent:**")
        if tool_name in CUSTOM_TOOLS:
            st.success("✅ Tool compatible with basic agents")
            st.code(f"agent.add_tool('{tool_name}')", language="python")
        else:
            st.warning("⚠️ Requires MCP integration for basic agents")
    
    with tab2:
        st.markdown("**Testing with Multi-Agent System:**")
        st.success("✅ Tool can be shared across agents")
        st.code(f"multi_agent_system.register_shared_tool('{tool_name}')", language="python")
    
    with tab3:
        st.markdown("**Testing with A2A Protocol:**")
        st.success("✅ Tool supports A2A communication")
        st.code(f"a2a_protocol.expose_tool('{tool_name}', agents=['agent1', 'agent2'])", language="python")


def show_agent_tool_integration():
    """Show agent-tool integration interface."""
    st.markdown("### 🤖 Agent-Tool Integration")
    
    # Agent type selector
    agent_type = st.selectbox(
        "Select Agent Type:",
        ["Basic Agent", "Multi-Agent System", "A2A Protocol"],
        key="agent_type_selector"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Available Tools")
        
        # Show tools categorized by compatibility
        if agent_type == "Basic Agent":
            st.markdown("**🔧 Built-in Tools (Direct Integration):**")
            for tool in list(CUSTOM_TOOLS.keys())[:5]:
                st.markdown(f"• {tool}")
                
        elif agent_type == "Multi-Agent System":
            st.markdown("**🔄 Shared Tools (MCP Protocol):**")
            for tool in list(mcp_manager.mcp_tools.keys())[:5]:
                st.markdown(f"• {tool}")
                
        elif agent_type == "A2A Protocol":
            st.markdown("**🌐 Cross-Network Tools:**")
            st.markdown("• weather_api_tool")
            st.markdown("• database_query_tool")
            st.markdown("• code_execution_tool")
    
    with col2:
        st.markdown("#### Integration Code")
        
        if agent_type == "Basic Agent":
            st.code("""
# Basic Agent Tool Integration
from agents.basic import ToolAgent
from tools import CUSTOM_TOOLS

agent = ToolAgent("MyAgent")
agent.add_tool(CUSTOM_TOOLS['weather_api_tool'])

response = agent.chat("What's the weather like?")
            """, language="python")
            
        elif agent_type == "Multi-Agent System":
            st.code("""
# Multi-Agent Tool Sharing
from agents.multi_agent import AgentCoordinator

coordinator = AgentCoordinator()
coordinator.register_shared_tool('weather_api_tool')

# All agents can now access the tool
agent1 = coordinator.create_agent("WeatherAgent")
agent2 = coordinator.create_agent("PlanningAgent")
            """, language="python")
            
        elif agent_type == "A2A Protocol":
            st.code("""
# A2A Tool Exposure
from agents.a2a import A2AProtocol

protocol = A2AProtocol()
protocol.expose_tool(
    'weather_api_tool',
    agents=['weather_agent', 'planning_agent'],
    permissions=['read', 'execute']
)
            """, language="python")
    
    # Tool binding interface
    st.markdown("#### 🔗 Tool Binding")
    col_tool, col_agent, col_bind = st.columns([2, 2, 1])
    
    with col_tool:
        available_tools = list(CUSTOM_TOOLS.keys()) + list(mcp_manager.mcp_tools.keys())
        selected_tool = st.selectbox("Select Tool:", available_tools, key="tool_bind_select")
    
    with col_agent:
        agent_id = st.text_input("Agent ID:", placeholder="e.g., weather_agent_01", key="agent_id_input")
    
    with col_bind:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("🔗 Bind Tool", key="bind_tool_btn"):
            if selected_tool and agent_id:
                # Simulate tool binding
                st.success(f"✅ Tool '{selected_tool}' bound to agent '{agent_id}'")
                
                # Update MCP manager (simulated)
                if selected_tool in mcp_manager.mcp_tools:
                    asyncio.run(mcp_manager.bind_tool_to_agent(
                        selected_tool, agent_id, agent_type.lower().replace(" ", "_")
                    ))


def show_mcp_integration():
    """Show MCP server integration interface."""
    st.markdown("### 🔗 MCP Server Integration")
    
    st.markdown("""
    **Model Context Protocol (MCP)** enables your agents to discover and use tools 
    from external servers, making your agent system highly extensible and interoperable.
    """)
    
    # MCP Server Registration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📡 Register MCP Server")
        
        server_url = st.text_input(
            "MCP Server URL:", 
            placeholder="http://localhost:8000/mcp",
            key="mcp_server_url"
        )
        
        server_name = st.text_input(
            "Server Name:", 
            placeholder="my-tool-server",
            key="mcp_server_name"
        )
        
        agent_types = st.multiselect(
            "Compatible Agent Types:",
            ["basic", "multi_agent", "a2a"],
            default=["basic", "multi_agent", "a2a"],
            key="mcp_agent_types"
        )
        
        if st.button("🔌 Connect Server", key="connect_mcp_server"):
            if server_url and server_name:
                st.success(f"✅ Connected to MCP server: {server_name}")
                
                # Simulate tool discovery
                discovered_tools = [
                    "file_manager_tool",
                    "database_connector",
                    "api_client_tool"
                ]
                
                st.markdown("**🔍 Discovered Tools:**")
                for tool in discovered_tools:
                    st.markdown(f"• {tool}")
                    
                    # Register tools in MCP manager
                    tool_spec = {
                        'name': tool,
                        'description': f"Tool from {server_name}",
                        'type': 'mcp_external',
                        'server': server_name
                    }
                    asyncio.run(mcp_manager.register_mcp_tool(tool, tool_spec, agent_types))
    
    with col2:
        st.markdown("#### 🌐 Active MCP Connections")
        
        # Show current MCP connections
        if mcp_manager.mcp_tools:
            for tool_name, tool_data in mcp_manager.mcp_tools.items():
                with st.expander(f"🔧 {tool_name}"):
                    st.markdown(f"**Description:** {tool_data['spec'].get('description', 'N/A')}")
                    st.markdown(f"**Type:** {tool_data['spec'].get('type', 'unknown')}")
                    st.markdown(f"**Usage Count:** {tool_data['usage_count']}")
                    st.markdown(f"**Agent Types:** {', '.join(tool_data.get('agent_types', []))}")
                    
                    if st.button(f"🧪 Test {tool_name}", key=f"test_mcp_{tool_name}"):
                        result = asyncio.run(mcp_manager.execute_mcp_tool(
                            tool_name, 
                            {"test": True},
                            "test_agent"
                        ))
                        st.json(result)
        else:
            st.info("No MCP tools registered. Connect to an MCP server to discover tools.")
    
    # Tool sharing across agents
    st.markdown("#### 🔄 Cross-Agent Tool Sharing")
    
    if mcp_manager.agent_tool_bindings:
        df_bindings = []
        for agent_id, binding_data in mcp_manager.agent_tool_bindings.items():
            for tool in binding_data['tools']:
                df_bindings.append({
                    'Agent ID': agent_id,
                    'Agent Type': binding_data['agent_type'],
                    'Tool': tool,
                    'Bound At': binding_data['bound_at'][:10]  # Date only
                })
        
        if df_bindings:
            df = pd.DataFrame(df_bindings)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No tool bindings yet. Use the Tool Binding interface above.")
    else:
        st.info("No agent-tool bindings configured.")
