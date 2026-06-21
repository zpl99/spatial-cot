import ast
import networkx as nx
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import deque
import json
from engine.base_engine import LLMEngine
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import pickle 

# --- OpenAI API and LLM Setup ---
# These agents power the graph-based retrieval used by Spatial CoT+ (embeddings
# via text-embedding-3-small, plus an LLM for verify/refine). They are created
# lazily on first use so that strategies which do not use retrieval
# (io / cot / spatial_cot) can run without an OPENAI_API_KEY.
text_embedding_agent = None
rag_agent = None
embedding_cache = {}


def _get_text_embedding_agent():
    global text_embedding_agent
    if text_embedding_agent is None:
        text_embedding_agent = LLMEngine("text-embedding-3-small")
    return text_embedding_agent


def _get_rag_agent():
    global rag_agent
    if rag_agent is None:
        rag_agent = LLMEngine("gpt-4o")
    return rag_agent


def visualize_graph(rag_system: 'SCT_GraphRAG', layout_type='hierarchical', show_details=True):
    """
    Creates and displays an enhanced visualization of the knowledge graph with improved
    aesthetics, hierarchical layout, and adaptive node sizing.
    
    Args:
        rag_system: The SCT_GraphRAG system instance
        layout_type: Layout algorithm ('hierarchical', 'spring', 'circular', or 'kamada_kawai')
        show_details: If True, shows labels, legends, and adaptive node sizes. 
                     If False, shows only graph structure with uniform node sizes (for large graphs)
    """
    
    def wrap_text(text, max_width=20):
        """
        Wraps text to fit within a specified width, breaking on spaces when possible.
        
        Args:
            text: The text to wrap
            max_width: Maximum characters per line
        
        Returns:
            Text with newlines inserted for wrapping
        """
        if len(text) <= max_width:
            return text
        
        words = text.split(' ')
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # If adding this word would exceed max_width
            if current_length + len(word) + len(current_line) > max_width:
                if current_line:  # If there's already content in current line
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:  # Word itself is longer than max_width
                    # Break the word
                    lines.append(word[:max_width])
                    current_line = [word[max_width:]] if len(word) > max_width else []
                    current_length = len(word[max_width:]) if len(word) > max_width else 0
            else:
                current_line.append(word)
                current_length += len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    G = rag_system.graph
    
    # Enhanced color scheme with gradient effects
    node_colors = {
        'question': '#4A90E2',      # Professional blue
        'concept': '#7ED321',        # Vibrant green
        'transformation': '#F5A623', # Warm orange
        'default': '#9B9B9B'         # Neutral gray
    }
    
    # Separate nodes by type for better visualization control
    question_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'question']
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'concept']
    transformation_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'transformation']
    
    # Calculate node importance (degree centrality) for adaptive sizing (only used if show_details=True)
    degree_centrality = nx.degree_centrality(G) if show_details else {}
    
    # Create labels only if show_details is True
    labels = {}
    if show_details:
        for node, data in G.nodes(data=True):
            node_type = data.get('type')
            
            if node_type == 'question':
                content = data['content']
                # Wrap question text to fit within node (max 25 chars per line)
                wrapped_content = wrap_text(content, max_width=25)
                # Limit to first 4 lines to avoid overly tall nodes
                lines = wrapped_content.split('\n')
                if len(lines) > 4:
                    wrapped_content = '\n'.join(lines[:4]) + '...'
                labels[node] = f"Q:\n{wrapped_content}"
            
            elif node_type == 'concept':
                entity = data.get('entity', 'N/A')
                core_concept = data.get('core_concept', 'N/A')
                # Wrap entity name if needed (max 20 chars per line)
                wrapped_entity = wrap_text(entity, max_width=20)
                labels[node] = f"{wrapped_entity}\n[{core_concept}]"
            
            elif node_type == 'transformation':
                rule_label = data['rule']
                # Wrap transformation rule (max 22 chars per line)
                wrapped_rule = wrap_text(rule_label, max_width=22)
                # Limit to first 5 lines
                lines = wrapped_rule.split('\n')
                if len(lines) > 5:
                    wrapped_rule = '\n'.join(lines[:5]) + '...'
                labels[node] = f"T:\n{wrapped_rule}"
            else:
                labels[node] = wrap_text(str(node), max_width=20)
    
    # Edge styling with improved color differentiation
    edge_colors_map = {
        'USES_CONCEPT': '#7CB342',      # Medium green - clearly distinct
        'HAS_TRANSFORMATION': '#FFA726', # Deep orange - warm and visible
        'PART_OF_PATH': '#E53935',      # Bright red - emphasis on path
        'INPUT_TO': '#1E88E5',          # Blue - clear and professional
        'OUTPUT_OF': '#8E24AA'          # Purple - distinct from all others
    }
    
    edge_labels = nx.get_edge_attributes(G, 'type')
    edge_color_list = []
    edge_width_list = []
    for edge in G.edges():
        edge_type = G.edges[edge].get('type', 'default')
        edge_color_list.append(edge_colors_map.get(edge_type, '#CCCCCC'))
        # Make PART_OF_PATH edges thicker to show transformation sequence
        if edge_type == 'PART_OF_PATH':
            edge_width_list.append(4.0)
        else:
            edge_width_list.append(2.5)
    
    # Create figure with clean styling (no title or subtitle, transparent background)
    fig, ax = plt.subplots(figsize=(42, 32), facecolor='none')  # Increased from (36, 28) to (42, 32)
    
    # Choose layout algorithm
    if layout_type == 'hierarchical':
        # Try to create a hierarchical layout based on node types
        pos = {}
        layer_spacing = 1.5
        
        # Position question nodes on the left
        for i, node in enumerate(question_nodes):
            pos[node] = (-2, i * 0.8 - len(question_nodes) * 0.4)
        
        # Position concept nodes in the middle
        for i, node in enumerate(concept_nodes):
            pos[node] = (0, i * 0.3 - len(concept_nodes) * 0.15)
        
        # Position transformation nodes on the right
        for i, node in enumerate(transformation_nodes):
            pos[node] = (2, i * 0.5 - len(transformation_nodes) * 0.25)
        
        # Apply spring layout refinement to avoid overlaps with tighter spacing
        pos = nx.spring_layout(G, pos=pos, k=0.2, iterations=50, fixed=None)  # Reduced k from 0.5 to 0.2
    
    elif layout_type == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, scale=0.3)  # Further reduced from 0.5 to 0.3
    elif layout_type == 'circular':
        pos = nx.circular_layout(G, scale=0.3)  # Further reduced from 0.5 to 0.3
    else:  # spring
        pos = nx.spring_layout(G, k=0.15, iterations=100, seed=42)  # k reduced from 0.3 to 0.15 for even tighter layout
    
    # Draw edges with varying styles
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_color_list,
                          width=edge_width_list,
                          alpha=0.5,
                          arrows=True,
                          arrowsize=25,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.15',
                          ax=ax,
                          node_size=1)  # Prevent edge from being drawn under nodes
    
    # Draw nodes by type with adaptive sizing (increased base size to match larger font)
    for node_list, color, label in [
        (question_nodes, node_colors['question'], 'Question'),
        (concept_nodes, node_colors['concept'], 'Concept'),
        (transformation_nodes, node_colors['transformation'], 'Transformation')
    ]:
        if node_list:
            # Calculate sizes based on mode
            if show_details:
                # Significantly larger node sizes to ensure text stays within boundaries
                # Increased substantially to prevent text overflow with 16pt font
                if label in ['Question', 'Transformation']:
                    sizes = [18000 + degree_centrality.get(node, 0) * 30000 for node in node_list]
                else:
                    sizes = [14000 + degree_centrality.get(node, 0) * 25000 for node in node_list]
            else:
                # Uniform sizing for structure-only mode
                sizes = [300 for _ in node_list]  # Small uniform size for large graphs
            
            node_pos = {k: v for k, v in pos.items() if k in node_list}
            
            nx.draw_networkx_nodes(G, pos,
                                  nodelist=node_list,
                                  node_color=color,
                                  node_size=sizes,
                                  alpha=0.9,
                                  edgecolors='white',
                                  linewidths=4.5 if show_details else 1.5,  # Also increased border width slightly
                                  ax=ax)
    
    # Draw labels with improved readability and larger font for better visibility (only if show_details=True)
    if show_details:
        nx.draw_networkx_labels(G, pos,
                               labels=labels,
                               font_size=16,  # Increased from 12 to 16 for even better readability
                               font_weight='bold',
                               font_color='#1A1A1A',
                               font_family='sans-serif',
                               ax=ax)
    
    # Edge labels are removed to avoid clutter - legend provides edge type information
    # (Previously we showed edge labels here, but it made the visualization too busy)
    
    # Create comprehensive legend with all edge types (only if show_details=True)
    if show_details:
        legend_elements = [
            # Node types
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=node_colors['question'], 
                       markersize=20, label=f'Question ({len(question_nodes)})', 
                       markeredgecolor='white', markeredgewidth=4),
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=node_colors['concept'], 
                       markersize=20, label=f'Concept ({len(concept_nodes)})', 
                       markeredgecolor='white', markeredgewidth=4),
            plt.Line2D([0], [0], marker='o', color='w', 
                       markerfacecolor=node_colors['transformation'], 
                       markersize=20, label=f'Transformation ({len(transformation_nodes)})', 
                       markeredgecolor='white', markeredgewidth=4),
            plt.Line2D([0], [0], color='white', linewidth=0, label=''),  # Spacer
            # Edge types - all 5 types
            plt.Line2D([0], [0], color=edge_colors_map['USES_CONCEPT'], 
                       linewidth=4, label='Uses Concept'),
            plt.Line2D([0], [0], color=edge_colors_map['HAS_TRANSFORMATION'], 
                       linewidth=4, label='Has Transformation'),
            plt.Line2D([0], [0], color=edge_colors_map['PART_OF_PATH'], 
                       linewidth=5, label='Part of Path'),
            plt.Line2D([0], [0], color=edge_colors_map['INPUT_TO'], 
                       linewidth=4, label='Input To'),
            plt.Line2D([0], [0], color=edge_colors_map['OUTPUT_OF'], 
                       linewidth=4, label='Output Of'),
        ]
        
        # Place legend in upper right corner with larger font sizes
        legend = ax.legend(handles=legend_elements, 
                          loc='upper right',
                          fontsize=14,  # Increased from 11 to 14
                          frameon=True, 
                          fancybox=True, 
                          shadow=True, 
                          title='Graph Elements', 
                          title_fontsize=17,  # Increased from 13 to 17
                          edgecolor='#BDC3C7',
                          facecolor='white',
                          framealpha=0.9)  # Slightly more transparent
        legend.get_frame().set_linewidth(2.5)  # Increased from 2 to 2.5
    
    # Statistics box removed for cleaner visualization
    
    ax.axis('off')
    ax.set_facecolor('none')  # Remove axis background
    fig.patch.set_alpha(0.0)  # Make figure background transparent
    
    # Set axis limits to ensure all nodes are visible with padding
    x_values = [pos[node][0] for node in G.nodes()]
    y_values = [pos[node][1] for node in G.nodes()]
    x_margin = (max(x_values) - min(x_values)) * 0.15  # 15% padding
    y_margin = (max(y_values) - min(y_values)) * 0.15
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    plt.tight_layout(pad=2.0)  # Add padding around the edges
    
    # Save the figure with transparent background
    output_filename = f'knowledge_graph_{layout_type}.png'
    plt.savefig(output_filename, 
                dpi=300, 
                bbox_inches='tight', 
                transparent=True,  # Ensure transparent background
                facecolor='none',
                edgecolor='none')
    
    # Print statistics based on mode
    if show_details:
        print("\n" + "="*80)
        print("📊 Knowledge Graph Visualization Generated Successfully!")
        print("="*80)
        print(f"   Layout: {layout_type.upper()}")
        print(f"   Total Nodes: {G.number_of_nodes()} ({len(question_nodes)} Q + {len(concept_nodes)} C + {len(transformation_nodes)} T)")
        print(f"   Total Edges: {G.number_of_edges()}")
        print(f"   Node sizes are proportional to their importance (degree centrality)")
        print(f"   💾 Saved to: {output_filename}")
        print("="*80)
        print("   💡 Tip: Close the plot window to continue the script")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("📊 Knowledge Graph Structure Visualization (Simplified Mode)")
        print("="*80)
        print(f"   Layout: {layout_type.upper()}")
        print(f"   Total Nodes: {G.number_of_nodes()} ({len(question_nodes)} Q + {len(concept_nodes)} C + {len(transformation_nodes)} T)")
        print(f"   Total Edges: {G.number_of_edges()}")
        print(f"   Mode: Structure-only (uniform node sizes, no labels/legends)")
        print(f"   💾 Saved to: {output_filename}")
        print("="*80)
        print("   💡 Tip: Use show_details=True for detailed view with labels")
        print("="*80 + "\n")
    
    plt.show()

def parse_data(file_content: str) -> list:
    """
    Parses the provided text file to extract questions, concepts, and transformation paths.
    """
    examples = []
    entries = file_content.strip().split('==================================================')
    for entry in entries:
        if not entry.strip():
            continue
        lines = [line.strip() for line in entry.strip().split('\n')]
        question = lines[0].replace('Q:', '').strip()
        try:
            concepts_start_idx = lines.index('Concepts:') + 1
        except ValueError:
            concepts_start_idx = -1
        try:
            extent_start_idx = lines.index('Extent:')
        except ValueError:
            extent_start_idx = -1
        try:
            transformations_start_idx = lines.index('Transformations:') + 1
        except ValueError:
            transformations_start_idx = -1
        concepts_end_idx = extent_start_idx if extent_start_idx != -1 else transformations_start_idx - 1 if transformations_start_idx != -1 else len(
            lines)
        transformations_end_idx = len(lines)
        concepts = {}
        if concepts_start_idx != -1:
            concept_lines = lines[concepts_start_idx:concepts_end_idx]
            for line in concept_lines:
                if line.startswith('-'):
                    match = re.match(r'- \[(\d+)\] (.*)', line)
                    if match:
                        concept_id, concept_def = match.groups()
                        parts = concept_def.split(':', 1)
                        concept_name = parts[0].strip()
                        if not concept_name and len(parts) > 1:
                            concept_name = parts[1].strip().split('(')[0].strip()
                        if not concept_name:
                            concept_name = f"unspecified_{concept_id}"
                        concepts[concept_id] = {"full_def": concept_def, "name": concept_name}
        transformations = []
        if transformations_start_idx != -1:
            transformation_lines = lines[transformations_start_idx:transformations_end_idx]
            for line in transformation_lines:
                if line:
                    transformations.append(line)
        examples.append({"question": question, "concepts": concepts, "transformations": transformations})
    return examples


# --- Utility and LLM Interaction Functions ---

def get_openai_embedding(text: str) -> np.ndarray:
    """Gets and caches text embeddings."""
    if text in embedding_cache:
        return embedding_cache[text]
    try:
        response = _get_text_embedding_agent().get_embedding(text)
        embedding = np.array(response)
        embedding_cache[text] = embedding
        return embedding
    except Exception as e:
        print(f"An error occurred while fetching embedding for '{text}': {e}")
        return np.zeros(1536)


def call_llm(prompt: list, temperature=0.2) -> str:
    """Wrapper for the LLM engine call. Returns the response text."""
    agent = _get_rag_agent()
    result = agent.respond(prompt, temperature=temperature)
    # result is the tuple (text, prompt_tokens, completion_tokens)
    text, prompt_tokens, completion_tokens = result

    # Accumulate token usage on the agent (for external tracking)
    if not hasattr(agent, 'total_prompt_tokens'):
        agent.total_prompt_tokens = 0
        agent.total_completion_tokens = 0

    agent.total_prompt_tokens += prompt_tokens
    agent.total_completion_tokens += completion_tokens

    return text


# ==============================================================================
# --- LLM FUNCTIONS FOR THE STP-IRR FRAMEWORK ---
# ==============================================================================
def llm_check_goal_achieved_with_context(
        query: str,
        final_concept: Dict[str, str],
        known_concepts: List[Dict[str, str]],
        transformation_path: List[str]
) -> bool:
    """
    Uses an LLM to semantically determine if the final goal has been achieved,
    considering the transformation path that has been constructed so far.
    """
    prompt = f"""
You are a senior GIScience analyst acting as a final reviewer. Your task is to determine if the reasoning process, represented by the "Transformation Path," is now complete and sufficient to answer the "Original User Query."

A process is considered **complete ("true")** if:
1.  The final concept produced by the last step of the Transformation Path directly and logically answers the user's main question.
2.  The Transformation Path itself forms a coherent and complete workflow.

A process is considered **incomplete ("false")** if:
1.  The Transformation Path is empty, and the initial concepts alone are not sufficient to answer the query.
2.  The path has only produced intermediate concepts (like 'cost surface' or 'boolean field') but has not yet generated the final entity the user is asking for (like 'least cost route' or filtered 'houses').

**Instructions**:
- Analyze the entire context.
- Return only "true" or "false".
- Ignore minor semantic differences (e.g., 'house' vs 'houses', 'route' vs 'path').

---
**Example 1: Goal NOT Achieved (Path is empty)**
Original User Query: "What houses are for sale and within 0.5km from the main roads in Utrecht"
Final Goal Concept: {{"entity": "houses", "core_concept": "object"}}
List of Known Concepts: [{{"entity": "main roads", "core_concept": "object"}}, {{"entity": "houses", "core_concept": "object"}}]
Transformation Path: []
Evaluation: false
(Reasoning: The path is empty. No filtering has been applied to the 'houses' object yet.)
---
**Example 2: Goal Achieved (Path is logically complete)**
Original User Query: "What is the least cost route from school to the closest road intersection based on slope and land use in Utrecht"
Final Goal Concept: {{"entity": "least cost route", "core_concept": "field"}}
List of Known Concepts: [..., {{"entity": "cost surface", "core_concept": "field"}}, {{"entity": "least cost path", "core_concept": "field"}}]
Transformation Path: ["[slope] + [land use] -> [cost surface]", "[school] + [cost surface] -> [least cost path]"]
Evaluation: true
(Reason-ing: The last step produced 'least cost path', which semantically matches the 'least cost route' goal and logically concludes the workflow.)
---
**Example 3: Goal NOT Achieved (Path is incomplete)**
Original User Query: "What houses are for sale and within 0.5km from the main roads in Utrecht"
Final Goal Concept: {{"entity": "houses", "core_concept": "object"}}
List of Known Concepts: [..., {{"entity": "distance field", "core_concept": "field"}}, {{"entity": "boolean field", "core_concept": "field"}}]
Transformation Path: ["[main roads] -> [distance field]", "[distance field] -> [boolean field]"]
Evaluation: false
(Reasoning: The path has successfully handled the distance constraint by creating a 'boolean field', but it has not yet applied this filter to the 'houses' object.)
---

**Task to perform:**
Original User Query: "{query}"
Final Goal Concept: {final_concept}
List of Known Concepts: {known_concepts}
Transformation Path: {transformation_path}
Evaluation (only "true" or "false", without any explanation):
"""
    user_input = [{"role": "user", "content": prompt}]
    response = call_llm(user_input).strip().lower()

    print(f"Goal Check: Is '{final_concept['entity']}' achieved given the path? LLM says: {response}")
    return response == "true"

def llm_decompose_query_structured(query: str) -> List[Dict[str, str]]:
    """Decomposes the query into a structured list of entities and their core concepts."""
    prompt = f"""
Your task is to act as a specialized GIScience analyst. From the user's question, extract all key spatial entities and identify their corresponding Spatial Core Concept type.

Follow these rules:
1.  Identify the specific real-world objects, phenomena, or analytical products.
2.  For each entity, determine its core concept type (e.g., object, field, event, network, objectquality, conamount).
3.  Return the results as a JSON list of objects, where each object has two keys: "entity" and "core_concept".
4.  All strings should be lowercase.

Here are the descriptions of spatial core concepts for your reference:
  - **Location**: Spatial reference describing where something is. Used in spatial distribution and geometry.
  - **Field**: Continuously varying values across space (e.g., elevation, distance, land use). Supports interpolation and aggregation.
  - **Object**: Discrete bounded entities with identity and attributes (e.g., buildings, trees, parks).
  - **Event**: Time-bound spatial occurrences with location and features (e.g., fires, trips).
  - **Network**: Structured spatial relationships among entities (e.g., roads, connections, flows).
  - **Amount**:
    - *Content Amount*: Aggregated values (count, sum, average).
    - *Coverage Amount*: Spatial extent (area, length, cluster size).
  - **Proportion**: Ratio between two amounts (e.g., density, rate), capturing relative quantities.


---
**Example 1:**
Question: "What is the Euclidean distance to recreational sites in Utrecht"
JSON Output:
[
    {{"entity": "euclidean distance", "core_concept": "field"}},
    {{"entity": "recreational sites", "core_concept": "object"}}
]
---
**Example 2:**
Question: "What is the least cost route from school to the closest road intersection based on slope and land use in Utrecht"
JSON Output:
[
    {{"entity": "least cost route", "core_concept": "field"}},
    {{"entity": "school", "core_concept": "object"}},
    {{"entity": "road intersection", "core_concept": "object"}},
    {{"entity": "slope", "core_concept": "field"}},
    {{"entity": "land use", "core_concept": "field"}}
]
---

**Question to process:**
Question: "{query}"
JSON Output(please output only the JSON object, without any additional text or explanation like "the json is", "json format", etc.):
"""
    user_input = [{"role": "user", "content": prompt}]
    response_text = call_llm(user_input)
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        list_str = response_text[start:end]

        parsed = ast.literal_eval(list_str)
        return parsed


def llm_predict_final_concept(query: str):
    """Predicts the final, primary output concept that the user's query aims to generate."""
    prompt = f"""
Your task is to act as a specialized GIScience analyst. From the user's question, identify the single, final spatial concept that the query is asking to produce.

Follow these rules:
1.  Determine the primary output of the required analysis.
2.  Identify both the entity name and its core concept type.
3.  Return a single JSON object with two keys: "entity" and "core_concept".
4.  All strings should be lowercase.
Here are the descriptions of spatial core concepts for your reference:
  - **Location**: Spatial reference describing where something is. Used in spatial distribution and geometry.
  - **Field**: Continuously varying values across space (e.g., elevation, distance, land use). Supports interpolation and aggregation.
  - **Object**: Discrete bounded entities with identity and attributes (e.g., buildings, trees, parks).
  - **Event**: Time-bound spatial occurrences with location and features (e.g., fires, trips).
  - **Network**: Structured spatial relationships among entities (e.g., roads, connections, flows).
  - **Amount**:
    - *Content Amount*: Aggregated values (count, sum, average).
    - *Coverage Amount*: Spatial extent (area, length, cluster size).
  - **Proportion**: Ratio between two amounts (e.g., density, rate), capturing relative quantities.

---
**Example 1:**
Question: "What is the Euclidean distance to recreational sites in Utrecht"
JSON Output:
{{"entity": "euclidean distance", "core_concept": "field"}}
---
**Example 2:**
Question: "How many restaurants are there within a 500 m radius of the Sydney Harbour Bridge in Sydney?"
JSON Output:
{{"entity": "number of restaurants", "core_concept": "conamount"}}
---

**Question to process:**
Question: "{query}"
JSON Output(please output only the JSON object, without any additional text or explanation like "the json is", "json format", etc.):
"""
    user_input = [{"role": "user", "content": prompt}]
    # response_texts = []
    # for temperature in [0.1, 0.3, 0.9]:  # Try multiple temperatures for robustness
    #     response_text = call_llm(user_input,temperature)[0]
    #     response_texts.append(json.loads(response_text))
    response_text = call_llm(user_input)
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        dict_str = response_text[start:end]

        parsed = ast.literal_eval(dict_str)
        return parsed
    return json.loads(response_text)


def llm_propose_next_step(query: str, final_concept: Dict[str, str], known_concepts: List[Dict[str, str]],
                          previous_steps: List[str]) -> Dict[str, Any]:
    """Based on the current state, proposes the next logical transformation step."""
    prompt = f"""
You are an expert GIScientist building a multi-step solution for a user's query. Your task is to determine only the single, most logical **next** step in the transformation path.

- The user's ultimate goal is to generate the concept: **{final_concept}**.
- So far, the known concepts available for use are: **{known_concepts}**.
- The transformation steps already completed are: **{previous_steps}**.

Here are the descriptions of spatial core concepts for your reference:
  - **Location**: Spatial reference describing where something is. Used in spatial distribution and geometry.
  - **Field**: Continuously varying values across space (e.g., elevation, distance, land use). Supports interpolation and aggregation.
  - **Object**: Discrete bounded entities with identity and attributes (e.g., buildings, trees, parks).
  - **Event**: Time-bound spatial occurrences with location and features (e.g., fires, trips).
  - **Network**: Structured spatial relationships among entities (e.g., roads, connections, flows).
  - **Amount**:
    - *Content Amount*: Aggregated values (count, sum, average).
    - *Coverage Amount*: Spatial extent (area, length, cluster size).
  - **Proportion**: Ratio between two amounts (e.g., density, rate), capturing relative quantities.

Based on this precise context, propose the single next transformation step. This step might create an intermediate concept or the final one.
Return a JSON object with two keys:
1. "transformation_rule": A string representing the operation (e.g., "[slope] + [land use] -> [cost surface]").
2. "output_concept": A dictionary for the new concept this step will generate, with "entity" and "core_concept" keys.

---
**Example 1: The First Step of a Complex Problem**
User Question: "What is the least cost route from school to the closest road intersection based on slope and land use in Utrecht"
Final Goal: {{"entity": "least cost route", "core_concept": "field"}}
Known Concepts: [{{"entity": "school", "core_concept": "object"}}, {{"entity": "road intersection", "core_concept": "object"}}, {{"entity": "slope", "core_concept": "field"}}, {{"entity": "land use", "core_concept": "field"}}]
Previous Steps: []
JSON Output:
{{
    "transformation_rule": "[slope] + [land use] -> [cost surface]",
    "output_concept": {{"entity": "cost surface", "core_concept": "field"}}
}}
---
**Example 2: An Intermediate Step in a Multi-Step Problem**
User Question: "What houses are for sale and within 0.5km from the main roads in Utrecht"
Final Goal: {{"entity": "houses", "core_concept": "object"}}
Known Concepts: [{{"entity": "main roads", "core_concept": "object"}}, {{"entity": "houses", "core_concept": "object"}}, {{"entity": "for sale", "core_concept": "objectquality"}}, {{"entity": "distance field", "core_concept": "field"}}]
Previous Steps: ["[main roads] -> [distance field]"]
JSON Output:
{{
    "transformation_rule": "[distance field] -> [boolean field]",
    "output_concept": {{"entity": "boolean field", "core_concept": "field"}}
}}
---

**Question to process:**
User Question: "{query}"
Final Goal: {final_concept}
Known Concepts: {known_concepts}
Previous Steps: {previous_steps}
JSON Output(please output only the JSON object, without any additional text or explanation like "the json is", "json format", etc.):
"""
    user_input = [{"role": "user", "content": prompt}]
    response_text = call_llm(user_input)
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        dict_str = response_text[start:end]

        parsed = ast.literal_eval(dict_str)
        return parsed

def llm_verify_and_refine_step(query: str, proposed_step: dict, retrieved_examples: List[str]) -> dict:
    """
    Compares the proposed step with retrieved examples to both verify/refine it
    and elicit CRITICAL, CONTEXTUAL knowledge from the original user query.
    """
    prompt = f"""
You are a meticulous GIScience reasoning verifier and a strategic knowledge extractor. An AI assistant has proposed a step to solve a problem.

Your task is to:
1.  **Refine the Step**: Analyze the proposed step and the retrieved examples, then produce the final, most correct version of the transformation step.

Return a single JSON object with the following keys:
- "final_transformation_rule": The corrected transformation rule string.
- "final_output_concept": A dictionary for the new concept with "entity" and "core_concept".

---
**Example 1: Eliciting a "Selection Constraint"**
Original User Query: "I'm at Base Hospital Delhi Cantt. Could you help me locate the nearest Gym from these options: [Outdoor Gym, GAME OF FITNESS, The Workout Zone, Exclusive fit gym]?"
Proposed Step: {{"transformation_rule": "[Base Hospital Delhi Cantt] -> [nearest gym]", "output_concept": {{"entity": "nearest gym", "core_concept": "object"}}}}
Retrieved Examples: ["Question: What is the Euclidean distance to recreational sites..."] 

Final Correct Step and Knowledge (JSON):
{{
    "final_transformation_rule": "[Base Hospital Delhi Cantt] -> [nearest gym]",
    "final_output_concept": {{"entity": "nearest gym", "core_concept": "object"}}
}}
---
**Example 2: Eliciting a "Temporal Constraint"**
Original User Query: "I want to visit the Louvre (open 9am-6pm) and the Eiffel Tower. I will start at 4pm."
Proposed Step: {{"transformation_rule": "[my location] -> [route to Louvre]", "output_concept": {{"entity": "route to Louvre", "core_concept": "network"}}}}
Retrieved Examples: ["Question: What is the shortest route from the resort center..."]

Final Correct Step and Knowledge (JSON):
{{
    "final_transformation_rule": "[my location] -> [route to Louvre]",
    "final_output_concept": {{"entity": "route to Louvre", "core_concept": "network"}}
}}
---

**Task to perform:**
Original User Query: "{query}"
Proposed Step: {json.dumps(proposed_step)}
Retrieved Examples from Knowledge Base: {json.dumps(retrieved_examples)}
Final Correct Step and Knowledge (JSON, no extra text like 'the json is', 'json format', etc.):
"""
    user_input = [{"role": "user", "content": prompt}]
    response_text = call_llm(user_input)
    try:
        return json.loads(response_text)
    except:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        dict_str = response_text[start:end]

        parsed = ast.literal_eval(dict_str)
        return parsed

# --- Core SCT-GraphRAG Class with Iterative Framework ---
class SCT_GraphRAG:
    def __init__(self, examples: list):
        self.graph = nx.DiGraph()
        self.examples_data = examples
        self._build_knowledge_graph(examples)

    def _create_descriptive_path_string(self, transformations: List[str], concepts_map: Dict[str, Dict]) -> str:
        """Converts formal transformation rules into a natural language string."""
        descriptive_steps = []
        for rule in transformations:
            inputs, outputs = self._parse_transformation_rule(rule, concepts_map)
            if inputs and outputs:
                descriptive_steps.append(f"combine {', '.join(inputs)} to generate {', '.join(outputs)}")
            elif outputs:
                descriptive_steps.append(f"generate {', '.join(outputs)}")
        return " then ".join(descriptive_steps)

    def _parse_transformation_rule(self, rule: str, concepts_map: Dict[str, Dict]):
        """Parses a transformation rule string to get input/output entity names."""
        parts = rule.split('→')
        try:
            inputs_str, outputs_str = parts[0], parts[1]
        except IndexError:
            return [], []

        def get_names(s: str) -> List[str]:
            ids = re.findall(r'\[(\d+)(?:_u)?\]', s)
            return [concepts_map[id]['name'] for id in ids if id in concepts_map]

        return get_names(inputs_str), get_names(outputs_str)

    def _build_knowledge_graph(self, examples: list):
        """
        Constructs a deeply structured knowledge graph.
        - Concept nodes store both entity and core_concept.
        - Transformation paths are converted to descriptive strings and embedded.
        - Transformation sequence is explicitly modeled with 'PART_OF_PATH' edges.
        """
        print("\n--- Building Deeply Structured Knowledge Graph ---")
        self.concept_name_to_node_id = {}

        for i, ex in enumerate(examples):
            question_node_id = f"Q{i}"
            self.graph.add_node(question_node_id, type='question', content=ex['question'],
                                embedding=get_openai_embedding(ex['question']))

            structured_concepts = []
            for cid, cdata in ex['concepts'].items():
                full_def = cdata.get('full_def', '')
                parts = full_def.split(':', 1)
                entity_name = parts[0].strip()
                core_concept = parts[1].strip().split('(')[0].strip() if len(parts) > 1 else 'unknown'

                structured_concepts.append({'entity': entity_name, 'core_concept': core_concept})

                if entity_name not in self.concept_name_to_node_id:
                    node_id = f"C_{entity_name.replace(' ', '_')}"
                    self.concept_name_to_node_id[entity_name] = node_id
                    # IMPROVEMENT 1: Concept node stores both components
                    self.graph.add_node(node_id, type='concept', entity=entity_name, core_concept=core_concept,
                                        full_def=full_def)

                self.graph.add_edge(question_node_id, self.concept_name_to_node_id[entity_name], type='USES_CONCEPT')

            self.graph.nodes[question_node_id]['structured_concepts'] = structured_concepts

            descriptive_path_str = self._create_descriptive_path_string(ex.get('transformations', []), ex['concepts'])
            if descriptive_path_str:
                self.graph.nodes[question_node_id]['descriptive_path_embedding'] = get_openai_embedding(
                    descriptive_path_str)

            prev_trans_node_id = None
            for j, t_rule in enumerate(ex.get('transformations', [])):
                trans_node_id = f"Q{i}_T{j}"
                self.graph.add_node(trans_node_id, type='transformation', rule=t_rule,
                                    embedding=get_openai_embedding(t_rule))
                self.graph.add_edge(question_node_id, trans_node_id, type='HAS_TRANSFORMATION')

                if prev_trans_node_id:
                    self.graph.add_edge(prev_trans_node_id, trans_node_id, type='PART_OF_PATH')
                prev_trans_node_id = trans_node_id

                inputs, outputs = self._parse_transformation_rule(t_rule, ex['concepts'])
                for name in inputs:
                    if name in self.concept_name_to_node_id: self.graph.add_edge(self.concept_name_to_node_id[name],
                                                                                 trans_node_id, type='INPUT_TO')
                for name in outputs:
                    if name in self.concept_name_to_node_id: self.graph.add_edge(trans_node_id,
                                                                                 self.concept_name_to_node_id[name],
                                                                                 type='OUTPUT_OF')

        print("\n--- Generating Embeddings for Concepts ---")
        self.graph_concept_nodes_info = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'concept':
                embedding = get_openai_embedding(data['entity'])  # Embed the entity name
                self.graph.nodes[node_id]['embedding'] = embedding
                self.graph_concept_nodes_info.append({
                    'id': node_id, 'entity': data['entity'], 'core_concept': data['core_concept'],
                    'embedding': embedding
                })
        print("--- Graph and Embeddings Built Successfully ---\n")

    def _compute_pareto_front(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Computes the Pareto front from a list of candidates with multiple objective scores.
        A candidate is on the front if no other candidate dominates it.
        A candidate 'A' dominates 'B' if 'A' is strictly better in at least one score
        and not worse in any other score.
        """
        pareto_front = []
        for i, cand_i in enumerate(candidates):
            is_dominated = False
            for j, cand_j in enumerate(candidates):
                if i == j:
                    continue

                # Check if cand_j dominates cand_i
                if (cand_j['scores']['question'] >= cand_i['scores']['question'] and
                    cand_j['scores']['concept'] >= cand_i['scores']['concept'] and
                    cand_j['scores']['transform'] >= cand_i['scores']['transform']) and \
                        (cand_j['scores']['question'] > cand_i['scores']['question'] or
                         cand_j['scores']['concept'] > cand_i['scores']['concept'] or
                         cand_j['scores']['transform'] > cand_i['scores']['transform']):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(cand_i)

        return pareto_front

    def _find_start_nodes_by_similarity(self, query_concepts: List[Dict[str, str]], top_n=2) -> List[str]:
        """
        Finds the closest matching concept nodes in the graph by combining
        entity similarity and core concept matching.
        """
        start_node_ids = set()
        for concept in query_concepts:
            query_embedding = get_openai_embedding(concept['entity'])

            similarities = []
            for node_info in self.graph_concept_nodes_info:
                # Calculate entity name similarity
                entity_sim = \
                cosine_similarity(query_embedding.reshape(1, -1), node_info['embedding'].reshape(1, -1))[0][0]
                # Reward exact or similar core concepts
                core_sim_bonus = 1.0 if concept['core_concept'] == node_info['core_concept'] else 0.5

                # Combine scores
                total_sim = entity_sim * core_sim_bonus
                similarities.append((total_sim, node_info['id']))

            similarities.sort(key=lambda x: x[0], reverse=True)
            for _, node_id in similarities[:top_n]:
                start_node_ids.add(node_id)

        return list(start_node_ids)

    def _calculate_entity_similarity(self, query_concepts: List[Dict[str, str]],
                                     example_concepts: List[Dict[str, str]]) -> float:
        """
        Calculates a sophisticated similarity score between two sets of structured concepts.

        This function compares each concept from the query against all concepts in the
        example. The similarity for each pair is a weighted average of:
        1. Semantic similarity of the 'entity' names (using cosine similarity).
        2. Match score of the 'core_concept' types (rewarding exact matches).

        The final score is the average of the best match scores for each query concept,
        ensuring a normalized score between 0 and 1.
        """
        # Handle edge cases where one or both lists are empty.
        if not query_concepts or not example_concepts:
            return 0.0

        # Create maps for easier lookup of core concepts.
        query_concept_map = {c['entity']: c['core_concept'] for c in query_concepts}
        example_concept_map = {c['entity']: c['core_concept'] for c in example_concepts}

        # --- Step 1: Batch Embedding ---
        # Collect all unique entity names from both sets to get embeddings in one go.
        # This is far more efficient than calling the embedding function inside a loop.
        all_entities = list(set(query_concept_map.keys()) | set(example_concept_map.keys()))
        embeddings = {entity: embedding_cache[entity] if entity in embedding_cache else get_openai_embedding(entity) for entity in all_entities}
        
        total_best_match_score = 0

        # --- Step 2: Iterate through each concept in the user's query ---
        for q_entity, q_core in query_concept_map.items():
            best_match_for_this_q_concept = 0

            # --- Step 3: Find the best matching concept in the example ---
            for e_entity, e_core in example_concept_map.items():

                # --- Step 3a: Calculate Entity Name Similarity ---
                # Use cosine similarity between the pre-fetched embeddings.
                entity_sim = cosine_similarity(
                    embeddings[q_entity].reshape(1, -1),
                    embeddings[e_entity].reshape(1, -1)
                )[0][0]

                # --- Step 3b: Calculate Core Concept Match Score ---
                # Give a high score for an exact match, and a partial score for a mismatch.
                # This ensures that 'school:object' is a better match for 'hospital:object'
                # than for 'slope:field'.
                core_sim = 1.0 if q_core == e_core else 0.5

                # --- Step 3c: Combine the scores ---
                # An average gives equal importance to both entity name and its abstract type.
                combined_sim = (entity_sim + core_sim) / 2

                if combined_sim > best_match_for_this_q_concept:
                    best_match_for_this_q_concept = combined_sim

            # Add the best possible score for the current query concept to the total.
            total_best_match_score += best_match_for_this_q_concept

        # --- Step 4: Normalize the final score ---
        # Divide by the number of query concepts to get an average score.
        # This ensures the result is always a value between 0 and 1.
        return total_best_match_score / len(query_concepts)

    def advanced_graph_retrieval(
            self,
            query: str,
            known_concepts: List[Dict[str, str]],
            proposed_step: str,
            top_k=10
    ) -> List[str]:
        """
        The core multi-factor retrieval and ranking engine.
        """
        print("--- Starting Advanced Graph-Based Retrieval ---")

        # --- Stage 1: Candidate Filtering via Graph Traversal ---
        start_node_ids = self._find_start_nodes_by_similarity(known_concepts)

        candidate_q_nodes = set()
        for start_node_id in start_node_ids:
            for pred in self.graph.predecessors(start_node_id):
                if self.graph.nodes[pred].get('type') == 'question': candidate_q_nodes.add(pred)
            for succ in self.graph.successors(start_node_id):
                if self.graph.nodes[succ].get('type') == 'transformation':
                    for q_node in self.graph.predecessors(succ):
                        if self.graph.nodes[q_node].get('type') == 'question': candidate_q_nodes.add(q_node)

        print(f"Found {len(candidate_q_nodes)} candidate examples via graph traversal.")
        if not candidate_q_nodes: return []

        # --- Stage 2: Multi-Factor Scoring and Ranking ---
        query_embedding = get_openai_embedding(query)
        proposed_step_embedding = get_openai_embedding(proposed_step)

        ranked_candidates = []
        scored_candidates = []
        for q_node_id in candidate_q_nodes:
            data = self.graph.nodes[q_node_id]

            # Score 1: Question Similarity
            question_score = cosine_similarity(query_embedding.reshape(1, -1), data['embedding'].reshape(1, -1))[0][0]

            # Score 2: Concept Similarity
            concept_score = self._calculate_entity_similarity(known_concepts, data.get('structured_concepts', []))

            # Score 3: Transformation Step Similarity
            max_transform_score = 0
            for succ in self.graph.successors(q_node_id):
                if self.graph.nodes[succ].get('type') == 'transformation':
                    trans_data = self.graph.nodes[succ]
                    sim = \
                    cosine_similarity(proposed_step_embedding.reshape(1, -1), trans_data['embedding'].reshape(1, -1))[
                        0][0]
                    if sim > max_transform_score: max_transform_score = sim


            example_index = int(q_node_id.replace('Q', ''))
            scored_candidates.append({
                "example_idx": example_index,
                "scores": {
                    'question': question_score,
                    'concept': concept_score,
                    'transform': max_transform_score
                }
            })

        # --- Stage 3: Pareto Front Computation ---
        print(f"Computing Pareto front from {len(scored_candidates)} scored candidates...")
        pareto_candidates = self._compute_pareto_front(scored_candidates)
        print(f"Identified {len(pareto_candidates)} non-dominated candidates on the Pareto front.")

        # Sort the Pareto front by a simple sum of scores for stable ordering, then take top_k
        pareto_candidates.sort(key=lambda x: sum(x['scores'].values()), reverse=True)

        top_candidate_indices = [c['example_idx'] for c in pareto_candidates[:top_k]]
        candidate_examples_text = []
        for i in top_candidate_indices:
            example_data = self.examples_data[i]

            # Format the concepts dictionary into a readable string list
            concepts_str = "\n".join([
                f"- [{cid}] {cdata['full_def']}"
                for cid, cdata in sorted(example_data['concepts'].items())
            ])

            # Construct the full example text
            full_text = (
                f"Question: {example_data['question']}\n"
                f"Concepts:\n{concepts_str}\n"
                f"Transformations: {example_data['transformations']}"
            )
            candidate_examples_text.append(full_text)

        if not candidate_examples_text: return []

        return candidate_examples_text

    def generate_transformation_path_iteratively(self, query: str, max_steps=6, mode="example_knowledge" # or 'concept_transformations_knowledge' or 'full_knowledge'
                                                 ) -> Dict[str, Any]:
        """
        The main query function, now with knowledge accumulation.
        """
        print("--- Step 1: Initial Query Analysis ---")
        query_concepts = llm_decompose_query_structured(query)
        final_concept = llm_predict_final_concept(query)
        print(f"Decomposed Concepts: {query_concepts}")
        print(f"Predicted Final Concept: {final_concept}\n")

        known_concepts = query_concepts
        final_transformation_path = []
        accumulated_knowledge = []
        retrieved_examples_list = []
        for i in range(max_steps):
            print(f"--- Iteration {i + 1}/{max_steps} ---")

            if llm_check_goal_achieved_with_context(query, final_concept, known_concepts, final_transformation_path):
                print("Goal achieved. Terminating reasoning path generation.")
                break

            # print("Proposing next step...")
            proposed_step = llm_propose_next_step(query, final_concept, known_concepts, final_transformation_path)
            # print(f"Proposed Step: {proposed_step}")

            print("Retrieving best example(s) for this step using advanced graph RAG...")
            retrieved_examples = self.advanced_graph_retrieval(
                query,
                known_concepts,
                proposed_step['transformation_rule']
            )

            retrieved_examples_list.extend(retrieved_examples)
            print(f"Retrieved Top Candidates: {retrieved_examples}")
            refinement_result = llm_verify_and_refine_step(query, proposed_step, retrieved_examples)
            final_step = {k: v for k, v in refinement_result.items() if k != 'knowledge_elicitation'}
            elicited_knowledge = refinement_result.get('knowledge_elicitation', '').strip()

            if elicited_knowledge:
                accumulated_knowledge.append(elicited_knowledge)

            final_transformation_path.append(final_step['final_transformation_rule'])
            output_concept = final_step['final_output_concept']
            if not any(c['entity'] == output_concept['entity'] for c in known_concepts):
                known_concepts.append(output_concept)

        if mode == "example_knowledge":
            return "\n".join(retrieved_examples_list), "\n".join(accumulated_knowledge)
        elif mode == "concept_transformations_knowledge":
            return self.format_reasoning_to_text({
                "final_query_analysis": {
                    "initial_concepts": query_concepts,
                    "final_concept_goal": final_concept
                },
                "generated_transformation_path": final_transformation_path,
                "final_known_concepts": known_concepts,
                "accumulated_knowledge": accumulated_knowledge
            })
        else:  # full_knowledge
            return self.format_reasoning_to_text({
                "final_query_analysis": {
                    "initial_concepts": query_concepts,
                    "final_concept_goal": final_concept
                },
                "generated_transformation_path": final_transformation_path,
                "final_known_concepts": known_concepts,
                "accumulated_knowledge": accumulated_knowledge
            }) + "\n\n--- Retrieved Examples ---\n" + "\n\n".join(retrieved_examples_list)


    def format_reasoning_to_text(self, reasoning_result: Dict[str, Any]) -> str:
        """
        Converts the raw reasoning_result JSON into a human-readable, descriptive
        multi-paragraph text block for the final prompt.
        """
        # --- Part 1: Format the Core Concepts ---
        all_concepts = reasoning_result['final_query_analysis']['initial_concepts']
        for concept in reasoning_result['final_known_concepts']:
            if not any(c['entity'] == concept['entity'] for c in all_concepts):
                all_concepts.append(concept)

        concepts_text = "**1. Core Concepts Involved:**\n"
        for concept in all_concepts:
            concepts_text += f"- **{concept['entity']}**: A concept of type '{concept['core_concept']}'.\n"

        # --- Part 2: Format the Transformation Path ---
        path_text = "\n**2. Transformation Path:**\n"
        path_with_knowledge = []
        for i, rule in enumerate(reasoning_result['generated_transformation_path']):
            knowledge = reasoning_result['accumulated_knowledge'][i] if i < len(
                reasoning_result['accumulated_knowledge']) else ""

            path_text += f"\n* **Step {i + 1}: `{rule}`**\n"
            if knowledge:
                path_text += f"    * **Elicited Knowledge**: {knowledge}\n"

        return concepts_text + path_text

    def save(self, filepath: str):
        """
        Saves the entire SCT_GraphRAG instance to a file using pickle.
        This preserves the graph, embeddings, and all helper data structures.
        """
        print(f"\n--- Saving knowledge graph to {filepath} ---")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print("Knowledge graph saved successfully.")

    @staticmethod
    def load(filepath: str) -> 'SCT_GraphRAG':
        """
        Loads an SCT_GraphRAG instance from a pickle file, skipping the build process.
        """
        print(f"--- Loading knowledge graph from {filepath} ---")
        with open(filepath, 'rb') as f:
            rag_system = pickle.load(f)
        print("Knowledge graph loaded successfully.")
        return rag_system
# --- Main Execution Block ---
if __name__ == '__main__':
    # Build the full SCT knowledge graph (443 questions) used for the main
    # results, from the merged corpus (train 309 + test 134) provided by
    # Xu et al. (2023). This produces rag_knowledge_graph.pkl, which map_eval.py
    # and poiqa.py load for Spatial CoT+.
    try:
        with open('data/rag_data/full_corpora.txt', 'r', encoding='utf-8') as f:
            file_content = f.read()
    except FileNotFoundError:
        print("Error: 'full_corpora.txt' not found. Please ensure your data file is in the correct location.")
        exit()
    gis_examples = parse_data(file_content)

    print(f"Building Knowledge Graph from {len(gis_examples)} examples...")
    rag_system = SCT_GraphRAG(gis_examples)
    rag_system.save("rag_knowledge_graph.pkl")
    print("Knowledge Graph built and saved to rag_knowledge_graph.pkl")

    # To reload a previously built graph instead of rebuilding:
    # rag_system = SCT_GraphRAG.load("rag_knowledge_graph.pkl")

    # visualize_graph(rag_system, layout_type='circular', show_details=True)

    # Quick smoke test of the iterative retrieval reasoning:
    test_query = '''What is the slope in Utrecht'''
    reasoning_result = rag_system.generate_transformation_path_iteratively(test_query, mode="concept_transformations_knowledge")
    print(reasoning_result)