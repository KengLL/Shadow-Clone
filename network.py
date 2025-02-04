import ast
import json
import zlib
import base64
from collections import defaultdict
import shutil

def parse_and_write_network_files(file_paths):
    """
    Parses network files and create new network file by reading the existing node and edge data, updating node attributes,
    and re-encoding compressed shadow data for each node. Writes updates to a new file instead of the original.

    Parameters:
    - file_paths (list of str): List of paths to the network files to be processed.

    Returns:
    None: Updates are written directly to a new file based on the first file in the list.
    """
    node_info = {}
    agents = defaultdict(lambda: defaultdict(dict))
    all_nodes = set()

    # First pass: read and process existing network data (supports both encoded and non-encoded formats)
    for layer_index, file_path in enumerate(file_paths):
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()

        if not lines:
            continue

        header = lines[0].strip().split()
        directed = header[1] == '1' if len(header) > 1 else False

        found_edges = False
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == 'EDGES':
                found_edges = True
                continue
                
            if not found_edges and layer_index == 0:
                parts = stripped.split(' ', 3)
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    agent_type = int(parts[1])
                    rank = int(parts[2])
                    
                    shadow = {}
                    if len(parts) > 3:
                        try:
                            # ======== Handle both encoded "data" ========
                            attribs = json.loads(parts[3])
                            if 'data' in attribs:  # Decode compressed data
                                encoded_str = attribs['data']
                                compressed = base64.b64decode(encoded_str)
                                json_str = zlib.decompress(compressed).decode('utf-8')
                                existing_shadow = json.loads(json_str).get('shadow', {})
                            
                            # Clean keys (remove spaces in tuples)
                            shadow = {
                                str(layer): {
                                    k.replace(" ", ""): v 
                                    for k, v in layer_data.items()
                                }
                                for layer, layer_data in existing_shadow.items()
                            }
                        except:
                            shadow = {}
                    # ===================================================================================
                    
                    node_info[node_id] = (node_id, agent_type, rank, shadow)
                    all_nodes.add(node_id)

            elif found_edges:
                parts = line.split(' ', 2)
                if len(parts) >= 2:
                    u_id, v_id = map(int, parts[:2])
                    if u_id in node_info and v_id in node_info:
                        u_tuple = node_info[u_id][:3]
                        v_tuple = node_info[v_id][:3]
                        
                        u_str = f"({','.join(map(str, u_tuple))})"
                        v_str = f"({','.join(map(str, v_tuple))})"
                        
                        weight = 1.0
                        if len(parts) > 2:
                            try:
                                attrs = json.loads(parts[2])
                                weight = attrs.get('weight', 1.0)
                            except:
                                pass
                        
                        agents[u_tuple][layer_index][v_tuple] = weight
                        if not directed:
                            agents[v_tuple][layer_index][u_tuple] = weight

    # Generate updated shadow data with compact tuple format
    result = []
    for node_id in sorted(all_nodes):
        uid_tuple = node_info[node_id][:3]
        existing_shadow = node_info[node_id][3]
        
        agent_dict = {}
        for layer in range(len(file_paths)):
            layer_data = {}
            layer_data.update(existing_shadow.get(str(layer), {}))
            layer_data.update({
                f"({','.join(map(str, k))})": v 
                for k, v in agents[uid_tuple].get(layer, {}).items()
            })
            agent_dict[str(layer)] = layer_data
        result.append(agent_dict)

    # Create a copy of the original file to write the updated content
    base_file_path = file_paths[0]
    copied_file_path = f"{base_file_path}_multi"
    shutil.copy(base_file_path, copied_file_path)

    with open(copied_file_path, 'r') as f:
        lines = f.read().splitlines()

    updated_lines = []
    node_index = 0
    found_edges = False

    for line in lines:
        stripped = line.strip()
        if stripped == 'EDGES':
            found_edges = True
            updated_lines.append(line)
            continue
            
        if not found_edges:
            if node_index == 0:  # Header line
                updated_lines.append(line)
            else:
                parts = line.strip().split(' ', 3)
                if len(parts) >= 3 and (node_id := int(parts[0])) in node_info:
                    # Compress and encode shadow data to bypass nested dict restrictions
                    nested_shadow = {"shadow": result[node_index - 1]}
                    json_str = json.dumps(nested_shadow, separators=(',', ':'))
                    compressed = zlib.compress(json_str.encode('utf-8'))
                    encoded_str = base64.b64encode(compressed).decode('utf-8')
                    shadow_data = json.dumps({"data": encoded_str}, separators=(',', ':'))
                    # ===========================================================
                    
                    new_line = f"{parts[0]} {parts[1]} {parts[2]} {shadow_data}"
                    updated_lines.append(new_line)
            node_index += 1
        else:
            updated_lines.append(line)

    with open(copied_file_path, 'w') as f:
        f.write('\n'.join(updated_lines))

    print(f"Updated network data has been written to: {copied_file_path}")

def decompress_and_convert_shadow_data(encoded_str):
    """
    Decompress and convert base64-encoded, zlib-compressed shadow data.

    Parameters:
    - encoded_str (str): A base64-encoded string containing zlib-compressed shadow data.

    Returns:
    - dict: The converted shadow data with layer IDs as integers and connection keys as tuples.
    """
    try:
        # Decode and decompress the shadow data
        compressed = base64.b64decode(encoded_str)
        json_str = zlib.decompress(compressed).decode('utf-8')
        raw_shadow = json.loads(json_str)['shadow']
        
        # Convert string keys to tuples and integer layer IDs
        shadow_data = {
            int(layer): {
                ast.literal_eval(conn_key): weight 
                for conn_key, weight in conns.items()
            }
            for layer, conns in raw_shadow.items()
        }

        return shadow_data
    except Exception as e:
        print(f"Error processing shadow data: {e}")
        return {}

