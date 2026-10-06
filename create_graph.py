import json

# 1. Load the entities we just extracted
with open("entities.json", "r") as f:
    entities = json.load(f)

# 2. Initialize our Graph structure
# A graph has "Nodes" (the dots/things) and "Edges" (the lines/connections)
graph = {
    "nodes": [],
    "edges": []
}

# 3. Add the main "Patient" node
graph["nodes"].append({
    "id": "Patient_01",
    "type": "Person",
    "label": "Patient"
})

# 4. Add Symptoms and connect them to the Patient
for symptom in entities["symptoms"]:
    symptom_id = f"Symptom_{symptom}"
    
    # Add the symptom as a node (a dot)
    graph["nodes"].append({
        "id": symptom_id,
        "type": "Symptom",
        "label": symptom
    })
    
    # Add an edge (a line) connecting Patient -> Symptom
    graph["edges"].append({
        "from": "Patient_01",
        "to": symptom_id,
        "relationship": "HAS_SYMPTOM"
    })

# 5. Add Duration 
if entities["duration"]:
    duration_text = entities["duration"][0] # "Three days"
    duration_id = f"Duration_{duration_text.replace(' ', '_')}"
    
    graph["nodes"].append({
        "id": duration_id,
        "type": "Time",
        "label": duration_text
    })
    
    # Connect the first symptom (Fever) to the duration
    first_symptom_id = f"Symptom_{entities['symptoms'][0]}"
    graph["edges"].append({
        "from": first_symptom_id,
        "to": duration_id,
        "relationship": "LASTS_FOR"
    })

# 6. Add Measurements (e.g., "Fever: 101")
for measurement in entities["measurements"]:
    if ":" in measurement:
        symptom_name, value = measurement.split(": ")
        value_id = f"Value_{value}"
        
        graph["nodes"].append({
            "id": value_id,
            "type": "Measurement",
            "label": value
        })
        
        symptom_node_id = f"Symptom_{symptom_name}"
        graph["edges"].append({
            "from": symptom_node_id,
            "to": value_id,
            "relationship": "HAS_VALUE"
        })

# 7. Save the graph to a file
with open("medical_graph.json", "w") as f:
    json.dump(graph, f, indent=2)

print("=== KNOWLEDGE GRAPH CREATED ===")
print(f"Total Nodes (Things): {len(graph['nodes'])}")
print(f"Total Edges (Connections): {len(graph['edges'])}")
print("\n✅ Graph saved to medical_graph.json")