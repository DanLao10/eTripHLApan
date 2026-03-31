#!/usr/bin/env python3
"""
Generate network architecture flowchart for eTripHLApan
Shows the complete data flow from input to output
"""

import os
import sys

# Create flowchart using graphviz
try:
    from graphviz import Digraph
    use_graphviz = True
except ImportError:
    use_graphviz = False
    print("Graphviz not available, will use matplotlib alternative")

if use_graphviz:
    # Create a directed graph
    dot = Digraph(comment='eTripHLApan Network Architecture', format='pdf')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.5', ranksep='1')
    
    # Set styling
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', 
             fontname='Arial', fontsize='10')
    
    # Input layer
    dot.node('input_pep', 'Peptide\n(8-14 aa)', fillcolor='lightgreen')
    dot.node('input_hla', 'HLA Allele\n(200 aa)', fillcolor='lightgreen')
    
    # Path 1: One-hot encoding
    dot.node('path1_enc', 'One-hot\nEncoding\n(20 dims)', fillcolor='lightyellow')
    dot.node('path1_pep', 'GRU1\n(20→128x2)', fillcolor='lightcyan')
    dot.node('path1_att', 'Attention1\n(256 dims)', fillcolor='lightcyan')
    dot.node('path1_hla', 'GRU2\n(20→128x2)', fillcolor='lightcyan')
    dot.node('path1_att2', 'Attention2\n(256 dims)', fillcolor='lightcyan')
    dot.node('path1_cat', 'Concat\n(512 dims)', fillcolor='lightyellow')
    dot.node('path1_fc', 'FC1\n(512→128)', fillcolor='lightyellow')
    
    # Path 2: Embedding + Numeric encoding
    dot.node('path2_enc', 'Numeric Enc\n+ Embedding\n(6 dims)', fillcolor='lightyellow')
    dot.node('path2_pep', 'GRU3\n(6→128x2)', fillcolor='lightcyan')
    dot.node('path2_att', 'Attention3\n(256 dims)', fillcolor='lightcyan')
    dot.node('path2_hla', 'GRU4\n(6→128x2)', fillcolor='lightcyan')
    dot.node('path2_att2', 'Attention4\n(256 dims)', fillcolor='lightcyan')
    dot.node('path2_cat', 'Concat\n(512 dims)', fillcolor='lightyellow')
    dot.node('path2_fc', 'FC2\n(512→128)', fillcolor='lightyellow')
    
    # Path 3: Physicochemical encoding
    dot.node('path3_enc', 'Physicochemical\nEncoding\n(28 dims)', fillcolor='lightyellow')
    dot.node('path3_pep', 'GRU5\n(28→128x2)', fillcolor='lightcyan')
    dot.node('path3_att', 'Attention5\n(256 dims)', fillcolor='lightcyan')
    dot.node('path3_hla', 'GRU6\n(28→128x2)', fillcolor='lightcyan')
    dot.node('path3_att2', 'Attention6\n(256 dims)', fillcolor='lightcyan')
    dot.node('path3_cat', 'Concat\n(512 dims)', fillcolor='lightyellow')
    dot.node('path3_fc', 'FC3\n(512→128)', fillcolor='lightyellow')
    
    # Fusion layer
    dot.node('fusion', 'Concat 3 Paths\n(384 dims)', fillcolor='lightcoral')
    
    # Final classification
    dot.node('fc_final1', 'FC\n(384→128)', fillcolor='lightyellow')
    dot.node('relu1', 'ReLU', fillcolor='lightgray')
    dot.node('fc_final2', 'FC\n(128→128)', fillcolor='lightyellow')
    dot.node('relu2', 'ReLU', fillcolor='lightgray')
    dot.node('dropout', 'Dropout\n(0.2)', fillcolor='lightgray')
    dot.node('fc_final3', 'FC\n(128→1)', fillcolor='lightyellow')
    dot.node('sigmoid', 'Sigmoid', fillcolor='lightyellow')
    dot.node('output', 'Binding Score\n[0, 1]', fillcolor='lightcoral')
    
    # Edges for Path 1
    dot.edge('input_pep', 'path1_enc')
    dot.edge('input_hla', 'path1_enc')
    dot.edge('path1_enc', 'path1_pep', label='Pep')
    dot.edge('path1_enc', 'path1_hla', label='HLA')
    dot.edge('path1_pep', 'path1_att')
    dot.edge('path1_hla', 'path1_att2')
    dot.edge('path1_att', 'path1_cat')
    dot.edge('path1_att2', 'path1_cat')
    dot.edge('path1_cat', 'path1_fc')
    dot.edge('path1_fc', 'fusion')
    
    # Edges for Path 2
    dot.edge('input_pep', 'path2_enc')
    dot.edge('input_hla', 'path2_enc')
    dot.edge('path2_enc', 'path2_pep', label='Pep')
    dot.edge('path2_enc', 'path2_hla', label='HLA')
    dot.edge('path2_pep', 'path2_att')
    dot.edge('path2_hla', 'path2_att2')
    dot.edge('path2_att', 'path2_cat')
    dot.edge('path2_att2', 'path2_cat')
    dot.edge('path2_cat', 'path2_fc')
    dot.edge('path2_fc', 'fusion')
    
    # Edges for Path 3
    dot.edge('input_pep', 'path3_enc')
    dot.edge('input_hla', 'path3_enc')
    dot.edge('path3_enc', 'path3_pep', label='Pep')
    dot.edge('path3_enc', 'path3_hla', label='HLA')
    dot.edge('path3_pep', 'path3_att')
    dot.edge('path3_hla', 'path3_att2')
    dot.edge('path3_att', 'path3_cat')
    dot.edge('path3_att2', 'path3_cat')
    dot.edge('path3_cat', 'path3_fc')
    dot.edge('path3_fc', 'fusion')
    
    # Edges for final classification
    dot.edge('fusion', 'fc_final1')
    dot.edge('fc_final1', 'relu1')
    dot.edge('relu1', 'fc_final2')
    dot.edge('fc_final2', 'relu2')
    dot.edge('relu2', 'dropout')
    dot.edge('dropout', 'fc_final3')
    dot.edge('fc_final3', 'sigmoid')
    dot.edge('sigmoid', 'output')
    
    # Save the graph
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'eTripHLApan')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'network_architecture')
    dot.render(output_file, cleanup=True)
    
    print(f"✓ Network architecture flowchart saved to: {output_file}.pdf")
    print(f"✓ Also available as: {output_file}.svg")

else:
    # Alternative: Create using matplotlib if graphviz not available
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Helper function to draw boxes
    def draw_box(ax, x, y, width, height, text, color='lightblue'):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Helper function to draw arrows
    def draw_arrow(ax, x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.2, mid_y, label, fontsize=8, style='italic')
    
    # Input layer
    draw_box(ax, 1, 8.5, 0.8, 0.6, 'Peptide\n(8-14 aa)', 'lightgreen')
    draw_box(ax, 1.8, 8.5, 0.8, 0.6, 'HLA Allele\n(200 aa)', 'lightgreen')
    
    # Path 1
    draw_box(ax, 0.5, 7, 1, 0.6, 'Path 1:\nOne-hot (20D)', 'lightyellow')
    draw_box(ax, 0.5, 5.5, 0.9, 0.6, 'GRU1+2\nAttention\n(512D)', 'lightcyan')
    draw_box(ax, 0.5, 4, 0.8, 0.6, 'FC1\n(128D)', 'lightyellow')
    
    # Path 2
    draw_box(ax, 2.5, 7, 1, 0.6, 'Path 2:\nEmbedding+Num (6D)', 'lightyellow')
    draw_box(ax, 2.5, 5.5, 0.9, 0.6, 'GRU3+4\nAttention\n(512D)', 'lightcyan')
    draw_box(ax, 2.5, 4, 0.8, 0.6, 'FC2\n(128D)', 'lightyellow')
    
    # Path 3
    draw_box(ax, 4.5, 7, 1, 0.6, 'Path 3:\nPhysicoChem (28D)', 'lightyellow')
    draw_box(ax, 4.5, 5.5, 0.9, 0.6, 'GRU5+6\nAttention\n(512D)', 'lightcyan')
    draw_box(ax, 4.5, 4, 0.8, 0.6, 'FC3\n(128D)', 'lightyellow')
    
    # Fusion
    draw_box(ax, 2.5, 2.5, 1.5, 0.6, 'Concat 3 Paths\n(384D)', 'lightcoral')
    
    # Final classification
    draw_box(ax, 2.5, 1.2, 1.2, 0.6, 'FC Layers + ReLU\nDropout (0.2)', 'lightyellow')
    draw_box(ax, 2.5, 0.2, 0.8, 0.6, 'Sigmoid\nOutput [0,1]', 'lightcoral')
    
    # Draw arrows
    for x in [0.5, 2.5, 4.5]:
        draw_arrow(ax, 1.2, 8.2, x, 7.3)
        draw_arrow(ax, x, 6.7, x, 5.8)
        draw_arrow(ax, x, 5.2, x, 4.3)
        draw_arrow(ax, x, 3.7, 2.5, 2.8)
    
    draw_arrow(ax, 2.5, 2.2, 2.5, 1.5)
    draw_arrow(ax, 2.5, 0.9, 2.5, 0.5)
    
    ax.set_title('eTripHLApan Network Architecture: Triple Coding Matrix\nPeptide + HLA Allele Binding Prediction', 
                fontsize=14, weight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Input'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor='lightcyan', edgecolor='black', label='GRU + Attention'),
        mpatches.Patch(facecolor='lightcoral', edgecolor='black', label='Fusion/Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'eTripHLApan')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'network_architecture_matplotlib.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Network architecture flowchart saved to: {output_file}")
    plt.close()

print("\n✓ Flowchart generation complete!")
