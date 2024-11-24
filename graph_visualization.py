import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from graph_structure import GraphStructure

class InteractiveGraph:
    def __init__(self, hardware_graph):
        self.G = hardware_graph.G
        self.edges = hardware_graph.edges
        self.enclosures = hardware_graph.enclosures
        self.pos = graphviz_layout(self.G, prog='dot', args='-Goverlap=scale -Gnodesep=22 -Gpad=5')
        self.selected_node = None
        self.selected_node = None
        self.fig, self.ax = plt.subplots()
        self.annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(20, 20),
            textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->")
        )
        self.annotation.set_visible(False)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.center_graph()
        self.draw_graph(hardware_graph)

    def center_graph(self):
        pos = self.pos
        x_coords = [x for x, y in pos.values()]
        y_coords = [y for x, y in pos.values()]
        x_center = (max(x_coords) + min(x_coords)) / 2
        y_center = (max(y_coords) + min(y_coords)) / 2

        for node in pos:
            pos[node] = (pos[node][0] * 8 - x_center, pos[node][1] * 8 - y_center)
        
        self.pos = pos

    def draw_graph(self, hardware_graph):
        self.ax.clear()
        edge_colors = ['red' if weight.endswith('G') else 'blue' for _, _, weight in self.edges]
        edge_weights = [self.G[u][v]['capacity'] / 10**9 for u, v in self.G.edges()]  # Adjust the edge weight for better visualization

        nx.draw(self.G, pos=self.pos, with_labels=True, node_color='skyblue', node_size=100, 
                edge_color=edge_colors, width=[w / 10 for w in edge_weights], font_size=5, font_weight='bold', 
                arrows=True, ax=self.ax)

        # Draw edge labels (capacities)
        edge_labels = {(u, v): f"{self.G[u][v]['label']}" for u, v in self.G.edges() if 'label' in self.G[u][v]}
        nx.draw_networkx_edge_labels(self.G, pos=self.pos, edge_labels=edge_labels, ax=self.ax)

        # Draw NVMeEnclosure boxes
        self.draw_enclosures()

        self.fig.canvas.draw()

    def draw_enclosures(self):
        colors = ['lightgreen', 'lightcoral']
        for idx, (enclosure, nodes) in enumerate(self.enclosures.items()):
            x_coords = [self.pos[node][0] for node in nodes if node in self.pos]
            y_coords = [self.pos[node][1] for node in nodes if node in self.pos]
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Make the box larger to fully contain the nodes
                padding = 22 # Increased padding for larger box
                self.ax.add_patch(Rectangle((x_min-padding, y_min-padding), x_max-x_min+2*padding, y_max-y_min+2*padding, fill=True, color=colors[idx % len(colors)], alpha=0.3))
                self.ax.text((x_min + x_max) / 2, y_min - padding, enclosure, horizontalalignment='center', verticalalignment='top', fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.6))

    def update_annotation(self, edge):
        self.annotation.xy = ((self.pos[edge[0]][0] + self.pos[edge[1]][0]) / 2, (self.pos[edge[0]][1] + self.pos[edge[1]][1]) / 2)
        self.annotation.set_text(f"{edge[0]} -> {edge[1]}: {self.G.edges[edge]['capacity']}")
        self.annotation.get_bbox_patch().set(facecolor='yellow', alpha=0.8)

    def on_hover(self, event):
        vis = self.annotation.get_visible()
        if event.inaxes == self.ax:
            for edge in self.G.edges():
                if edge[0] in self.pos and edge[1] in self.pos:
                    pos1 = self.pos[edge[0]]
                    pos2 = self.pos[edge[1]]
                    if self.is_on_edge(event, pos1, pos2):
                        self.update_annotation(edge)
                        self.annotation.set_visible(True)
                        self.fig.canvas.draw_idle()
                        return
        if vis:
            self.annotation.set_visible(False)
            self.fig.canvas.draw_idle()

    def is_on_edge(self, event, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        distance = abs((y2 - y1) * event.xdata - (x2 - x1) * event.ydata + x2 * y1 - y2 * x1) / ((y2 - y1)**2 + (x2 - x1)**2)**0.5
        return distance < 0.03

    def on_click(self, event):
        pass

    def on_release(self, event):
        pass

    def on_motion(self, event):
        pass

if __name__ == "__main__":
    file_path = 'graph.json'
    edges, enclosures, mttfs, mtrs, _ = GraphStructure.parse_input_from_json(file_path)
    hardware_graph = GraphStructure(edges, enclosures, mttfs, mtrs)
    interactive_graph = InteractiveGraph(hardware_graph)
    plt.show()
