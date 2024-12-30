import networkx as nx
import joblib
from graph_offline import build_vehicle_shareability_network, generate_requests, generate_vehicles, generate_zone_travel_times
import matplotlib.pyplot as plt

class OfflineSolver():
    def __init__(self, requests, vehicles, zone_travel_times):
        self.G = build_vehicle_shareability_network(requests, vehicles, zone_travel_times)
        self.source = 's'
        self.sink = 't'
        self.G.add_node(self.source, demand=len(vehicles), kind="source")  
        self.G.add_node(self.sink, demand=-len(vehicles), kind="target")  
        
        for vehicle in vehicles:
            self.G.add_edge(self.source, vehicle['id'], weight=0, capacity=1)
        #
        for req in requests:
            self.G.add_edge(req['id'], self.sink, weight=0, capacity=1)
        # visualize graph
        pos = nx.spring_layout(self.G)
        pos = self.calculate_layout(pos, self.G)
        self.pos = pos
        self.draw_graph(pos)
    def calculate_layout(self, pos, G):
        pos[self.source] = (0, 2.5)
        vehicle_y = 1.5
        vehicle_x = -1
        for vehicle in G.nodes(data=True):
            if vehicle[1].get('kind') == 'vehicle':
                pos[vehicle[0]] = (vehicle_x, vehicle_y)
                vehicle_x += 0.5
        pos[self.sink] = (0, -2)
        return pos
    

    def draw_graph(self, pos):
        plt.figure(figsize=(12, 8))  
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', ax=plt.gca())
        
        edge_labels = {(u, v): f"{d['capacity']}" for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title("Vehicle Shareability Network")
        plt.show()
    def draw_max_flow(self, pos):
        plt.figure(figsize=(12, 8))  
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', ax=plt.gca())
        
        edge_labels = {(u, v): f"{d['weight']}" for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title("Vehicle Shareability Network")
        plt.show()
               
    def run(self):
        flowValue, flowDict = nx.maximum_flow(self.G, self.source, self.sink)
        print("Flow Value:", flowValue)
        
        for node in flowDict:
            for neighbor in flowDict[node]:
                for u, v, d in self.G.edges(data=True):
                    if u == node and v == neighbor:
                        d['weight'] = flowDict[node][neighbor]
        self.draw_max_flow(self.pos)
        

def load_requests():
    return joblib.load("requests.pth")
def load_zone_travel_times():
    return joblib.load("zone_travel_times.pth")
def load_zones(zone_travel_times):
    zones = set()
    for zone1, zone2 in zone_travel_times:
        zones.add(zone1)
        zones.add(zone2)
    return list(zones)
if __name__ == "__main__":
    requests = load_requests()
    zone_travel_times = load_zone_travel_times()
    zones = load_zones(zone_travel_times)
    vehicles = generate_vehicles(10000, zones)
    solver = OfflineSolver(requests, vehicles, zone_travel_times)
    solver.run()

# if __name__ == "__main__":
#     num_requests = 10
#     num_vehicles = 5
#     zones = ['A', 'B', 'C', 'D', 'E']
    
#     requests = generate_requests(num_requests, zones)
#     vehicles = generate_vehicles(num_vehicles, zones)
#     zone_travel_times = generate_zone_travel_times(zones)
    
#     solver = OfflineSolver(requests, vehicles, zone_travel_times)
#     solver.run()
