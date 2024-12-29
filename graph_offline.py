import networkx as nx
import random
zones = ['A', 'B', 'C', 'D', 'E']

def generate_requests(num_requests, zones):
    requests = []
    for _ in range(num_requests):
        pickup_zone = random.choice(zones)
        dropoff_zone = random.choice(zones)
        pickup_time = random.randint(0, 10)  
        dropoff_time = pickup_time + random.randint(1, 3)  
        requests.append({
            'id': len(requests) + 1,
            'pickup_zone': pickup_zone,
            'dropoff_zone': dropoff_zone,
            'pickup_time': pickup_time,
            'dropoff_time': dropoff_time
        })
    return requests

def generate_vehicles(num_vehicles, zones):
    vehicles = []
    for _ in range(num_vehicles):
        initial_zone = random.choice(zones)
        vehicles.append({
            'id': "a"+str(len(vehicles) + 1),
            'initial_zone': initial_zone
        })
    return vehicles

def generate_zone_travel_times(zones):
    travel_times = {}
    for zone1 in zones:
        for zone2 in zones:
            if zone1 != zone2:
                travel_times[(zone1, zone2)] = random.randint(1, 3)  
            else:
                travel_times[(zone1, zone2)] = 0
    return travel_times


def build_vehicle_shareability_network(requests, vehicles, zone_travel_times):
    G = nx.DiGraph()
    
    for req in requests:
        G.add_node(req['id'], 
                   kind="request",
                   pickup_zone=req['pickup_zone'], 
                   dropoff_zone=req['dropoff_zone'],
                   pickup_time=req['pickup_time'], 
                   dropoff_time=req['dropoff_time'])
    
    for vehicle in vehicles:
        G.add_node(vehicle['id'], kind='vehicle', initial_zone=vehicle['initial_zone'])
        for req in requests:
            if vehicle['initial_zone'] == req['pickup_zone']:
                G.add_edge(vehicle['id'], req['id'], weight=0, capacity=1)  
    
    for req1 in requests:
        for req2 in requests:
            if req1['dropoff_time'] <= req2['pickup_time']:
                if req1['dropoff_time'] + zone_travel_times.get((req1['dropoff_zone'], req2['pickup_zone']), float('inf')) <= req2['pickup_time']:
                    G.add_edge(req1['id'], req2['id'], weight=0, capacity=1)  
    
    return G


if __name__ == "__main__":
    num_requests = 10
    num_vehicles = 5
    requests = generate_requests(num_requests, zones)
    vehicles = generate_vehicles(num_vehicles, zones)
    zone_travel_times = generate_zone_travel_times(zones)

    G = build_vehicle_shareability_network(requests, vehicles, zone_travel_times)

    for node in G.nodes(data=True):
        print(f"Node {node[0]}: {node[1]}")
    for edge in G.edges(data=True):
        print(f"Edge {edge[0]}->{edge[1]}: {edge[2]}")
