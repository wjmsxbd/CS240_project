import json
import numpy as np
import networkx as nx
import copy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib
seed = 23
eps = 1e-7
SAMPLE_SIZE = 10
SAMPLE_LENGTH = 10
L = 370
FLEET_SIZE = 5000
NUM_EPOCH_DAY = 24 * 60 // 5
A = 2.5
B = 1.26
C = 0.1
random.seed(seed)
np.random.seed(seed)
probability_zone = [0 for i in range(L)]
# position_vehicles = [random.randint(0,369) for i in range(FLEET_SIZE)] # position of vehicles
idle_vehicles = [0 for i in range(L)] # number of vehicles in each zone
# for vehicle in position_vehicles:
#     idle_vehicles[vehicle] += 1
busy_vehicles = {} # format "epoch": [record01,record02 ...]

history_data = {}

def init_position(probability_zone):
    # for i in range(L):
    #     probability_zone[i] = 1
    pb_zone = np.array(probability_zone,dtype=np.int32)
    normalize_zone = pb_zone / np.sum(pb_zone)
    zone = np.random.choice(L,size=FLEET_SIZE,p=normalize_zone)
    for i in range(len(zone)):
        idle_vehicles[zone[i]] += 1

def UniformSample(data,epoch):
    mod_epoch = (int(epoch)) % NUM_EPOCH_DAY
    if str(mod_epoch) not in data.keys():
        return None
    sample_index = np.random.choice(len(data[str(mod_epoch)]), SAMPLE_SIZE, replace=False)
    sample_data = [copy.deepcopy(data[str(mod_epoch)][idx]) for idx in sample_index]
    for i in range(len(sample_data)):
        pickup_epoch = sample_data[i]['pickup_epoch']
        dropoff_epoch = sample_data[i]['dropoff_epoch']
        sample_data[i]['dropoff_epoch'] = int(epoch) + dropoff_epoch - pickup_epoch
        sample_data[i]['pickup_epoch'] = int(epoch)
    return sample_data

class ProbabilityMatrix:
    def __init__(self):
        self.matrix = np.zeros((L, L),dtype=np.long)
        self.normalization_matrix = np.zeros((L,L),dtype=np.float32)

    def Update(self,data):
        self.matrix += data

    def Sample(self,zone,num_samples):
        self.normalization_matrix[zone] = self.matrix[zone] / np.sum(self.matrix[zone])
        end_zones = np.random.choice(L,size=num_samples,p=self.normalization_matrix[zone])
        
        return end_zones

def InsertRecordInHistory(data,epoch):
    mod_epoch = (int(epoch)) % NUM_EPOCH_DAY
    if str(mod_epoch) not in history_data.keys():
        history_data[str(mod_epoch)] = []
    history_data[str(mod_epoch)].extend(data[epoch])

def GetPossibleIdleVehicles(epoch):
    num = 0
    for i in range(SAMPLE_LENGTH):
        if str(int(epoch) + i + 1) in busy_vehicles.keys():
            num += len(busy_vehicles[str(int(epoch) + i + 1)])
    return num

def GetCurrentEpochBusyVehicles(epoch,zone):
    if epoch not in busy_vehicles.keys():
        return 0
    num = 0
    for i in range(len(busy_vehicles[epoch])):
        if busy_vehicles[epoch][i]['DOLocationID'] == zone:
            num += 1
    return num

def GetID(epoch,zone):
    return epoch * L + zone

def gather_request(request):
    request = sorted(request,key=lambda x:(x['dropoff_epoch'] - x['pickup_epoch'],x['PULocationID'],x['DOLocationID']))
    requests = []
    i = 0
    while i < len(request):
        p = i + 1
        
        while (p < len(request)) and ((request[p]['dropoff_epoch'] - request[p]['pickup_epoch']) == (request[i]['dropoff_epoch'] - request[i]['pickup_epoch'])) and (request[p]['PULocationID'] == request[i]['PULocationID']) and (request[p]['DOLocationID'] == request[i]['DOLocationID']):
            p += 1
        record = copy.deepcopy(request[i])
        record['count'] = p - i
        i = p
        requests.append(record)
    return requests

def GetGraph(epoch,current_request,future_request):
    num_nodes = L * (2 + SAMPLE_LENGTH)
    S = num_nodes
    T = S + 1
    G = nx.DiGraph()
    sum_idle_vehicles = sum(idle_vehicles)
    possible_idle_vehicles = GetPossibleIdleVehicles(epoch)
    # print(f"demand:{sum_idle_vehicles}")
    G.add_node(S,demand=int(-sum_idle_vehicles-possible_idle_vehicles))
    G.add_node(T,demand=int(sum_idle_vehicles+possible_idle_vehicles))
    # S -> current epoch request
    for i in range(L):
        node_id = GetID(0,i)
        G.add_node(node_id,demand=0)
        G.add_edge(S,node_id,capacity=idle_vehicles[i],weight=0)

    # future epoch request
    for i in range(SAMPLE_LENGTH):
        for j in range(L):
            node_id = GetID(i+1,j)
            demand = GetCurrentEpochBusyVehicles(str(int(epoch)+i+1),j)
            G.add_node(node_id,demand=0)
            if demand:
                G.add_edge(S,node_id,capacity=demand,weight=0)

    # q + L + 1
    for i in range(L):
        node_id = GetID(SAMPLE_LENGTH+1,i)
        G.add_node(node_id,demand=0)
        G.add_edge(node_id,T,weight=0)

    # epoch -> epoch + 1
    for i in range(SAMPLE_LENGTH+1):
        for j in range(L):
            node_id = GetID(i,j)
            nxt_node_id = GetID(i+1,j)
            G.add_edge(node_id,nxt_node_id,weight=0)

    # current request
    requests = gather_request(current_request)
    for i in range(len(requests)):
        start = requests[i]['PULocationID']
        end = requests[i]['DOLocationID']
        count = requests[i]['count']
        time = requests[i]['dropoff_epoch'] - requests[i]['pickup_epoch'] + 1
        if time > SAMPLE_LENGTH + 1:
            time = SAMPLE_LENGTH + 1
        node_id = GetID(0,start)
        nxt_node_id = GetID(time,end)
        G.add_edge(node_id,nxt_node_id,capacity=count,weight=-1)
    
    # future request
    for i in range(SAMPLE_LENGTH):
        if future_request[i] is None:
            continue
        requests = gather_request(future_request[i])
        for j in range(len(requests)):
            start = requests[j]['PULocationID']
            end = requests[j]['DOLocationID']
            count = requests[j]['count']
            time = requests[j]['dropoff_epoch'] - requests[j]['pickup_epoch'] + 1
            node_id = GetID(i+1,start)
            nxt_node_id = GetID(min(i+1+time,SAMPLE_LENGTH + 1),end)
            G.add_edge(node_id,nxt_node_id,capacity=count,weight=-1)

    return G,S,T

def GetGraph2(epoch,current_request,future_request):
    num_nodes = L * (2 + SAMPLE_LENGTH)
    S = num_nodes
    T = S + 1
    G = nx.DiGraph()
    sum_idle_vehicles = sum(idle_vehicles)
    possible_idle_vehicles = GetPossibleIdleVehicles(epoch)
    print(f"demand:{sum_idle_vehicles}")
    G.add_node(S)
    G.add_node(T)
    # S -> current epoch request
    for i in range(L):
        node_id = GetID(0,i)
        G.add_node(node_id)
        G.add_edge(S,node_id,capacity=idle_vehicles[i],weight=0)

    # future epoch request
    for i in range(SAMPLE_LENGTH):
        for j in range(L):
            node_id = GetID(i+1,j)
            demand = GetCurrentEpochBusyVehicles(str(int(epoch)+i+1),j)
            G.add_node(node_id,)
            G.add_edge(S,node_id,capacity=demand,weight=0)

    # q + L + 1
    for i in range(L):
        node_id = GetID(SAMPLE_LENGTH+1,i)
        G.add_node(node_id)
        G.add_edge(node_id,T,weight=0)

    # epoch -> epoch + 1
    for i in range(SAMPLE_LENGTH+1):
        for j in range(L):
            node_id = GetID(i,j)
            nxt_node_id = GetID(i+1,j)
            G.add_edge(node_id,nxt_node_id,weight=0)

    # current request
    requests = gather_request(current_request)
    for i in range(len(requests)):
        start = requests[i]['PULocationID']
        end = requests[i]['DOLocationID']
        count = requests[i]['count']
        time = requests[i]['dropoff_epoch'] - requests[i]['pickup_epoch'] + 1
        if time > SAMPLE_LENGTH + 1:
            time = SAMPLE_LENGTH + 1
        node_id = GetID(0,start)
        nxt_node_id = GetID(time,end)
        G.add_edge(node_id,nxt_node_id,capacity=count,weight=-1)
    
    # future request
    for i in range(SAMPLE_LENGTH):
        if future_request[i] is None:
            continue
        requests = gather_request(future_request[i])
        for j in range(len(requests)):
            start = requests[j]['PULocationID']
            end = requests[j]['DOLocationID']
            count = requests[j]['count']
            time = requests[j]['dropoff_epoch'] - requests[j]['pickup_epoch'] + 1
            node_id = GetID(i+1,start)
            nxt_node_id = GetID(min(i+1+time,SAMPLE_LENGTH + 1),end)
            G.add_edge(node_id,nxt_node_id,capacity=count,weight=-1)

    return G,S,T

def GetSolution(flowDict,epoch):
    Solutions = np.zeros((L,L),dtype=np.long)
    # current request
    for i in range(L):
        node_id = GetID(epoch,i)
        nxt_epoch_node_id = GetID(epoch+1,i)
        for nxt_node_id in flowDict[node_id].keys():
            if nxt_node_id == nxt_epoch_node_id:
                continue
            Solutions[i,nxt_node_id % L] += flowDict[node_id][nxt_node_id]
    return Solutions

def Dispatch(pm:ProbabilityMatrix,current_request):
    request_matrix = [[[] for i in range(L)] for j in range(L)]
    for request in current_request:
        request_matrix[request['PULocationID']][request['DOLocationID']].append(request)
    # for i in range(L):
    #     for j in range(L):
    #         if request_matrix[i][j] != []:
    #             request_matrix[i][j] = sorted(request_matrix[i][j],key=lambda x:-(((x['dropoff_epoch'] - x['pickup_epoch']))) )
    cnt = 0
    award = 0
    for source_zone in range(L):
        if np.sum(pm.matrix[source_zone]) == 0:
            continue
        target_zone = pm.Sample(source_zone,idle_vehicles[source_zone])
        for zone in target_zone:
            if len(request_matrix[source_zone][zone]):
                request = request_matrix[source_zone][zone][-1]
                if str(request['dropoff_epoch']) not in busy_vehicles.keys():
                    busy_vehicles[str(request['dropoff_epoch'])] = []
                busy_vehicles[str(request['dropoff_epoch'])].append(request)
                idle_vehicles[request['PULocationID']] -= 1
                award += request['total_amount']
                request_matrix[source_zone][zone].pop()
                cnt += 1
    return cnt,award

def Dispatch2(solution,current_request):
    request_matrix = [[[] for i in range(L)] for j in range(L)]
    for request in current_request:
        request_matrix[request['PULocationID']][request['DOLocationID']].append(request)
    for i in range(L):
        for j in range(L):
            if request_matrix[i][j] != []:
                request_matrix[i][j] = sorted(request_matrix[i][j],key=lambda x:-(x['dropoff_epoch'] - x['pickup_epoch']))
    cnt = 0
    award = 0
    for i in range(L):
        for j in range(L):
            while solution[i][j]:
                request = request_matrix[i][j][-1]
                if str(request['dropoff_epoch']) not in busy_vehicles.keys():
                    busy_vehicles[str(request['dropoff_epoch'])] = []
                busy_vehicles[str(request['dropoff_epoch'])].append(request)
                idle_vehicles[request['PULocationID']] -= 1
                request_matrix[i][j].pop()
                award += request['total_amount']
                solution[i][j] -= 1
                cnt += 1
    return cnt,award
    

def ReleaseBusyVehicle(epoch):
    del_keys = []
    busy_vehicles_keys = sorted(busy_vehicles.keys(),key=lambda x: int(x))
    for key in busy_vehicles_keys:
        if int(key) <= int(epoch):
            requests = busy_vehicles[key]
            for request in requests:
                idle_vehicles[request['DOLocationID']] += 1
            del_keys.append(key)
        else:
            break
    for key in del_keys:
        del busy_vehicles[key]

def Solve(data):
    epochs = sorted(data.keys(),key=lambda x:int(x))
    tqdm_bar = tqdm(enumerate(epochs),total=len(epochs))
    day_RSR = []
    day_Reward = []
    all_len = 0
    all_cnt = 0
    all_reward = 0
    for _,epoch in tqdm_bar:
        if int(epoch) < NUM_EPOCH_DAY:
            InsertRecordInHistory(data,epoch)
            current_request = data[epoch]
            for request in current_request:
                probability_zone[request['PULocationID']] += 1
            continue

        if int(epoch) % NUM_EPOCH_DAY == 0 and epoch != '288':
            day_RSR.append(all_cnt / (all_len))
            day_Reward.append(all_reward)
            all_cnt = 0
            all_len = 0
            all_reward = 0
            init_position(probability_zone)
        elif epoch == '288':
            init_position(probability_zone)
        ReleaseBusyVehicle(epoch)
        future_request = []
        current_request = data[epoch]
        for request in current_request:
            probability_zone[request['PULocationID']] += 1
        InsertRecordInHistory(data,epoch)
        for i in range(SAMPLE_LENGTH):
            future_request.append(UniformSample(history_data,str(1+i+int(epoch))))
        Graph,S,T = GetGraph(epoch,current_request,future_request)
        flowDict = nx.max_flow_min_cost(Graph,S,T)
        min_cost = nx.cost_of_flow(Graph,flowDict)
        pm = ProbabilityMatrix()
        for i in range(SAMPLE_LENGTH+1):
            solutions = GetSolution(flowDict,epoch=i)
            pm.Update(solutions)
        cnt,award = Dispatch(pm,current_request)
        # cnt,award = Dispatch2(GetSolution(flowDict,epoch=0),current_request)
        RSR = cnt / len(current_request)
        all_cnt += cnt
        all_len += len(current_request)
        AWARD = award / 10000
        all_reward += AWARD
        tqdm_bar.set_postfix(min_cost=-min_cost,RSR=RSR,AWARD=AWARD,requests=len(current_request),cnt=cnt,step=_)
    day_RSR.append(all_cnt / (all_len))
    day_Reward.append(all_reward)
    print(day_RSR)
    print(day_Reward)

    
    


if __name__ == "__main__":
    # data = joblib.load('requests.pth')
    # print(data)
    with open("request_day1_2.json", "r") as f:
        data = json.load(f)

    Solve(data)
    
    