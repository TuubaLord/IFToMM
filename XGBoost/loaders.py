import pandas as pd
import numpy as np

EVENT_PATH = "../CARE_To_Compare/Wind Farm C/datasets/"
EVENT_INFO = "../CARE_To_Compare/Wind Farm C/event_info.csv"

def load_event_for_training(event_n):
    load_event(event_n)


def load_event_for_testing(event_n):
    pass



def load_event(event_n):
    events = pd.read_csv(EVENT_INFO, sep=";")
    event_cls = None
    event_type = events[events["event_id"] == event_n]["event_label"].values[0]
    if event_type == "anomaly":
        event_cls = 1
    else:
        event_cls = 0
    
    event = pd.read_csv(f"{EVENT_PATH}{event_n}.csv", sep=";")
    y = np.where(
        event['train_test'] == "train",
        0.0,  event_cls)    
    return event, y


event, y  = load_event(1)
print(event, y)