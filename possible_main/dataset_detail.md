# Few-shot intent detection 

- few-shot intent detection with/without Out-of-Scope (OOS) intents. 

     Dataset  |	  Description                                           |   #Train   |    #Valid    |  #Test  	|
BANKING77     |	one banking domain with 77 intents 	                |   8622     |     1540     |    3080 	|
CLINC150      | 10 domains and 150 intents 	                        |   15000    |     3000     |    4500 	|
HWU64 	      | personal assistant with 64 intents and several domains 	|   8954     |     1076     |    1076 	|
SNIPS         |	snips voice platform with 7 intents 	                |   13084    |     700      |    700    |
ATIS 	      | airline travel information system 	                |   4478     |     500      |    893    | 	


What is OOS quries:


OOD-OOS: i.e., out-of-domain OOS. General out-of-scope queries which are not supported by the dialog systems, also called out-of-domain OOS. For instance, requesting an online NBA/TV show service in a banking system.


ID-OOS: i.e., in-domain OOS. Out-of-scope queries which are more related to the in-scope intents

ref- https://github.com/jianguoz/Few-Shot-Intent-Detection

### Data structure:

```python
Datasets/
├── BANKING77
│   ├── train
│   ├── train_10
│   ├── train_5
│   ├── valid
│   └── test
├── CLINC150
│   ├── train
│   ├── train_10
│   ├── train_5
│   ├── valid
│   ├── test
│   ├── oos
│       ├──train
│       ├──valid
│       └──test
├── HWU64
│   ├── train
│   ├── train_10
│   ├── train_5
│   ├── valid
│   └── test
├── SNIPS
│   ├── train
│   ├── valid
│   └── test
├── ATIS
│   ├── train
│   ├── valid
│   └── test
├── BANKING77-OOS
│   ├── train
│   ├── valid
│   ├── test
│   ├── id-oos
│   │   ├──train
│   │   ├──valid
│   │   └──test
│   ├── ood-oos
│       ├──valid
│       └──test
├── CLINC-Single-Domain-OOS
│   ├── banking
│   │   ├── train
│   │   ├── valid
│   │   ├── test
│   │   ├── id-oos
│   │   │   ├──valid
│   │   │   └──test
│   │   ├── ood-oos
│   │       ├──valid
│   │       └──test
│   ├── credit_cards
│   │   ├── train
│   │   ├── valid
│   │   ├── test
│   │   ├── id-oos
│   │   │   ├──valid
│   │   │   └──test
│   │   ├── ood-oos
│   │       ├──valid
└── └──     └──test
```
