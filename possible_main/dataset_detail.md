# Few-shot intent detection 

- few-shot intent detection with/without Out-of-Scope (OOS) intents.

![image](https://user-images.githubusercontent.com/31414731/141806826-b3f213b6-b6db-480e-87b7-d1c62b51fb9b.png)

What is OOS quries:

OOD-OOS: i.e., out-of-domain OOS. General out-of-scope queries which are not supported by the dialog systems, also called out-of-domain OOS. For instance, requesting an online NBA/TV show service in a banking system.


ID-OOS: i.e., in-domain OOS. Out-of-scope queries which are more related to the in-scope intents

![image](https://user-images.githubusercontent.com/31414731/141806952-76dc3083-1fe2-4929-9cc0-c51fc253700e.png)

![image](https://user-images.githubusercontent.com/31414731/141807020-66fdc484-8f03-4baf-8a95-558419192739.png)

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
