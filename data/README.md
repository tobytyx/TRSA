# Data File Description


## Meta File
- belief_state.json: Listing all the possibel belief state, used as constraints to do DB search.
- vocab.json: The shared vocabulary of both the predictor and the generator.
- act_ontology.json: Listing the 600+ possible dialog acts, all in a triple format, consiting of domain, function and arguments.


## Data
- delex.json: Baseline 提供的log info, 提供db_pointers
- train, val, test.json: The files are lists of dialogs, every dialog is represented with a dictionary.
```
[
  {
    name: string,         对话名
    turn: int,            对话轮数
    sys: string,          系统delex回复
    sys_orig: string,     系统原始回复
    user: string,         user delex回复
    user_orig: string,    user原始回复
    BS: dict,             dict版的Belief state
    belief_state: list    0-1 vector Belief state
    act:  list            domain-act-argument
    dialogue_action: list 0-1 vector Dialogue action *44, [domain*10, act域*7, argument域*27]
    KB: int               the number of entries in the KB which meet the requirement of current BS constraint
    source: dict          the selected row among the queries results, recovered from the system response.
    history: list         sys, user的历史delex对话
    db_pointer: list      0-1 vector 30位，'restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital'， 每个5位
    }
  }
  ...
]

```
